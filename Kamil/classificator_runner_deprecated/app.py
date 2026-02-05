import streamlit as st
import os
import pandas as pd
from Kamil.classificator_runner.data_preprocessor import load_csv_from_path, load_csv_from_bytes, split_and_preprocess, select_features_label, list_columns, get_available_features
from Kamil.classificator_runner.classificator_runner import run_classifier_experiment, run_classifier_experiment_with_cv, load_and_display_experiment
from Kamil.classificator_runner.experiment_loader import show_experiment_loader, get_classifier_params_from_session
from Kamil.classificator_runner.ui_helpers import display_traditional_results, display_cv_results

st.set_page_config(page_title="Classifier Selector + Runner", layout="wide")
st.title("Classifier Selector — z trenowaniem i zapisem eksperymentów")

st.write("Aplikacja do testowania klasyfikatorów z pełnym pipeline'em: preprocessing → trenowanie → ewaluacja → zapis wyników.")

# --- NOWA SEKCJA: Wczytywanie eksperymentów ---
experiment_loaded = show_experiment_loader()

if st.session_state.get('experiment_loaded'):
    st.info(f"🔄 Wczytano konfigurację z eksperymentu: {st.session_state.get('loaded_experiment_file', 'N/A')}")

st.divider()

# --- Data import section (PRZENIESIONE NA POCZĄTEK) ---
st.header("1) Wybierz plik CSV")

# Sprawdź czy folder input_files istnieje, jeśli nie - utwórz go
input_folder = "input_files"
if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    st.info(f"📁 Utworzono folder `{input_folder}` - umieść w nim pliki CSV.")

# Szukaj plików CSV w folderze input_files
csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
selected_local = None

if csv_files:
    st.success(f"📁 Znaleziono {len(csv_files)} plików CSV w folderze `{input_folder}/`")
    
    # Próbuj dopasować plik z wczytanego eksperymentu LUB z ostatnio używanego pliku
    default_file_idx = 0
    
    # Pierwszeństwo: wczytany eksperyment
    if st.session_state.get('experiment_loaded'):
        loaded_filename = st.session_state.get('dataset_info', {}).get('filename')
        if loaded_filename in csv_files:
            default_file_idx = csv_files.index(loaded_filename) + 1  # +1 bo pierwszy element to "- brak wyboru -"
    # Drugie pierwszeństwo: ostatnio używany plik (z poprzedniego eksperymentu)
    elif 'last_used_csv_file' in st.session_state and st.session_state['last_used_csv_file'] in csv_files:
        default_file_idx = csv_files.index(st.session_state['last_used_csv_file']) + 1
    
    selected_local = st.selectbox(f"Pliki CSV w folderze `{input_folder}/`:", 
                                 options=["- brak wyboru -"] + csv_files,
                                 index=default_file_idx)
else:
    st.warning(f"⚠️ Brak plików CSV w folderze `{input_folder}/`")
    st.info(f"💡 Umieść pliki .csv w folderze `{input_folder}/` i odśwież stronę.")

uploaded_file = st.file_uploader("Albo wgraj plik CSV", type=['csv'])

df = None
dataset_info = {}

if selected_local and selected_local != "- brak wyboru -":
    # Załaduj plik z folderu input_files
    file_path = os.path.join(input_folder, selected_local)
    df = load_csv_from_path(file_path)
    dataset_info = {"source": "local_file", "filename": selected_local, "full_path": file_path}
    # ZAPISZ OSTATNIO UŻYWANY PLIK
    st.session_state['last_used_csv_file'] = selected_local
elif uploaded_file is not None:
    df = load_csv_from_bytes(uploaded_file)
    dataset_info = {"source": "uploaded_file", "filename": uploaded_file.name}
    # ZAPISZ INFORMACJĘ O UPLOADED FILE
    st.session_state['last_used_csv_file'] = uploaded_file.name

# JEŚLI BRAK PLIKU, ALE MAMY WYNIKI - POKAŻ JE I POZWÓL DALEJ
if df is None:
    if 'last_experiment_data' in st.session_state:
        st.warning("⚠️ Nie wybrano pliku CSV. Wyświetlane są ostatnie wyniki eksperymentu.")
        
        # Pokaż sekcję wyników na górze
        st.divider()
        st.header("📊 Ostatnie wyniki eksperymentu")
        
        last_data = st.session_state['last_experiment_data']
        last_filename = st.session_state.get('last_experiment_filename', 'N/A')
        last_method = st.session_state.get('last_experiment_method', 'N/A')
        
        exp_info = last_data['experiment_info']
        st.info(f"📁 **Plik:** {last_filename} | 🔧 **Metoda:** {last_method} | 🤖 **Klasyfikator:** {exp_info['classifier_name']} | 📅 **Data:** {exp_info['timestamp']}")
        
        # Przycisk do wyczyszczenia wyników
        if st.button("🗑️ Wyczyść wyniki", help="Usuwa wyświetlane wyniki z ekranu"):
            for key in ['last_experiment_data', 'last_experiment_filename', 'last_experiment_method', 'last_used_csv_file']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # Wyświetl wyniki
        if last_method == "Train/Test Split":
            results = last_data['results']
            display_traditional_results(results)
        elif last_method == "Cross-Validation":
            cv_results = last_data['results']['cross_validation']
            display_cv_results(cv_results)
        else:
            traditional_results = last_data['results']['traditional']
            cv_results = last_data['results']['cross_validation']
            
            st.subheader("🔄 Wyniki Cross-Validation")
            display_cv_results(cv_results)
            
            st.subheader("📊 Wyniki Train/Test Split")
            display_traditional_results(traditional_results)
        
        st.divider()
        st.info(f"👆 Umieść plik CSV w folderze `{input_folder}/` lub wgraj plik powyżej, aby przeprowadzić nowy eksperyment.")
        st.stop()
    else:
        st.info(f"Nie wybrano pliku. Umieść pliki CSV w folderze `{input_folder}/` lub wgraj nowy plik.")
        st.stop()

# Dodaj informacje o datasecie
dataset_info.update({
    "shape": list(df.shape),
    "columns": list(df.columns),
    "dtypes": df.dtypes.astype(str).to_dict()
})

# Pokaż informację o źródle danych
if dataset_info["source"] == "local_file":
    st.success(f"✅ Wczytano plik: `{input_folder}/{dataset_info['filename']}`")
else:
    st.success(f"✅ Wczytano przesłany plik: `{dataset_info['filename']}`")

st.write("**Podgląd danych:**")
st.dataframe(df.head())
st.write(f"Kształt danych: {df.shape}")

st.divider()

# --- Model selection (PRZENIESIONE TUTAJ) ---
st.header("2) Wybór klasyfikatora i parametrów")

# Użyj wartości z session_state jeśli eksperyment został wczytany
default_clf = st.session_state.get('clf_name', 'RandomForest')
clf_name = st.selectbox("Wybierz klasyfikator:", ("RandomForest", "SVM", "GradientBoosting", "KNN"), 
                       index=["RandomForest", "SVM", "GradientBoosting", "KNN"].index(default_clf))

st.write("**Wybrany klasyfikator:**", clf_name)

# Parametry klasyfikatora - użyj wczytanych wartości
loaded_params = get_classifier_params_from_session(clf_name)
params = {}
col1, col2 = st.columns(2)

with col1:
    if clf_name == "RandomForest":
        params['n_estimators'] = st.number_input("n_estimators", min_value=1, value=loaded_params.get('n_estimators', 100))
        params['max_depth'] = st.number_input("max_depth (0 = None)", min_value=0, value=0 if loaded_params.get('max_depth') is None else loaded_params.get('max_depth', 0))
        params['random_state'] = st.number_input("random_state", min_value=0, value=loaded_params.get('random_state', 42))
        if params['max_depth'] == 0:
            params['max_depth'] = None
            
    elif clf_name == "SVM":
        params['C'] = st.number_input("C", min_value=0.01, value=loaded_params.get('C', 1.0))
        kernel_options = ("rbf", "linear", "poly", "sigmoid")
        default_kernel_idx = 0
        if loaded_params.get('kernel') in kernel_options:
            default_kernel_idx = kernel_options.index(loaded_params['kernel'])
        params['kernel'] = st.selectbox("kernel", kernel_options, index=default_kernel_idx)
        params['random_state'] = st.number_input("random_state", min_value=0, value=loaded_params.get('random_state', 42))
        
    elif clf_name == "GradientBoosting":
        params['n_estimators'] = st.number_input("n_estimators", min_value=1, value=loaded_params.get('n_estimators', 100))
        params['learning_rate'] = st.number_input("learning_rate", min_value=0.01, value=loaded_params.get('learning_rate', 0.1))
        params['max_depth'] = st.number_input("max_depth", min_value=1, value=loaded_params.get('max_depth', 3))
        params['random_state'] = st.number_input("random_state", min_value=0, value=loaded_params.get('random_state', 42))
        
    elif clf_name == "KNN":
        params['n_neighbors'] = st.number_input("n_neighbors", min_value=1, value=loaded_params.get('n_neighbors', 5))
        weights_options = ("uniform", "distance")
        default_weights_idx = 0
        if loaded_params.get('weights') in weights_options:
            default_weights_idx = weights_options.index(loaded_params['weights'])
        params['weights'] = st.selectbox("weights", weights_options, index=default_weights_idx)

with col2:
    st.write("**Parametry klasyfikatora:**")
    st.json(params)

st.divider()

# --- Feature and label selection ---
st.header("3) Wybór cech i kolumny klasy")
cols = list_columns(df)

# Użyj wczytanej wartości label_col
loaded_label = st.session_state.get('label_col')
default_label_idx = 0
if loaded_label and loaded_label in cols:
    default_label_idx = cols.index(loaded_label) + 1  # +1 bo pierwszy element to None

label_col = st.selectbox("Wybierz kolumnę klasy", options=[None] + cols, index=default_label_idx)
available_features = get_available_features(df, label_col)

if label_col is None:
    st.warning("Wybierz kolumnę etykiety przed dalszymi krokami.")
    st.stop()

# Użyj wczytanych cech
loaded_features = st.session_state.get('selected_features', [])
# Filtruj tylko te cechy, które istnieją w obecnym datasecie
valid_loaded_features = [f for f in loaded_features if f in available_features]
default_features = valid_loaded_features if valid_loaded_features else available_features

selected_features = st.multiselect(
    "Wybierz kolumny cech (features)", 
    options=available_features, 
    default=default_features,
    help="Odznacz kolumny, których nie chcesz użyć jako cechy."
)

X, y, selection_messages = select_features_label(df, selected_features, label_col)
for m in selection_messages:
    if 'Brak' in m or 'Nie' in m:
        st.error(m)
    else:
        st.info(m)

if X is None or y is None:
    st.stop()

st.divider()

# --- Preprocessing section ---
st.header("4) Podział na train/test i preprocessing")

# Sprawdź kolumny kategoryczne
if X is not None:
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        st.warning(f"⚠️ Wykryto kolumny kategoryczne: {cat_cols}. Będą automatycznie zakodowane.")

# Użyj wczytanych wartości preprocessingu
col1, col2 = st.columns(2)
with col1:
    test_size = st.slider("Rozmiar zbioru testowego", min_value=0.1, max_value=0.5, 
                         value=st.session_state.get('test_size', 0.2), step=0.05)
    random_state = st.number_input("Random state (split)", min_value=0, 
                                  value=st.session_state.get('random_state', 42))

with col2:
    drop_na = st.checkbox("Usuń wiersze z NaN", value=st.session_state.get('drop_na', False))
    fill_mean = st.checkbox("Wypełnij NaN średnią (z train)", value=st.session_state.get('fill_mean', False))
    standardize = st.checkbox("Standaryzacja (fitted na train)", value=st.session_state.get('standardize', False))
    one_hot = st.checkbox("One-hot encoding (domyślnie: label encoding)", value=st.session_state.get('one_hot', False), 
                         help="Zaznacz dla one-hot, odznacz dla prostego label encoding")

# Przygotuj informacje o preprocessingu
preprocessing_info = {
    "test_size": test_size,
    "random_state": random_state,
    "drop_na": drop_na,
    "fill_mean": fill_mean,
    "standardize": standardize,
    "one_hot": one_hot,
    "selected_features": selected_features,
    "label_column": label_col
}

st.divider()

# --- Training section ---
st.header("5) Trenowanie i ewaluacja")

# Dodaj opcje ewaluacji
eval_method = st.radio(
    "Wybierz metodę ewaluacji:",
    ["Train/Test Split", "Cross-Validation", "Oba (Train/Test + Cross-Validation)"],
    index=2
)

cv_folds = 5
if eval_method in ["Cross-Validation", "Oba (Train/Test + Cross-Validation)"]:
    cv_folds = st.slider("Liczba foldów dla Cross-Validation", min_value=3, max_value=10, value=5)

if st.button("🚀 Uruchom eksperyment", type="primary", width='stretch'):
    with st.spinner("Wykonuję podział danych i preprocessing..."):
        X_train, X_test, y_train, y_test, split_messages = split_and_preprocess(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            drop_na=drop_na, 
            fill_mean=fill_mean, 
            standardize=standardize, 
            one_hot=one_hot
        )
        
        st.write("**Preprocessing zakończony:**")
        for m in split_messages:
            st.info(m)
    
    with st.spinner("Trenuję klasyfikator i zapisuję wyniki..."):
        if eval_method == "Train/Test Split":
            # Tradycyjna ewaluacja
            experiment_result = run_classifier_experiment(
                clf_name=clf_name,
                params=params,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                preprocessing_info=preprocessing_info,
                dataset_info=dataset_info
            )
        else:
            # Cross-validation lub oba
            experiment_result = run_classifier_experiment_with_cv(
                clf_name=clf_name,
                params=params,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                cv_folds=cv_folds,
                preprocessing_info=preprocessing_info,
                dataset_info=dataset_info
            )
        
        experiment_data = experiment_result['experiment_data']
        saved_filename = experiment_result['saved_filename']
    
    # ZAPISZ WYNIKI W SESSION_STATE
    st.session_state['last_experiment_data'] = experiment_data
    st.session_state['last_experiment_filename'] = saved_filename
    st.session_state['last_experiment_method'] = eval_method
    
    # Wyczyść flagę wczytanego eksperymentu po uruchomieniu nowego
    if 'experiment_loaded' in st.session_state:
        del st.session_state['experiment_loaded']
    
    st.success(f"✅ Eksperyment zakończony! Wyniki zapisane do: `{saved_filename}`")

# WYŚWIETL WYNIKI Z SESSION_STATE (jeśli istnieją)
if 'last_experiment_data' in st.session_state:
    st.divider()
    st.header("📊 Ostatnie wyniki eksperymentu")
    
    # Pokaż podstawowe informacje o ostatnim eksperymencie
    last_data = st.session_state['last_experiment_data']
    last_filename = st.session_state.get('last_experiment_filename', 'N/A')
    last_method = st.session_state.get('last_experiment_method', 'N/A')
    
    exp_info = last_data['experiment_info']
    st.info(f"📁 **Plik:** `saved_evaluations/{last_filename}` | 🔧 **Metoda:** {last_method} | 🤖 **Klasyfikator:** {exp_info['classifier_name']} | 📅 **Data:** {exp_info['timestamp']}")
    
    # Przycisk do wyczyszczenia wyników
    if st.button("🗑️ Wyczyść wyniki", help="Usuwa wyświetlane wyniki z ekranu"):
        del st.session_state['last_experiment_data']
        del st.session_state['last_experiment_filename'] 
        del st.session_state['last_experiment_method']
        st.rerun()
    
    # Wyświetl wyniki w zależności od metody ewaluacji
    if last_method == "Train/Test Split":
        # Pokaż tradycyjne wyniki
        results = last_data['results']
        display_traditional_results(results)
    elif last_method == "Cross-Validation":
        # Pokaż tylko wyniki CV
        cv_results = last_data['results']['cross_validation']
        display_cv_results(cv_results)
    else:
        # Pokaż oba
        traditional_results = last_data['results']['traditional']
        cv_results = last_data['results']['cross_validation']
        
        st.subheader("🔄 Wyniki Cross-Validation")
        display_cv_results(cv_results)
        
        st.subheader("📊 Wyniki Train/Test Split")
        display_traditional_results(traditional_results)