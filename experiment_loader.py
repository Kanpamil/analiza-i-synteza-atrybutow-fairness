import streamlit as st
import json
import os
from typing import Dict, Any, Optional

def load_experiment_config(filename: str) -> Dict[str, Any]:
    """
    Wczytuje konfigurację eksperymentu z pliku JSON z folderu saved_evaluations.
    """
    output_folder = "saved_evaluations"
    filepath = os.path.join(output_folder, filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def show_experiment_loader():
    """
    Wyświetla interfejs do wczytywania eksperymentów na początku strony.
    """
    st.header("🔄 Wczytaj poprzedni eksperyment")
    
    # Sprawdź folder saved_evaluations
    output_folder = "saved_evaluations"
    if not os.path.exists(output_folder):
        st.info(f"Brak folderu `{output_folder}` z zapisanymi eksperymentami.")
        return False
    
    json_files = [f for f in os.listdir(output_folder) if f.endswith('.json')]
    
    if not json_files:
        st.info(f"Brak zapisanych eksperymentów w folderze `{output_folder}/`.")
        return False
    
    # Posortuj pliki według daty modyfikacji (najnowsze pierwsze)
    json_files_with_time = []
    for f in json_files:
        filepath = os.path.join(output_folder, f)
        mtime = os.path.getmtime(filepath)
        json_files_with_time.append((f, mtime))
    
    json_files_with_time.sort(key=lambda x: x[1], reverse=True)
    json_files = [f[0] for f in json_files_with_time]
    
    st.success(f"📁 Znaleziono {len(json_files)} eksperymentów w folderze `{output_folder}/`")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_experiment = st.selectbox(
            "Wybierz eksperyment do wczytania:", 
            options=["- brak wyboru -"] + json_files,
            key="experiment_selector",
            help="Pliki posortowane według daty (najnowsze pierwsze)"
        )
    
    with col2:
        load_button = st.button("🔄 Wczytaj", type="primary")
    
    if load_button and selected_experiment != "- brak wyboru -":
        try:
            experiment_data = load_experiment_config(selected_experiment)
            apply_experiment_config_to_session(experiment_data)
            st.session_state['loaded_experiment_file'] = selected_experiment
            st.success(f"✅ Wczytano eksperyment: {selected_experiment}")
            st.rerun()  # Restart aplikacji z nowymi ustawieniami
            return True
        except Exception as e:
            st.error(f"Błąd podczas wczytywania eksperymentu: {str(e)}")
            return False
    
    # Pokaż podgląd wybranego eksperymentu
    if selected_experiment != "- brak wyboru -":
        try:
            experiment_data = load_experiment_config(selected_experiment)
            exp_info = experiment_data['experiment_info']
            
            with st.expander("👁️ Podgląd eksperymentu"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Data:** {exp_info['timestamp']}")
                    st.write(f"**Klasyfikator:** {exp_info['classifier_name']}")
                    st.write(f"**Dataset:** {exp_info.get('dataset_info', {}).get('filename', 'N/A')}")
                
                with col2:
                    results = experiment_data.get('results', {})
                    if 'traditional' in results:
                        # Eksperyment z CV
                        st.write(f"**Train Accuracy:** {results['traditional'].get('train_accuracy', 'N/A'):.4f}")
                        st.write(f"**Test Accuracy:** {results['traditional'].get('test_accuracy', 'N/A'):.4f}")
                        cv_acc = results.get('cross_validation', {}).get('accuracy_test', {})
                        if cv_acc:
                            st.write(f"**CV Accuracy:** {cv_acc.get('mean', 'N/A'):.4f} ±{cv_acc.get('std', 'N/A'):.4f}")
                    elif results:
                        # Tradycyjny eksperyment
                        st.write(f"**Train Accuracy:** {results.get('train_accuracy', 'N/A'):.4f}")
                        st.write(f"**Test Accuracy:** {results.get('test_accuracy', 'N/A'):.4f}")
                        st.write(f"**Cechy:** {results.get('n_features', 'N/A')}")
                
                st.write("**Parametry klasyfikatora:**")
                st.json(exp_info['classifier_params'])
        except Exception as e:
            st.error(f"Błąd podczas odczytu podglądu: {str(e)}")
    
    return False

def apply_experiment_config_to_session(experiment_data: Dict[str, Any]) -> None:
    """
    Aplikuje konfigurację eksperymentu do streamlit session_state.
    """
    exp_info = experiment_data['experiment_info']
    
    # Ustawienia klasyfikatora
    st.session_state['clf_name'] = exp_info['classifier_name']
    st.session_state['clf_params'] = exp_info['classifier_params']
    
    # Ustawienia preprocessingu
    preprocessing = exp_info.get('preprocessing_info', {})
    st.session_state['test_size'] = preprocessing.get('test_size', 0.2)
    st.session_state['random_state'] = preprocessing.get('random_state', 42)
    st.session_state['drop_na'] = preprocessing.get('drop_na', False)
    st.session_state['fill_mean'] = preprocessing.get('fill_mean', False)
    st.session_state['standardize'] = preprocessing.get('standardize', False)
    st.session_state['one_hot'] = preprocessing.get('one_hot', False)
    
    # Ustawienia cech i etykiet
    st.session_state['selected_features'] = preprocessing.get('selected_features', [])
    st.session_state['label_col'] = preprocessing.get('label_column', None)
    
    # Informacje o datasecie
    st.session_state['dataset_info'] = exp_info.get('dataset_info', {})
    st.session_state['experiment_loaded'] = True
    st.session_state['loaded_experiment_file'] = None

def get_classifier_params_from_session(clf_name: str) -> Dict[str, Any]:
    """
    Pobiera parametry klasyfikatora z session_state lub zwraca domyślne.
    """
    if 'clf_params' in st.session_state and st.session_state.get('clf_name') == clf_name:
        return st.session_state['clf_params']
    
    # Domyślne parametry
    defaults = {
        'RandomForest': {'n_estimators': 100, 'max_depth': None, 'random_state': 42},
        'SVM': {'C': 1.0, 'kernel': 'rbf', 'random_state': 42},
        'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
        'KNN': {'n_neighbors': 5, 'weights': 'uniform'}
    }
    return defaults.get(clf_name, {})