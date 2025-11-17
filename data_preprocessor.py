# Pliki: 
# 1) data_preprocessor.py  (moduł z funkcjami do wczytywania i preprocessingu oraz wyborem kolumn features/label)
# 2) classifier_selector.py (interfejs Streamlit, importuje moduł preprocessingu)

# ---------------------- data_preprocessor.py ----------------------
"""
Moduł: data_preprocessor.py
Funkcje:
- load_csv_from_path(path) -> pd.DataFrame | None
- load_csv_from_bytes(io_bytes) -> pd.DataFrame | None
- apply_preprocessing(df, drop_na=False, fill_mean=False, standardize=False, one_hot=False, selected_cols=None) -> (pd.DataFrame, list_of_messages)
- select_features_label(df, feature_cols, label_col) -> (X_df, y_series, messages)
- list_columns(df) -> list[str]

Użycie: umieść ten plik w tym samym katalogu co classifier_selector.py
"""

# data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional


def load_csv_from_path(path: str) -> Optional[pd.DataFrame]:
    """Wczytuje plik CSV z podanej ścieżki. Zwraca DataFrame lub None i nie podnosi wyjątku.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_csv_from_bytes(io_bytes) -> Optional[pd.DataFrame]:
    """Wczytuje CSV z obiektu bytes-like przesłanego przez Streamlit file_uploader.
    """
    try:
        return pd.read_csv(io_bytes)
    except Exception:
        return None


def list_columns(df: pd.DataFrame) -> List[str]:
    """Zwraca listę nazw kolumn z dataframe (pomocnicza)."""
    return list(df.columns)


def select_features_label(df: pd.DataFrame, feature_cols: Optional[List[str]], label_col: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
    """
    Waliduje i zwraca X (DataFrame z cechami) i y (Series z etykietą).
    Zwraca (X, y, messages). Jeśli coś jest nie tak, X lub y może być None i messages zawiera informacje o błędach.

    - feature_cols: lista kolumn, które mają być cechami. Automatycznie wyklucza label_col jeśli jest wybrana.
    - label_col: nazwa kolumny etykiety. Jeśli None lub nie istnieje -> y będzie None i komunikat.
    """
    messages: List[str] = []
    X = None
    y = None

    cols = list(df.columns)
    
    # Sprawdź czy label_col istnieje
    if label_col is None:
        messages.append("Nie wybrano kolumny etykiety (label_col).")
        return None, None, messages

    if label_col not in cols:
        messages.append(f"Kolumna etykiety '{label_col}' nie istnieje w dataframe.")
        return None, None, messages

    # Automatycznie usuń label_col z feature_cols jeśli tam jest
    if feature_cols and label_col in feature_cols:
        feature_cols = [col for col in feature_cols if col != label_col]
        messages.append(f"Automatycznie usunięto kolumnę etykiety '{label_col}' z listy cech.")

    # Jeśli brak feature_cols po usunięciu label_col
    if not feature_cols:
        messages.append("Brak wybranych kolumn cech (feature_cols) po wykluczeniu kolumny etykiety.")
        return None, None, messages

    # Sprawdź czy wszystkie feature_cols istnieją
    missing_features = [c for c in feature_cols if c not in cols]
    if missing_features:
        messages.append(f"Nie znaleziono kolumn cech: {missing_features}")
        return None, None, messages

    # Wytnij X i y
    X = df[feature_cols].copy()
    y = df[label_col].copy()
    messages.append(f"Wybrane cechy: {len(feature_cols)} kolumn")
    messages.append(f"Wybrana kolumna etykiety: {label_col}")

    return X, y, messages


def get_available_features(df: pd.DataFrame, label_col: Optional[str] = None) -> List[str]:
    """
    Zwraca listę kolumn, które mogą być użyte jako cechy (wszystkie kolumny z wyjątkiem label_col).
    
    - df: DataFrame
    - label_col: kolumna etykiety do wykluczenia z listy cech
    
    Returns: lista nazw kolumn dostępnych jako cechy
    """
    all_cols = list(df.columns)
    if label_col and label_col in all_cols:
        return [col for col in all_cols if col != label_col]
    return all_cols


def split_and_preprocess(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42, 
                        drop_na: bool = False, fill_mean: bool = False, standardize: bool = False, 
                        one_hot: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Dzieli dane na treningowe i testowe, a następnie stosuje preprocessing osobno dla każdego zbioru.
    WAŻNE: Transformacje (standardization, mean imputation) są fitowane tylko na danych treningowych
    i następnie aplikowane na danych testowych, aby uniknąć data leakage.
    
    Returns:
        X_train, X_test, y_train, y_test, messages
    """
    messages = []
    
    # 1. Najpierw dropna jeśli wybrane (na pełnym zbiorze)
    if drop_na:
        before_rows = len(X)
        # Usuń wiersze z NaN z obu X i y
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask].copy()
        y_clean = y[mask].copy()
        messages.append(f"dropna: usunięto {before_rows - len(X_clean)} wierszy z NaN")
    else:
        X_clean = X.copy()
        y_clean = y.copy()
    
    # 2. Automatyczne wykrycie kolumn kategorycznych (przed podziałem)
    cat_cols = X_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        messages.append(f"Wykryto kolumny kategoryczne: {cat_cols}")
    
    # 3. Podział na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state, 
        stratify=y_clean if len(y_clean.unique()) > 1 else None
    )
    messages.append(f"Podział danych: {len(X_train)} treningowych, {len(X_test)} testowych (test_size={test_size})")
    
    # 4. One-hot encoding ALBO automatyczne label encoding dla kolumn kategorycznych
    if cat_cols:
        if one_hot:
            # One-hot encoding (dopasowany na train, aplikowany na test)
            # Get dummies for train
            X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
            # Get dummies for test (może mieć inne kategorie)
            X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
            
            # Upewnij się, że test ma te same kolumny co train
            missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
            for col in missing_cols:
                X_test_encoded[col] = 0
                
            # Usuń dodatkowe kolumny z test, które nie są w train
            extra_cols = set(X_test_encoded.columns) - set(X_train_encoded.columns)
            X_test_encoded = X_test_encoded.drop(columns=extra_cols)
            
            # Uporządkuj kolumny
            X_test_encoded = X_test_encoded[X_train_encoded.columns]
            
            X_train = X_train_encoded
            X_test = X_test_encoded
            messages.append(f"one-hot encoding: zakodowano kolumny {cat_cols}")
        else:
            # Automatyczne label encoding dla kolumn kategorycznych
            from sklearn.preprocessing import LabelEncoder
            
            for col in cat_cols:
                le = LabelEncoder()
                # Fit na połączonych danych, żeby mieć wszystkie kategorie
                combined_values = pd.concat([X_train[col], X_test[col]]).astype(str)
                le.fit(combined_values)
                
                # Transform osobno
                X_train[col] = le.transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                
                unique_cats = list(le.classes_)
                messages.append(f"label encoding: {col} -> {unique_cats}")
    
    # 5. Fill mean (średnia z train, aplikowana na test)
    if fill_mean:
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            train_means = X_train[num_cols].mean()
            X_train[num_cols] = X_train[num_cols].fillna(train_means)
            X_test[num_cols] = X_test[num_cols].fillna(train_means)
            messages.append(f"fillna mean: zastosowano dla kolumn {list(num_cols)} (średnie z train)")
        else:
            messages.append("fillna mean: brak kolumn numerycznych")
    
    # 6. Standardization (fitted na train, transform na test)
    if standardize:
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
            messages.append(f"Standaryzacja: fitted na train, applied na test dla kolumn {list(num_cols)}")
        else:
            messages.append("Standaryzacja: brak kolumn numerycznych")
    
    return X_train, X_test, y_train, y_test, messages
