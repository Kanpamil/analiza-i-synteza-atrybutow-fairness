import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator

class ClassifierRunner:
    """
    Klasa do trenowania klasyfikatorów i zapisywania wyników eksperymentów.
    """
    
    def __init__(self):
        self.supported_classifiers = {
            'RandomForest': RandomForestClassifier,
            'SVM': SVC,
            'GradientBoosting': GradientBoostingClassifier,
            'KNN': KNeighborsClassifier
        }
    
    def create_classifier(self, clf_name: str, params: Dict[str, Any]) -> Optional[BaseEstimator]:
        """
        Tworzy instancję klasyfikatora na podstawie nazwy i parametrów.
        """
        if clf_name not in self.supported_classifiers:
            raise ValueError(f"Nieobsługiwany klasyfikator: {clf_name}. Dostępne: {list(self.supported_classifiers.keys())}")
        
        classifier_class = self.supported_classifiers[clf_name]
        return classifier_class(**params)
    
    def train_and_evaluate(self, clf_name: str, params: Dict[str, Any], 
                          X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Trenuje klasyfikator i zwraca wyniki ewaluacji.
        """
        # Utworz klasyfikator
        clf = self.create_classifier(clf_name, params)
        
        # Trenowanie
        start_time = datetime.now()
        clf.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predykcja
        start_time = datetime.now()
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Metryki
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Confusion matrix
        train_cm = confusion_matrix(y_train, y_pred_train)
        test_cm = confusion_matrix(y_test, y_pred_test)
        
        # Classification report
        train_report = classification_report(y_train, y_pred_train, output_dict=True)
        test_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Pobierz unikalne klasy
        unique_classes = sorted(list(set(y_train) | set(y_test)))
        
        results = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'train_confusion_matrix': train_cm.tolist(),
            'test_confusion_matrix': test_cm.tolist(),
            'train_classification_report': train_report,
            'test_classification_report': test_report,
            'training_time_seconds': training_time,
            'prediction_time_seconds': prediction_time,
            'classes': unique_classes,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': X_train.shape[1]
        }
        
        return results
    
    def save_experiment(self, experiment_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Zapisuje dane eksperymentu do pliku JSON w folderze saved_evaluations.
        Zwraca nazwę utworzonego pliku.
        """
        # Utwórz folder saved_evaluations jeśli nie istnieje
        output_folder = "saved_evaluations"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if filename is None:
            # Nowy format nazwy: {klasyfikator}_{plik_wejściowy}_{data}.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            classifier_name = experiment_data.get('experiment_info', {}).get('classifier_name', 'unknown')
            
            # Pobierz nazwę pliku wejściowego bez rozszerzenia
            dataset_info = experiment_data.get('experiment_info', {}).get('dataset_info', {})
            input_filename = dataset_info.get('filename', 'unknown')
            if input_filename.lower().endswith('.csv'):
                input_filename = input_filename[:-4]  # Usuń .csv
            
            # Zastąp problematyczne znaki w nazwie pliku
            classifier_name = classifier_name.replace(' ', '_')
            input_filename = input_filename.replace(' ', '_').replace('-', '_')
            
            filename = f"{classifier_name}_{input_filename}_{timestamp}.json"
        
        # Pełna ścieżka do pliku
        filepath = os.path.join(output_folder, filename)
        
        # Konwersja pandas Series/DataFrame na listy dla JSON
        def convert_to_serializable(obj):
            if isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Rekurencyjnie konwertuj wszystkie obiekty
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_to_serializable(data)
        
        serializable_data = deep_convert(experiment_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        return filename  # Zwraca tylko nazwę pliku, bez ścieżki
    
    def run_full_experiment(self, clf_name: str, params: Dict[str, Any],
                           X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           preprocessing_info: Dict[str, Any] = None,
                           dataset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Uruchamia pełny eksperyment: trenowanie, ewaluację i zapis wyników.
        """
        # Informacje o eksperymencie
        experiment_timestamp = datetime.now().isoformat()
        
        # Trenowanie i ewaluacja
        results = self.train_and_evaluate(clf_name, params, X_train, X_test, y_train, y_test)
        
        # Przygotuj pełne dane eksperymentu
        experiment_data = {
            'experiment_info': {
                'timestamp': experiment_timestamp,
                'classifier_name': clf_name,
                'classifier_params': params,
                'dataset_info': dataset_info or {},
                'preprocessing_info': preprocessing_info or {}
            },
            'data_shapes': {
                'X_train_shape': list(X_train.shape),
                'X_test_shape': list(X_test.shape),
                'y_train_shape': list(y_train.shape),
                'y_test_shape': list(y_test.shape)
            },
            'feature_names': list(X_train.columns),
            'results': results
        }
        
        # Zapisz do pliku
        filename = self.save_experiment(experiment_data)
        
        return {
            'experiment_data': experiment_data,
            'saved_filename': filename
        }
    
    def load_experiment(self, filename: str) -> Dict[str, Any]:
        """
        Wczytuje zapisany eksperyment z pliku JSON z folderu saved_evaluations.
        """
        output_folder = "saved_evaluations"
        filepath = os.path.join(output_folder, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def reproduce_experiment(self, experiment_file: str, 
                           X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Odtwarza eksperyment na podstawie zapisanego pliku konfiguracji.
        """
        experiment_config = self.load_experiment(experiment_file)
        
        clf_name = experiment_config['experiment_info']['classifier_name']
        params = experiment_config['experiment_info']['classifier_params']
        
        return self.run_full_experiment(
            clf_name=clf_name,
            params=params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessing_info=experiment_config['experiment_info'].get('preprocessing_info'),
            dataset_info=experiment_config['experiment_info'].get('dataset_info')
        )
    def cross_validate_classifier(self, clf_name: str, params: Dict[str, Any], 
                                X: pd.DataFrame, y: pd.Series, 
                                cv_folds: int = 5, scoring: list = None) -> Dict[str, Any]:
        """
        Wykonuje cross-validation dla klasyfikatora.
        
        Args:
            clf_name: nazwa klasyfikatora
            params: parametry klasyfikatora
            X: cechy
            y: etykiety
            cv_folds: liczba foldów dla cross-validation
            scoring: lista metryk do ewaluacji
            
        Returns:
            Dict z wynikami cross-validation
        """
        if scoring is None:
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Utwórz klasyfikator
        clf = self.create_classifier(clf_name, params)
        
        # Utwórz stratified k-fold (zachowuje proporcje klas)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Wykonaj cross-validation
        start_time = datetime.now()
        cv_results = cross_validate(
            clf, X, y, 
            cv=cv, 
            scoring=scoring, 
            return_train_score=True,
            n_jobs=-1  # Wykorzystaj wszystkie dostępne rdzenie
        )
        cv_time = (datetime.now() - start_time).total_seconds()
        
        # Przetwórz wyniki
        results = {
            'cv_folds': cv_folds,
            'cv_time_seconds': cv_time,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'classes': sorted(list(set(y))),
            'scoring_metrics': scoring
        }
        
        # Dla każdej metryki dodaj statystyki
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results[f'{metric}_test'] = {
                'scores': test_scores.tolist(),
                'mean': float(np.mean(test_scores)),
                'std': float(np.std(test_scores)),
                'min': float(np.min(test_scores)),
                'max': float(np.max(test_scores))
            }
            
            results[f'{metric}_train'] = {
                'scores': train_scores.tolist(),
                'mean': float(np.mean(train_scores)),
                'std': float(np.std(train_scores)),
                'min': float(np.min(train_scores)),
                'max': float(np.max(train_scores))
            }
        
        # Dodaj fit_time i score_time
        results['fit_time'] = {
            'mean': float(np.mean(cv_results['fit_time'])),
            'std': float(np.std(cv_results['fit_time'])),
            'total': float(np.sum(cv_results['fit_time']))
        }
        
        results['score_time'] = {
            'mean': float(np.mean(cv_results['score_time'])),
            'std': float(np.std(cv_results['score_time'])),
            'total': float(np.sum(cv_results['score_time']))
        }
        
        return results
    
    def run_full_experiment_with_cv(self, clf_name: str, params: Dict[str, Any],
                                   X_train: pd.DataFrame, X_test: pd.DataFrame,
                                   y_train: pd.Series, y_test: pd.Series,
                                   cv_folds: int = 5,
                                   preprocessing_info: Dict[str, Any] = None,
                                   dataset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Uruchamia pełny eksperyment z train/test split + cross-validation.
        """
        experiment_timestamp = datetime.now().isoformat()
        
        # 1. Tradycyjna ewaluacja (train/test split)
        traditional_results = self.train_and_evaluate(
            clf_name, params, X_train, X_test, y_train, y_test
        )
        
        # 2. Cross-validation na pełnym zbiorze (X_train + X_test)
        X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
        
        cv_results = self.cross_validate_classifier(
            clf_name, params, X_full, y_full, cv_folds
        )
        
        # 3. Przygotuj pełne dane eksperymentu
        experiment_data = {
            'experiment_info': {
                'timestamp': experiment_timestamp,
                'classifier_name': clf_name,
                'classifier_params': params,
                'dataset_info': dataset_info or {},
                'preprocessing_info': preprocessing_info or {},
                'cv_folds': cv_folds
            },
            'data_shapes': {
                'X_train_shape': list(X_train.shape),
                'X_test_shape': list(X_test.shape),
                'X_full_shape': list(X_full.shape),
                'y_train_shape': list(y_train.shape),
                'y_test_shape': list(y_test.shape),
                'y_full_shape': list(y_full.shape)
            },
            'feature_names': list(X_train.columns),
            'results': {
                'traditional': traditional_results,
                'cross_validation': cv_results
            }
        }
        
        # Zapisz do pliku
        filename = self.save_experiment(experiment_data)
        
        return {
            'experiment_data': experiment_data,
            'saved_filename': filename
        }

# Funkcje pomocnicze dla streamlit
def run_classifier_experiment(clf_name: str, params: Dict[str, Any],
                             X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series,
                             preprocessing_info: Dict[str, Any] = None,
                             dataset_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Funkcja wrapper dla łatwego użycia w streamlit.
    """
    runner = ClassifierRunner()
    return runner.run_full_experiment(
        clf_name, params, X_train, X_test, y_train, y_test,
        preprocessing_info, dataset_info
    )
    
def run_classifier_experiment_with_cv(clf_name: str, params: Dict[str, Any],
                                     X_train: pd.DataFrame, X_test: pd.DataFrame,
                                     y_train: pd.Series, y_test: pd.Series,
                                     cv_folds: int = 5,
                                     preprocessing_info: Dict[str, Any] = None,
                                     dataset_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Funkcja wrapper dla łatwego użycia w streamlit z cross-validation.
    """
    runner = ClassifierRunner()
    return runner.run_full_experiment_with_cv(
        clf_name, params, X_train, X_test, y_train, y_test,
        cv_folds, preprocessing_info, dataset_info
    )

def load_and_display_experiment(filename: str) -> Dict[str, Any]:
    """
    Wczytuje i formatuje wyniki eksperymentu do wyświetlenia.
    """
    runner = ClassifierRunner()
    return runner.load_experiment(filename)