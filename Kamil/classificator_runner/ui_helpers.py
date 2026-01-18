import streamlit as st
import pandas as pd

# Funkcje pomocnicze do wyświetlania wyników
def display_traditional_results(results):
    """Wyświetla tradycyjne wyniki train/test"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train Accuracy", f"{results['train_accuracy']:.4f}")
    with col2:
        st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
    with col3:
        st.metric("Czas trenowania", f"{results['training_time_seconds']:.2f}s")
    
    # Confusion matrices
    st.subheader("Confusion Matrix")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Train:**")
        train_cm = pd.DataFrame(
            results['train_confusion_matrix'], 
            index=results['classes'], 
            columns=results['classes']
        )
        st.dataframe(train_cm)
    
    with col2:
        st.write("**Test:**")
        test_cm = pd.DataFrame(
            results['test_confusion_matrix'], 
            index=results['classes'], 
            columns=results['classes']
        )
        st.dataframe(test_cm)

def display_cv_results(cv_results):
    """Wyświetla wyniki cross-validation"""
    # Podstawowe metryki
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        acc_test = cv_results['accuracy_test']
        st.metric(
            "Accuracy (CV)", 
            f"{acc_test['mean']:.4f} ±{acc_test['std']:.4f}",
            help=f"Min: {acc_test['min']:.4f}, Max: {acc_test['max']:.4f}"
        )
    
    with col2:
        prec_test = cv_results['precision_macro_test']
        st.metric(
            "Precision (CV)", 
            f"{prec_test['mean']:.4f} ±{prec_test['std']:.4f}",
            help=f"Min: {prec_test['min']:.4f}, Max: {prec_test['max']:.4f}"
        )
    
    with col3:
        rec_test = cv_results['recall_macro_test']
        st.metric(
            "Recall (CV)", 
            f"{rec_test['mean']:.4f} ±{rec_test['std']:.4f}",
            help=f"Min: {rec_test['min']:.4f}, Max: {rec_test['max']:.4f}"
        )
    
    with col4:
        f1_test = cv_results['f1_macro_test']
        st.metric(
            "F1-Score (CV)", 
            f"{f1_test['mean']:.4f} ±{f1_test['std']:.4f}",
            help=f"Min: {f1_test['min']:.4f}, Max: {f1_test['max']:.4f}"
        )
    
    # Szczegółowe wyniki CV
    with st.expander("📈 Szczegółowe wyniki Cross-Validation"):
        metrics_df = pd.DataFrame({
            'Fold': list(range(1, cv_results['cv_folds'] + 1)),
            'Accuracy': cv_results['accuracy_test']['scores'],
            'Precision': cv_results['precision_macro_test']['scores'],
            'Recall': cv_results['recall_macro_test']['scores'],
            'F1-Score': cv_results['f1_macro_test']['scores']
        })
        st.dataframe(metrics_df, width='stretch')
        
        # Informacje o czasie
        st.write(f"**Czas CV:** {cv_results['cv_time_seconds']:.2f}s")
        st.write(f"**Średni czas fit:** {cv_results['fit_time']['mean']:.3f}s ±{cv_results['fit_time']['std']:.3f}s")
        st.write(f"**Folds:** {cv_results['cv_folds']}, **Próbki:** {cv_results['n_samples']}, **Cechy:** {cv_results['n_features']}")