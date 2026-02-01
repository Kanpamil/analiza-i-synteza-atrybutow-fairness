SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] {
    min-width: 250px;
    max-width: 250px;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
</style>
"""

METRIC_PRETTY_NAMES = {
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Balanced Accuracy',
    'f1_score': 'F1 Score',
    'precision': 'Precision',
    'recall': 'Recall (TPR)',
    'specificity': 'Specificity (TNR)',
    'mcc': 'MCC',
    'g_mean': 'G-Mean',
    'fnr': 'False Negative Rate',
    'fpr': 'False Positive Rate',
    'tnr': 'True Negative Rate',
    'tpr': 'True Positive Rate',
    'statistical_parity_difference': 'Statistical Parity Difference',
    'statistical_parity_ratio': 'Statistical Parity Ratio',
    'equal_opportunity_difference': 'Equal Opportunity Difference',
    'equal_opportunity_ratio': 'Equal Opportunity Ratio',
    'equalized_odds_difference_avg': 'Equalized Odds Difference (Avg)',
    'equalized_odds_difference_max': 'Equalized Odds Difference (Max)',
    'equalized_odds_ratio_avg': 'Equalized Odds Ratio (Avg)',
    'equalized_odds_ratio_max': 'Equalized Odds Ratio (Max)',
    'predictive_equality_difference': 'Predictive Equality Difference',
    'predictive_equality_ratio': 'Predictive Equality Ratio'
}

def get_pretty_name(metric_key):
    return METRIC_PRETTY_NAMES.get(metric_key, metric_key)