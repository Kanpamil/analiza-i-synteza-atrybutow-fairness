MAPPING = {
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Balanced accuracy',
    'f1_score': 'F1 score',
    'fnr': 'False negative rate',
    'fpr': 'False Positive rate',
    'g_mean': 'G-mean',
    'mcc': 'Matthews correlation coefficient',
    'precision': 'Precision',
    'recall': 'Recall',
    'specificity': 'Specificity',
    'tnr': 'True negative rate',
    'tpr': 'True positive rate',
    'equal_opportunity_difference': 'Equal opportunity difference',
    'equal_opportunity_ratio': 'Equal opportunity ratio',
    'equalized_odds_difference_mean': 'Equalized odds difference mean',
    'equalized_odds_difference_max': 'Equalized odds difference max',
    'equalized_odds_ratio_mean': 'Equalized odds ratio mean',
    'equalized_odds_ratio_min': 'Equalized odds ratio min',
    'predictive_equality_difference': 'Predictive equality difference',
    'predictive_equality_ratio': 'Predictive equality ratio',
    'statistical_parity_difference': 'Statistical parity difference',
    'statistical_parity_ratio': 'Statistical parity ratio'
}

def to_display_name(metric_name):
    return MAPPING[metric_name]

def to_metric_name(display_name):
    reverse_mapping = {v: k for k, v in MAPPING.items()}
    return reverse_mapping[display_name]
