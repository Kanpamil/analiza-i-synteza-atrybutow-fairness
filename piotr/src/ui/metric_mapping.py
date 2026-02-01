MAPPING = {
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Balanced accuracy',
    'f1_score': 'F1 score',
    'fnr': 'False negative rate',
    'fpr': 'False Positive rate',
    'g_mean': 'G-mean',
    'mcc': 'Matthews correlation coefficient',
    'npv': 'Negative predictive value',
    'ppv': 'Positive predictive value',
    'precision': 'Precision',
    'recall': 'Recall',
    'specificity': 'Specificity',
    'tnr': 'True negative rate',
    'tpr': 'True positive rate',
    'accuracy_equality_difference': 'Accuracy equality difference',
    'accuracy_equality_ratio': 'Accuracy equality ratio',
    'equal_opportunity_difference': 'Equal opportunity difference',
    'equal_opportunity_ratio': 'Equal opportunity ratio',
    'predictive_equality_difference': 'Predictive equality difference',
    'predictive_equality_ratio': 'Predictive equality ratio',
    'predictive_parity_positive_difference': 'Predictive parity positive difference',
    'predictive_parity_positive_ratio': 'Predictive parity positive ratio',
    'predictive_parity_negative_difference': 'Predictive parity negative difference',
    'predictive_parity_negative_ratio': 'Predictive parity negative ratio',
    'statistical_parity_difference': 'Statistical parity difference',
    'statistical_parity_ratio': 'Statistical parity ratio'
}

def to_display_name(metric_name):
    return MAPPING[metric_name]

def to_metric_name(display_name):
    reverse_mapping = {v: k for k, v in MAPPING.items()}
    return reverse_mapping[display_name]
