import numpy as np
from dataclasses import dataclass

def safe_div(n, d):
    return np.divide(n, d, out=np.full_like(d, np.nan, dtype=np.float64), where=d!=0)

def metric(func):
    func._is_metric = True
    return property(func)

class BaseMetrics:
    @classmethod
    def metric_names(cls):
        return sorted(
            name for name in dir(cls)
            if isinstance(attr := getattr(cls, name), property) and getattr(attr.fget, '_is_metric', False)
        )

    def calculate(self, metric_name):
        return getattr(self, metric_name)

@dataclass
class ClassificationMetrics(BaseMetrics):
    tp: np.ndarray
    fp: np.ndarray
    tn: np.ndarray
    fn: np.ndarray
    p = property(lambda self: self.tp + self.fn)
    n = property(lambda self: self.fp + self.tn)
    p_pred = property(lambda self: self.tp + self.fp)
    n_pred = property(lambda self: self.tn + self.fn)
    total = property(lambda self: self.p + self.n)

    @staticmethod
    def component_names():
        return np.array(['TP', 'FP', 'TN', 'FN'])

    def __add__(self, other):
        return ClassificationMetrics(self.tp + other.tp, self.fp + other.fp, self.tn + other.tn, self.fn + other.fn)

    accuracy = metric(lambda self: safe_div(self.tp + self.tn, self.total))
    balanced_accuracy = metric(lambda self: (self.tpr + self.tnr) / 2)
    f1_score = metric(lambda self: safe_div(2 * self.precision * self.recall, self.precision + self.recall))
    fnr = metric(lambda self: safe_div(self.fn, self.p))
    fpr = metric(lambda self: safe_div(self.fp, self.n))
    g_mean = metric(lambda self: np.sqrt(self.recall * self.specificity))
    mcc = metric(lambda self: safe_div((self.tp * self.tn) - (self.fp * self.fn), np.sqrt(self.p * self.n * self.p_pred * self.n_pred)))
    npv = metric(lambda self: safe_div(self.tn, self.n_pred))
    ppv = metric(lambda self: safe_div(self.tp, self.p_pred))
    precision = metric(lambda self: self.ppv)
    recall = metric(lambda self: self.tpr)
    specificity = metric(lambda self: self.tnr)
    tnr = metric(lambda self: safe_div(self.tn, self.n))
    tpr = metric(lambda self: safe_div(self.tp, self.p))

@dataclass
class FairnessMetrics(BaseMetrics):
    tp_prot: np.ndarray
    fp_prot: np.ndarray
    tn_prot: np.ndarray
    fn_prot: np.ndarray
    tp_unprot: np.ndarray
    fp_unprot: np.ndarray
    tn_unprot: np.ndarray
    fn_unprot: np.ndarray
    prot = property(lambda self: ClassificationMetrics(self.tp_prot, self.fp_prot, self.tn_prot, self.fn_prot))
    unprot = property(lambda self: ClassificationMetrics(self.tp_unprot, self.fp_unprot, self.tn_unprot, self.fn_unprot))
    total = property(lambda self: self.prot + self.unprot)

    @staticmethod
    def component_names():
        return np.array(['TPp', 'FPp', 'TNp', 'FNp', 'TPup', 'FPup', 'TNup', 'FNup'])

    accuracy_equality_difference = metric(lambda self: self.prot.accuracy - self.unprot.accuracy)
    accuracy_equality_ratio = metric(lambda self: safe_div(self.prot.accuracy, self.unprot.accuracy))
    equal_opportunity_difference = metric(lambda self: self.prot.tpr - self.unprot.tpr)
    equal_opportunity_ratio = metric(lambda self: safe_div(self.prot.tpr, self.unprot.tpr))
    predictive_equality_difference = metric(lambda self: self.prot.fpr - self.unprot.fpr)
    predictive_equality_ratio = metric(lambda self: safe_div(self.prot.fpr, self.unprot.fpr))
    predictive_parity_negative_difference = metric(lambda self: self.prot.npv - self.unprot.npv)
    predictive_parity_negative_ratio = metric(lambda self: safe_div(self.prot.npv, self.unprot.npv))
    predictive_parity_positive_difference = metric(lambda self: self.prot.ppv - self.unprot.ppv)
    predictive_parity_positive_ratio = metric(lambda self: safe_div(self.prot.ppv, self.unprot.ppv))
    statistical_parity_difference = metric(lambda self: safe_div(self.prot.p_pred, self.prot.total) - safe_div(self.unprot.p_pred, self.unprot.total))
    statistical_parity_ratio = metric(lambda self: safe_div(safe_div(self.prot.p_pred, self.prot.total), safe_div(self.unprot.p_pred, self.unprot.total)))
