import numpy as np
from dataclasses import dataclass

def safe_div(n, d):
    return np.divide(n, d, out=np.full_like(d, np.nan, dtype=float), where=d!=0)

def metric(func):
    func._is_metric = True
    return property(func)

class Report:
    @classmethod
    def available_metrics(cls):
        metric_names = []
        for base in cls.mro():
            for name, value in base.__dict__.items():
                if isinstance(value, property) and hasattr(value.fget, '_is_metric'):
                    metric_names.append(name)
        return sorted(metric_names)

    def calculate(self, metric_name):
        return getattr(self, metric_name)

# https://en.wikipedia.org/wiki/Template:Diagnostic_testing_diagram
@dataclass
class ClassReport(Report):
    tp: float
    fp: float
    tn: float
    fn: float
    p = property(lambda self: self.tp + self.fn)
    n = property(lambda self: self.fp + self.tn)
    p_pred = property(lambda self: self.tp + self.fp)
    n_pred = property(lambda self: self.tn + self.fn)
    total = property(lambda self: self.p + self.n)

    def __add__(self, other):
        return ClassReport(self.tp + other.tp, self.fp + other.fp, self.tn + other.tn, self.fn + other.fn)

    accuracy = metric(lambda self: safe_div(self.tp + self.tn, self.total))
    balanced_accuracy = metric(lambda self: (self.tpr + self.tnr) / 2)
    f1_score = metric(lambda self: safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn))
    fnr = metric(lambda self: safe_div(self.fn, self.p))
    fpr = metric(lambda self: safe_div(self.fp, self.n))
    g_mean = metric(lambda self: np.sqrt(self.recall * self.specificity))
    mcc = metric(lambda self: safe_div((self.tp * self.tn) - (self.fp * self.fn), np.sqrt(self.p * self.n * self.p_pred * self.n_pred)))
    precision = metric(lambda self: safe_div(self.tp, self.p_pred))
    recall = metric(lambda self: self.tpr)
    specificity = metric(lambda self: self.tnr)
    tnr = metric(lambda self: safe_div(self.tn, self.n))
    tpr = metric(lambda self: safe_div(self.tp, self.p))

@dataclass
class FairReport(Report):
    prot: ClassReport
    unprot: ClassReport
    total = property(lambda self: self.prot + self.unprot)

    equal_opportunity_difference = metric(lambda self: self.prot.tpr - self.unprot.tpr)
    equal_opportunity_ratio = metric(lambda self: safe_div(self.prot.tpr, self.unprot.tpr))
    equalized_odds_difference_avg = metric(lambda self: (abs(self.prot.tpr - self.unprot.tpr) + abs(self.prot.fpr - self.unprot.fpr)) / 2)
    equalized_odds_difference_max = metric(lambda self: max(abs(self.prot.tpr - self.unprot.tpr), abs(self.prot.fpr - self.unprot.fpr)))
    equalized_odds_ratio_avg = metric(lambda self: (safe_div(self.prot.tpr, self.unprot.tpr) + safe_div(self.prot.fpr, self.unprot.fpr)) / 2)
    equalized_odds_ratio_max = metric(lambda self: max(safe_div(self.prot.tpr, self.unprot.tpr), safe_div(self.prot.fpr, self.unprot.fpr)))
    predictive_equality_difference = metric(lambda self: self.prot.fpr - self.unprot.fpr)
    predictive_equality_ratio = metric(lambda self: safe_div(self.prot.fpr, self.unprot.fpr))
    statistical_parity_difference = metric(lambda self: safe_div(self.prot.p_pred, self.prot.total) - safe_div(self.unprot.p_pred, self.unprot.total))
    statistical_parity_ratio = metric(lambda self: safe_div(safe_div(self.prot.p_pred, self.prot.total), safe_div(self.unprot.p_pred, self.unprot.total)))
