import unittest
import numpy as np
from src.core.metrics import ClassReport, FairReport

class TestMetrics(unittest.TestCase):
    def test_class_report_numpy(self):
        class_report = ClassReport(
            tp=np.array([0.1, 0.35]),
            fp=np.array([0.2, 0.25]),
            tn=np.array([0.3, 0.10]),
            fn=np.array([0.4, 0.30])
        )

        np.testing.assert_array_almost_equal(class_report.accuracy, np.array([0.40, 0.45]))
        np.testing.assert_array_almost_equal(class_report.mcc, np.array([-0.218218, -0.171184]))

    def test_fair_report_numpy(self):
        prot_report = ClassReport(
            tp=np.array([0.05, 0.5]),
            fp=np.array([0.10, 0.3]),
            tn=np.array([0.11, 0.2]),
            fn=np.array([0.04, 0.1])
        )

        unprot_report = ClassReport(
            tp=np.array([0.30, 0.1]),
            fp=np.array([0.15, 0.2]),
            tn=np.array([0.05, 0.3]),
            fn=np.array([0.20, 0.3])
        )

        fair_report = FairReport(prot=prot_report, unprot=unprot_report)

        np.testing.assert_array_almost_equal(fair_report.equal_opportunity_difference, np.array([-0.044444, 0.583333]))

if __name__ == '__main__':
    unittest.main()
