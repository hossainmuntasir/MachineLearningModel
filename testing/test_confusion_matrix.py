import unittest
import pandas as pd 
from viz_classes import ConfusionMatrixCreator
from sklearn.metrics import confusion_matrix

class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_parquet("test_data.parquet")

        self.fig = ConfusionMatrixCreator(self.df)

    def test_metric_calculation(self):
        cm = confusion_matrix(self.df.Fan_status,self.df.Predicted,labels=['On','Off'])
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]

        TPR = TP / (TP + FN)
        FNR = FN / (FN + TP)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)

        # Create confusion matrix with TPR, FNR, FPR, TNR
        calculated_cm = [[TPR, FNR], [FPR, TNR]]
        vis_metrics = list(self.fig.fig.data[0].z)

        self.assertListEqual(calculated_cm,vis_metrics)

if __name__ == "__main__":
    with open("confusionmatrixtest_output.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner)

