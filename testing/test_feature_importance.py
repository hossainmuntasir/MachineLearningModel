import unittest 
import pandas as pd
from viz_classes import FeatureImportanceCreator
from joblib import load

class TestFeatureImportance(unittest.TestCase):
    def setUp(self):
        self.model = load("trained_RFC.joblib")
        self.fig = FeatureImportanceCreator(self.model)
    
    def test_vis_length(self):
        no_items = len(self.fig.fig.data[0].x)
        self.assertEqual(no_items,11)
    
    def test_features(self):
        viz_features = list(self.fig.fig.data[0].y)
        test_features = list(self.model.feature_names_in_)
        self.assertCountEqual(viz_features,test_features)
    
    def test_feature_importance_values(self):
        viz_importance_values = list(self.fig.fig.data[0].x)
        test_importance_values = list(self.model.feature_importances_)
        self.assertCountEqual(viz_importance_values,test_importance_values)

if __name__ == "__main__":
    with open("featureimportancetest_output.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner)