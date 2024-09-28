import unittest
import pandas as pd 
from viz_classes import ScatterPlotCreator
import datetime

class TestScatterPlot(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_parquet("test_data.parquet")

    def test_different_labels(self):
        test_cases = [
            ("On","On","Fan: On, Prediction: On"),
            ("Off","Off",'Fan: Off, Prediction: Off'),
            ("On","Off",'Fan: On, Prediction: Off'),
            ("Off","On",'Fan: Off, Prediction: On')
        ]

        for fan_status, prediction, vis_label in test_cases:
            # generate input date
            test_date = self.df[((self.df['Fan_status'] == fan_status) & (self.df['Predicted'] == prediction))].Datetime.iloc[0].date()
            # generate scatter plot
            vis = ScatterPlotCreator(self.df,date=test_date)
            # retrieve number of rows from dataframe
            calculated_data = len(self.df[(self.df['Fan_status'] == fan_status) & (self.df['Predicted'] == prediction) & (self.df['Datetime'].dt.date == test_date)])
            var_test = vis_label
            # empty list to store x-axis data points
            vis_retrieved_data = []
            scatter_trace = vis.fig.data
            for trace in scatter_trace:
                if trace.name == var_test:
                    for i in trace.x:
                        vis_retrieved_data.append(i)

            self.assertEqual(calculated_data,len(vis_retrieved_data))
    
    def test_none_case(self):

        test_cases = [
            'Fan ON, Prediction On',
            'Fan OFF, Prediction OFF',
            'Fan ON, Prediction OFF',
            'Fan OFF, Prediction ON'
        ]
        for test in test_cases:
            # generate input date
            test_date = pd.Timestamp("2023-01-03").date()
            # generate scatter plot
            vis = ScatterPlotCreator(self.df,date=test_date)
            var_test = test
            # empty list to store x-axis data points
            vis_retrieved_data = []
            scatter_trace = vis.fig.data
            for trace in scatter_trace:
                if trace.name == var_test:
                    for i in trace.x:
                        vis_retrieved_data.append(i)

            self.assertIsNone(None,vis_retrieved_data[0])

if __name__ == "__main__":
    with open("scatterplottest_output.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner)