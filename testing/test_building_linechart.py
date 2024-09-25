import pandas as pd
from viz_classes import LineChartCreator
import unittest
import numpy as np

class TestLineChart(unittest.TestCase):
    def setUp(self):
        data = {
            "building_no": [1, 1, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1],
            'Zone_name': ['ActEast', 'ActEast', 'ActEast', 'Ac-2-1', 'Ahu-G-01', 'West3', 'West3', 'Ac-2-1', 'Ahu-G-01', 'ActEast', 'Ac-2-1', 'Ahu-G-01', 'West3', 'Ac-2-1', 'Ahu-G-01', 'ActEast', 'West3', 'Ac-2-1', 'Ahu-G-01', 'ActEast'],
            'Datetime': pd.date_range('2023-03-01', periods=20, freq='D'),
            'Season': [2]*20,
            'Faulty': [True, True, True, False, False, True, False, True, False,  True, False, False, True, False, True, False, True, False, True, False],
            'Fan_status': ['On', 'On', 'On', 'Off', 'Off', 'On', 'Off', 'On', 'Off', 'On', 
                        'Off', 'Off', 'On', 'On', 'Off', 'On', 'On', 'Off', 'On', 'On'],
            'Fan_time_diff': [1000,3000,4000,5000,6000,9000,10000,12000,8000,7000,8800,9000,10000,13000,1000,2000,3000,4000,7000,1000],
            'Predicted': [2000,4000,6000,3000,10000,12000,8000,10000,3000,2000,8200,5700,3000,9500,8100,4300,7500,6700,8900,3000]
        }

        self.df = pd.DataFrame(data)
        self.linechart = LineChartCreator(self.df,1,'ActEast')

    def test_check_filtering(self):
        filtered = self.df[(self.df.building_no == 1) & (self.df.Fan_status == 'On') & (self.df.Zone_name == "ActEast")]
        self.assertEqual(len(filtered),6)
    
    # test for actual and predicted test cases
    def test_traces(self):
        traces = self.linechart.fig.data
        self.assertEqual(len(traces),2)
    
    # test groupby day
    def test_groupby_day(self):
        expected = 6
        self.assertEqual(len(self.linechart.fig.data[0].x),expected)

    # test groupby week 
    def test_groupby_week(self):
        updated_fig = self.linechart.update_fig(self.df,1,'ActEast',agg='week')
        expected = 4
        self.assertEqual(len(updated_fig.data[0].x),expected)

    # test groupby month 
    def test_groupby_month(self):
        updated_fig = self.linechart.update_fig(self.df,1,'ActEast',agg='month')
        expected = 2
        self.assertEqual(len(updated_fig.data[0].x),expected)
if __name__ == "__main__":
    with open("linecharttest_output.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner)