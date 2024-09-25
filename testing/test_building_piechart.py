import unittest 
import pandas as pd 
from viz_classes import PieChartCreator

class TestPieChart(unittest.TestCase):
    def setUp(self):
        data = {
            "building_no": [1,1,2,3,1,1],
            'Zone_name': ['ActEast','ActEast','Ac-2-1','Ahu-G-01','West3','West3'],
            'Datetime': ['2023-03-24','2023-03-24','2023-03-24','2023-03-24','2023-03-24','2023-03-24'],
            'Season':[2,2,2,2,2,2],
            'Faulty':[True,True,False,False,True,False],
            'Fan_status':['On','On','On','Off','On','Off'],
            'Fan_time_diff':[3000,4000,6000,7000,8000,9000],
            'Predicted':[7000,8000,10000,2000,8000,9000]
        }

        self.df = pd.DataFrame(data)
        self.piechart = PieChartCreator(self.df,1,'ActEast')

    def test_filtering(self):
        filtered = self.df[(self.df.building_no == 1) & (self.df.Fan_status == 'On') & (self.df.Zone_name == "ActEast")]
        self.assertEqual(len(filtered),2)
    
    def test_chart_values(self):
        savings = self.piechart.fig.data[0].values[0]
        regular_usage = self.piechart.fig.data[0].values[1]
        test_savings = (3000 + 4000) - (7000 + 8000)
        test_regular_usage = 7000 + 8000
        self.assertEqual(savings,test_savings)
        self.assertEqual(regular_usage,test_regular_usage)

if __name__ == "__main__":
    with open("piecharttest_output.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner)

    