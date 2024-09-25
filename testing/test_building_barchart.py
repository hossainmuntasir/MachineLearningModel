import unittest
import pandas as pd
from viz_classes import BarChartCreator

class TestBarChartCreator(unittest.TestCase):
    def setUp(self):
        data = {
            "building_no": [1,1,2,3,1,1],
            'Zone_name': ['ActEast','ActEast','Ac-2-1','Ahu-G-01','West3','West3'],
            'Datetime': ['2023-03-24','2023-03-24','2023-03-24','2023-03-24','2023-03-24','2023-03-24'],
            'Season':[2,2,2,2,2,2],
            'Faulty':[True,True,False,False,True,False],
            'Fan_status':['On','Off','On','Off','On','Off'],
            'Fan_time_diff':[3000,4000,6000,7000,8000,9000],
            'Predicted':[7000,8000,10000,2000,8000,9000]
        }

        self.df = pd.DataFrame(data)
        self.barchart = BarChartCreator(self.df,building_no=1)
    
    # test whether data filetering is functioning or not
    def test_filtering(self):
        filtered_df = self.barchart.df[(self.barchart.df.building_no == 1) & (self.barchart.df.Fan_status == 'On')]
        self.assertEqual(len(filtered_df),2)
    
    # test whether correct number of traces are being returned for each unique zone
    def test_traces(self):
        zones = self.barchart.df[(self.barchart.df.building_no == 1) & (self.barchart.df.Fan_status == 'On')].Zone_name.unique()
        self.assertEqual(len(self.barchart.fig.data)-2,len(zones)*2) # 2 traces expected, 2 is being deducted as they are the traces for the legend
    
    # check whether correct axis titles are generated or not
    def test_layout(self):
        layout = self.barchart.fig.layout
        self.assertIn("<b>Building 1 Zones</b>",layout.xaxis.title.text)
        self.assertIn('<b>Total Time Fan On (hours)</b>',layout.yaxis.title.text)

if __name__ == "__main__":
    with open("barcharttest_output.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner)
