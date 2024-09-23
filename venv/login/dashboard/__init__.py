from dashboard.model_comparison import ModelComparisonDashboard
from dashboard.model_evaluation import ModelEvaluationDashboard

import pandas as pd
from joblib import load

def create_modelevaluation_dashboards(server):
    df = pd.read_parquet("dashboard/summary_all.parquet")
    building1 = ModelEvaluationDashboard(df,1,server,"/dashboard1/")
    building2 = ModelEvaluationDashboard(df,2,server,"/dashboard2/")
    building3 = ModelEvaluationDashboard(df,3,server,"/dashboard3/")

    return building1.app, building2.app, building3.app

def create_modelcomparison_dashboard(server, building_no):
    df = pd.read_parquet("dashboard/model_comparison_test_data.parquet")
    model = load('dashboard/trained_RFC.joblib')
    dashboard = ModelComparisonDashboard(df, building_no, model, server, '/model-comparison/')
    return dashboard.app