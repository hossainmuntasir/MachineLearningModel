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

def create_modelcomparison_dashboard(server):
    def load_dashboard(df, model, server, url, column_name):
        df['Predicted'] = df[column_name]
        return ModelComparisonDashboard(df, model, server, url)
    
    df1 = pd.read_parquet("dashboard/building1_predicted.parquet")
    df2 = pd.read_parquet("dashboard/building2_predicted.parquet")
    df3 = pd.read_parquet("dashboard/building3_predicted.parquet")
    
    model_rfc1 = load('dashboard/models/randomforest_building1_tuned.joblib')
    model_rfc2 = load('dashboard/models/randomforest_building2_tuned.joblib')
    model_rfc3 = load('dashboard/models/randomforest_building3_tuned.joblib')
    model_xgb1 = load('dashboard/models/trained_xgb_model1.joblib')
    model_xgb2 = load('dashboard/models/trained_xgb_model2.joblib')
    model_xgb3 = load('dashboard/models/trained_xgb_model3.joblib')
    model_hgb1 = load('dashboard/models/histgb_building1_tuned.joblib')
    model_hgb2 = load('dashboard/models/histgb_building2_tuned.joblib')
    model_hgb3 = load('dashboard/models/histgb_building3_tuned.joblib')
    
    dashboard_rfc1 = load_dashboard(df1, model_rfc1, server, '/modelcomparison_rfc1/', 'PredictedRFC')
    dashboard_rfc2 = load_dashboard(df2, model_rfc2, server, '/modelcomparison_rfc2/', 'PredictedRFC')
    dashboard_rfc3 = load_dashboard(df3, model_rfc3, server, '/modelcomparison_rfc3/', 'PredictedRFC')
    dashboard_xgb1 = load_dashboard(df1, model_xgb1, server, '/modelcomparison_xgb1/', 'PredictedXGB')
    dashboard_xgb2 = load_dashboard(df2, model_xgb2, server, '/modelcomparison_xgb2/', 'PredictedXGB')
    dashboard_xgb3 = load_dashboard(df3, model_xgb3, server, '/modelcomparison_xgb3/', 'PredictedXGB')
    dashboard_hgb1 = load_dashboard(df1, model_hgb1, server, '/modelcomparison_hgb1/', 'PredictedHGB')
    dashboard_hgb2 = load_dashboard(df2, model_hgb2, server, '/modelcomparison_hgb2/', 'PredictedHGB')
    dashboard_hgb3 = load_dashboard(df3, model_hgb3, server, '/modelcomparison_hgb3/', 'PredictedHGB')
    
    return \
        dashboard_rfc1.app, \
        dashboard_rfc2.app, \
        dashboard_rfc3.app, \
        dashboard_xgb1.app, \
        dashboard_xgb2.app, \
        dashboard_xgb3.app, \
        dashboard_hgb1.app, \
        dashboard_hgb2.app, \
        dashboard_hgb3.app