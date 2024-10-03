from dashboard.viz_classes import BarChartCreator, LineChartCreator, PieChartCreator, FeatureImportanceCreator, ConfusionMatrixCreator, ScatterPlotCreator
from dash import Dash, html, Input,Output,State, callback, dcc, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as go
import plotly.graph_objects as go
import datetime
import dash
import plotly.graph_objects as go

class ModelComparisonDashboard:
    def __init__(self, df, building_no, model, input_server,url_base):
        self.df = df[df.building_no==building_no]

        # Initialize charts
        self.initial_features = FeatureImportanceCreator(model)
        self.initial_confusion_mat = ConfusionMatrixCreator(df)
        self.initial_scatter = ScatterPlotCreator(df, self.df.Datetime.dt.date.min())

        # Initialize app
        self.app = dash.Dash(__name__, 
                             server=input_server, 
                             url_base_pathname=url_base, 
                             external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.layout = self.build()
        self.set_callbacks()

    def build(self):
        return dbc.Container([
            dbc.Row([
                html.Div("ML Model Analytics Platform",className="text-primary text-center fs-3"),
                dcc.Store(id='date_range', storage_type='session'),
                dcc.Store(id='df_size', storage_type='session')
            ]),
            dbc.Row([
                html.Br(style={'background-color':'black'}),
                dbc.Col([
                    html.H2('Feature Importance')
                ],width=8),
                        dbc.Col([
                    html.H2('Confusion Matrix')
                ],width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="feature-importance-graph",figure=self.initial_features.fig)
                ],width=8),
                dbc.Col([
                    dcc.Graph(id="confusion-matrix",figure=self.initial_confusion_mat.fig)
                ],width=4),
                html.Br(),
            ]),
            dbc.Row([
                html.Br(),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H2('Building Data')
                ],width=9),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='scatter', figure=self.initial_scatter.fig),
                ],width=9),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.H3(["Filter"], style={'text-align':'center'}),
                            html.H5("Zones"),
                            dcc.Dropdown(
                                options=[{'label':val, 'value':val} for val in sorted(self.df.Zone_name.unique())],
                                id='zone',
                                value=self.df.Zone_name.unique(),
                                multi=True
                            ),
                            html.Br(),
                            
                            html.H5("Date"),
                            dcc.Dropdown(
                                options=[{'label':value.strftime('%d %b %Y'), 'value':value} for value in self.df.Datetime.sort_values().dt.date.unique()],
                                id='date-picker',
                                value=self.df.Datetime.sort_values().dt.date.unique()[0],
                                multi=False, clearable=False
                            ),
                            html.Br(),
                            html.Br(),
                            
                            html.H5('Time Range'),
                            dcc.RangeSlider(
                                id='time_slider',
                                marks={i: {'label': f'0{i}:00' if i<10 else f'{i}:00'} for i in range(0, 25, 3)},  # Label for each hour
                                min=0,
                                max=24,
                                step=1,
                                value=[0, 24]  # Initial range from midnight to 11:59 PM
                            ),
                            html.Br(),
                        ]),
                    ])
                ],width=3),
            ]),
        ],fluid=True)

    def set_callbacks(self):
        @self.app.callback(
            Output(component_id='scatter', component_property='figure'),
            [Input('zone', 'value'),
            Input('date-picker', 'value'),
            Input('time_slider', 'value')],
        )
        def update(zone, date, time_slider):
            updated_scatter = self.initial_scatter.update_fig(date, zone, time_slider)
            return updated_scatter

    def run(self):
        self.app.run_server(debug=True, port=8000)