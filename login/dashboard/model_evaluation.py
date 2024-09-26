from dashboard.viz_classes import BarChartCreator, LineChartCreator, PieChartCreator
from dash import Dash, html, Input,Output,State, callback, dcc, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as go
import plotly.graph_objects as go
import datetime
import dash
import plotly.graph_objects as go

class ModelEvaluationDashboard:
    def __init__(self, df, building_no,input_server,url_base):
        self.df = df[(df.building_no==building_no)&(df.Fan_status=='On')]
        self.building_no = building_no
        self.selected_zone = self.df.Zone_name.unique()[0]

        # Initialize charts
        self.initial_bar = BarChartCreator(df, building_no)
        self.initial_pie = PieChartCreator(df, building_no, self.selected_zone)
        self.initial_line = LineChartCreator(df, building_no, self.selected_zone)

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
                html.Div(self.building_no, id='building_no', hidden=True),
                html.H1(f"Building {self.building_no} Savings"),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="bar-graph", figure=self.initial_bar.fig, style={"height": "300px"})
                ], width=9),
                dbc.Col([
                    dcc.Graph(id="pie-chart", figure=self.initial_pie.fig, style={"height": "300px"})
                ], width=3)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="line-chart", figure=self.initial_line.fig, style={"height": "300px"})
                ], width=9),
                dbc.Col([
                    dbc.Row([
                        html.Div(                        
                            dbc.Table([
                                html.Thead(html.Tr([html.Th('Total Usage'), html.Th('Predicted Usage')])),
                                html.Tbody(html.Tr([html.Td(id='bar_total'), html.Td(id='bar_predicted')])),
                            ], color='secondary', bordered=True, id='table', size='lg'
                            ), id='table_wrapper'
                        )
                    ]),
                    dbc.Row([
                        html.P([
                            html.B('Group fan status by:'),
                            html.Br(),
                            dcc.Dropdown(
                                options=[
                                    {'label': 'Day', 'value': 'day'},
                                    {'label': 'Week', 'value': 'week'},
                                    {'label': 'Month', 'value': 'month'}
                                ],
                                clearable=False,
                                value="day",
                                id="agg-type",
                            )
                        ]),
                        html.P([
                            html.B('Date Range:'),
                            html.Br(),
                            dcc.DatePickerRange(
                                min_date_allowed=self.df.Datetime.min(),
                                max_date_allowed=self.df.Datetime.max(),
                                start_date=self.df.Datetime.min(),
                                end_date=self.df.Datetime.max(),
                                id='date-picker-range',
                                clearable=False,
                                updatemode='bothdates',
                                display_format="MMM Do, YY",
                                start_date_placeholder_text="MMM Do, YY",
                                end_date_placeholder_text="MMM Do, YY"
                            )
                        ]),
                    ])
                ])
            ]),
        ], fluid=True)

    def set_callbacks(self):
        @self.app.callback(
            [Output("bar-graph", "figure"),
             Output("line-chart", "figure"),
             Output("pie-chart", "figure"),
             Output('bar_total', 'children'),
             Output('bar_predicted', 'children')],
            [Input("agg-type", "value"),
             Input('bar-graph', 'clickData'),
             Input("date-picker-range", 'start_date'),
             Input("date-picker-range", 'end_date')],
        )
        def update(type_agg, clickData, start_date, end_date):
            start = [int(x) for x in start_date.split("-")]
            end = [int(x) for x in end_date.split("-")]
            zone = clickData['points'][0]['x'] if clickData else self.selected_zone
            date_range_var = [datetime.date(start[0], start[1], start[2]), datetime.date(end[0], end[1], end[2])]

            updated_bar = self.initial_bar.update_fig(self.df, building_no=self.building_no, date_range=date_range_var)
            updated_line = self.initial_line.update_fig(self.df, building_no=self.building_no, zone_name=zone, date_range=date_range_var, agg=type_agg)
            updated_pie = self.initial_pie.update_fig(self.df, building_no=self.building_no, zone_name=zone, date_range=date_range_var)

            bar_vals = []
            # Update bar chart border if needed
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'bar-graph.clickData':
                name = clickData['points'][0]['label']
                self.selected_zone = name
            else:
                name = None

            if self.selected_zone:
                for t in updated_bar.data:
                    if t.name == self.selected_zone:
                        t.marker.opacity = 1
                        t.marker.line.width = 1
                        bar_vals.append(round(t.y[0],2))
                    else:
                        t.marker.opacity = 0.6
                        t.marker.line.width = 0.5

            return updated_bar, updated_line, updated_pie, bar_vals[0], bar_vals[1]

    def run(self):
        self.app.run_server(debug=True, port=8000)