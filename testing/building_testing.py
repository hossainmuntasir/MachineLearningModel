import pandas as pd
import dash 
from dash import dcc, html, Dash, Input, Output,State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go 
from viz_classes import BarChartCreator,PieChartCreator,LineChartCreator
import datetime
import time

def test(dash_duo):
    # set initial values for charts
    df = pd.read_parquet("../summary_all.parquet")
    building_no = 1
    df = df[(df.building_no==building_no)&(df.Fan_status=='On')]
    sel_zone = df.Zone_name.unique()[0]

    # initialize charts
    initial_bar = BarChartCreator(df,building_no)
    initial_pie = PieChartCreator(df,building_no,sel_zone)
    initial_line = LineChartCreator(df,building_no,sel_zone)

    app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dbc.Container([
        dbc.Row([
            html.Div(building_no, id='building_no', hidden=True),
            html.H1(f"Building {building_no} Savings"),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="bar-graph", figure=initial_bar.fig, style={"height": "300px"})
            ], width=9),
            dbc.Col([
                dcc.Graph(id="pie-chart", figure=initial_pie.fig, style={"height": "300px"})
            ], width=3)
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="line-chart", figure=initial_line.fig, style={"height": "300px"})
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
                    html.Div(id="test-initialize"),
                    html.P([
                        html.B('Date Range:'),
                        html.Br(),
                        dcc.DatePickerRange(
                            min_date_allowed=df.Datetime.min(),
                            max_date_allowed=df.Datetime.max(),
                            start_date=df.Datetime.min(),
                            end_date=df.Datetime.max(),
                            id='date-picker-range',
                            clearable=False,
                            updatemode='bothdates',
                            display_format="MMM Do, YY",
                            start_date_placeholder_text="MMM Do, YY",
                            end_date_placeholder_text="MMM Do, YY"
                        )
                    ])
                ])
            ])
        ]),
    ], fluid=True)

    @callback(
        [Output("bar-graph", "figure"),
        Output("line-chart", "figure"),
        Output("pie-chart", "figure"),
        Output('bar_total', 'children'),
        Output('bar_predicted', 'children'),
        Output("test-initialize","children")],
        [Input("agg-type", "value"),
        Input('bar-graph', 'clickData'),
        Input("date-picker-range", 'start_date'),
        Input("date-picker-range", 'end_date')],
    )
    def update(type_agg, clickData, start_date, end_date):
        start = [int(x) for x in start_date.split("-")]
        end = [int(x) for x in end_date.split("-")]
        selected_zone = df.Zone_name.unique()[0]
        zone = clickData['points'][0]['x'] if clickData else selected_zone
        date_range_var = [datetime.date(start[0], start[1], start[2]), datetime.date(end[0], end[1], end[2])]

        updated_bar = initial_bar.update_fig(df, building_no=building_no, date_range=date_range_var)
        updated_line = initial_line.update_fig(df, building_no=building_no, zone_name=zone, date_range=date_range_var, agg=type_agg)
        updated_pie = initial_pie.update_fig(df, building_no=building_no, zone_name=zone, date_range=date_range_var)

        bar_vals = []
        # Update bar chart border if needed
        ctx = dash.callback_context
        if ctx.triggered[0]['prop_id'] == 'bar-graph.clickData':
            name = clickData['points'][0]['label']
            selected_zone = name
        else:
            name = None

        if selected_zone:
            for t in updated_bar.data:
                if t.name == selected_zone:
                    t.marker.opacity = 1
                    t.marker.line.width = 1
                    bar_vals.append(round(t.y[0],2))
                else:
                    t.marker.opacity = 0.6
                    t.marker.line.width = 0.5
        date_string = f"{start_date} {end_date} "
        string = f"{type_agg} {selected_zone} {date_string}"
        return updated_bar, updated_line, updated_pie, bar_vals[0], bar_vals[1], string

    dash_duo.start_server(app)

    time.sleep(15)
    # Test initialized dropdown 
    assert dash_duo.find_element("#test-initialize").text == "day Acu101 2023-04-26 2024-03-22"

    # Verify that there are no errors in the browser console
    assert dash_duo.get_logs() == [], "Browser console should contain no error"