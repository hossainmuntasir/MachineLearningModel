from dashboard.viz_classes import BarChartCreator, LineChartCreator, PieChartCreator
from dash import Dash, html, Input,Output,State, callback, dcc, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as go
import plotly.graph_objects as go
import datetime
import dash
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

import plotly.express as px
import plotly.graph_objects as go
from joblib import dump,load

dashboard_df = pd.read_parquet("dashboard/summary_all.parquet")

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
        self.app = dash.Dash(__name__,server=input_server,url_base_pathname=url_base, external_stylesheets=[dbc.themes.BOOTSTRAP])
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

def create_building_dashboards(server):
    building1 = ModelEvaluationDashboard(dashboard_df,1,server,"/dashboard1/")
    building2 = ModelEvaluationDashboard(dashboard_df,2,server,"/dashboard2/")
    building3 = ModelEvaluationDashboard(dashboard_df,3,server,"/dashboard3/")

    return building1.app, building2.app, building3.app

def create_modelevaluation_dashboard(server):
    df = pd.read_excel("dashboard/prototype_data_extended.xlsx")
    model = load('dashboard/trained_RFC.joblib')

    def design_feature_importance(model):
        fis = pd.DataFrame(zip(model.feature_names_in_, model.feature_importances_), columns=['Features','Score'])
        fis = fis.sort_values('Score')
        fig = px.bar(fis, x='Score', y='Features', orientation='h')
        fig.update_layout(
            # title='<b>Feature Importance</b>', 
            height=400,
            margin=dict(t=10,l=10,r=10,b=10))
        return fig


    def design_confusion_matrix(df):
        cm = confusion_matrix(df.Fan_status, df.prediction, labels=['On', 'Off'])

        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]

        TPR = TP / (TP + FN)
        FNR = FN / (FN + TP)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)

        # Create confusion matrix with TPR, FNR, FPR, TNR
        cm = [[TPR, FNR], [FPR, TNR]]

        # Create the heatmap
        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=['On', 'Off'],
                y=['On', 'Off'],
                colorscale='blues',
                showscale=False))

        # Add text annotations to each cell
        for i in range(len(['On', 'Off'])):
            for j in range(len(['On', 'Off'])):
                fig.add_annotation(
                    x=['On', 'Off'][j], y=['On', 'Off'][i],
                    text=str(round(cm[i][j], 4)),
                    showarrow=False,
                    font=dict(color='white' if cm[i][j] >0.5 else 'black')
                )

        # Update layout
        fig.update_layout(
            width=500,
            height=400,
            # title='<b>Confusion Matrix</b>',
            margin=dict(t=10,l=10,r=10,b=10),
            yaxis=dict(
                title='<b>Actual</b>',
                autorange='reversed',
                tickfont=dict(size=15)
            ),
            xaxis=dict(
                side='top',
                title='<b>Predicted</b>',
                tickfont=dict(size=15)
            )
        )

        return fig

    


    def generate_pastel_colors_plotly(num_colors):
        # Get the 'pastel' color scale from Plotly
        pastel_colors_scale = px.colors.qualitative.Pastel
        
        # Calculate the number of colors to sample from the scale
        num_samples = int((num_colors + len(pastel_colors_scale) - 1) / len(pastel_colors_scale))
        
        # Sample colors from the scale
        pastel_colors = pastel_colors_scale * num_samples
        
        return pastel_colors[:num_colors]

    def get_hover_text(df):
        text = []
        for (_, row) in df.iterrows():
            buil_string = f'<b>Building {row.building_no}</b><br><br>'
            name_string = f'<b>Zone:</b> {row.Zone_name}<br>'
            date_string = f'<b>Date:</b> {row.Date.strftime("%Y-%m-%d")}<br>'
            time_string = f'<b>Time:</b> {row.Time}<br>'
            ztem_string = f'<b>Zone temp:</b> {round(row.Zone_temp,2)}<br>'
            stem_string = f'<b>Slab temp:</b> {round(row.Slab_temp, 2)}'
            text.append(buil_string+name_string+date_string+time_string+ztem_string+stem_string)
        return text

    def design_scatterplot(dfo, colorby):
        # break df down into all possible outcomes
        a = dfo[(dfo.Fan_status=='On')&(dfo.prediction=='On')]
        b = dfo[(dfo.Fan_status=='Off')&(dfo.prediction=='Off')]
        c = dfo[(dfo.Fan_status=='On')&(dfo.prediction=='Off')]
        d = dfo[(dfo.Fan_status=='Off')&(dfo.prediction=='On')]

        # initalise figure
        fig = go.Figure()
        
        # create symbol and color maps
        symbols_map = {'On':'circle', 'Off':'cross'}
        colors = generate_pastel_colors_plotly(dfo[colorby].nunique())
        colors_map = {val:color for val,color in zip(dfo[colorby].unique(), colors)}
        
        # create mapping for int cols and their related string cols 
        # eg. DDOW int=[0,1,2, etc], string=['Monday', 'Tuesday', 'Wedsnesday', etc]
        if f'{colorby}_string' in dfo.columns:
            colorby_dict = {k:v for k,v in zip(dfo[colorby].unique(), dfo[f'{colorby}_string'].unique())}
        else:
            colorby_dict = {val:val for val in dfo[colorby].unique()}
        
        # mapping for explaining symbols for all possible outcomes
        keys = ['Fan ON, Prediction On', 'Fan OFF, Prediction OFF', 'Fan ON, Prediction OFF', 'Fan OFF, Prediction ON']
        symbols = ['circle','cross','circle','cross']
        
        # for each of the possible outcomes
        for i, dfs in enumerate([a,b,c,d]):
            # add empty trace to apply key to legend
            fig.add_trace(
                go.Scattergl(
                    x=[None],
                    y=[None],
                    name=keys[i],
                    mode='markers',
                    # legendgroup=keys[i],
                    legendgroup='keys',
                    legendgrouptitle_text='<b>Keys</b>',
                    marker=dict(
                        symbol=symbols[i],
                        size=12,
                        color='lightgrey',
                        line=dict(
                            width=1,
                            color='black' if i>1 else 'lightgrey'
                        )
                    )
                )
            )
            # add trace with data to plot
            for val in dfs[colorby].unique():            
                temp = dfs[dfs[colorby]==val]
                hover_text = get_hover_text(temp)
                fig.add_trace(
                    go.Scattergl(
                        x=temp.x,
                        y=temp.y,
                        text=hover_text,
                        hoverinfo='text',
                        name=f'Fan: {dfs.Fan_status.unique()[0]}, Prediction: {dfs.prediction.unique()[0]}',
                        showlegend=False,
                        mode='markers',
                        opacity=0.6,
                        marker=dict(
                            size=10,
                            color=colors_map[val],
                            symbol=symbols_map[dfs.Fan_status.unique()[0]],
                            line=dict(
                                width=1 if i>1 else 1,
                                color='black' if i>1 else 'grey'
                            )
                        )
                    )
                )
            # end loop

        # for each of the different colors used in the plot
        for val in sorted(dfo[colorby].unique()):
            # add empty trace to apply color info to legend
            fig.add_trace(
                go.Scattergl(
                    x=[None],
                    y=[None],
                    mode='markers',
                    name=f'{colorby_dict[val]}',
                    # legendgroup=f'{colorby_dict[val]}',
                    legendgroup=colorby,
                    legendgrouptitle_text=f'<b>{colorby}</b>',
                    marker=dict(
                        symbol='square',
                        size=12,
                        color=colors_map[val],
                        line=dict(
                            width=1,
                            color='white'
                        )
                    )
                )
            )

        fig.update_layout(
            xaxis=dict(showticklabels=False), 
            yaxis=dict(showticklabels=False), 
            margin=dict(t=10,l=10,r=10,b=10),
            hoverlabel=dict(align='left'), 
            height=600,
            legend=dict(
                x=-0.01, 
                xanchor='right',)
            )
        
        return fig


    # decompose callback functions
    def update_slider_values(building_no, zone,start_date, end_date):
        dff = df.copy()
        if (building_no):
            dff = dff[dff.building_no==int(building_no)]
        if (zone):
            dff = dff[dff.Zone_name.isin(zone)]
        if (start_date) and (end_date):
            dff = dff[(dff.Date>=start_date)&(dff.Date<=end_date)]
        
        return len(dff), len(dff)

    def update_date_range_values(building_no, zone):
        dff = df.copy()
        if (building_no):
            dff = dff[dff['building_no'] == building_no]
        if (zone):
            dff = dff[dff.Zone_name.isin(zone)]

        return dff.Date.min(), dff.Date.max()

    bar_fi = design_feature_importance(model)
    cm = design_confusion_matrix(df)

    ml_evaluation_app = Dash(__name__,server=server,url_base_pathname='/mlevaluation/',external_stylesheets=[dbc.themes.MORPH])
    
    ml_evaluation_app.layout = dbc.Container([
        dbc.Row([
            html.Div("ML model analytics platform",className="text-primary text-center fs-3"),
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
                dcc.Graph(id="feature-importance-graph",figure=bar_fi)
            ],width=8),
            dbc.Col([
                dcc.Graph(id="confusion-matrix",figure=cm)
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
                dcc.Graph(id='scatter'),
                html.Div([
                    html.Div([
                        html.Div([
                            html.B("Change plot coloring:")
                        ], style={'margin-right': '10px', 'width':'50%'}),
                        html.Div([
                            dcc.RadioItems(
                                id='colorby',
                                options=[
                                    {'label': 'Zone Name', 'value': 'Zone_name'},
                                    {'label': 'Day of Week', 'value': 'DOW'},
                                    {'label': 'Season', 'value': 'Season'}
                                ],
                                value='Zone_name',
                                labelStyle={'display': 'inline-block', 'margin-right': '20px'},  # Arrange labels inline with margin between them
                                inputStyle={'margin-right': '5px'},  # Add margin between buttons
                                style={'display': 'inline-block'}  # Display the radio items inline
                            )
                        ], style={'flex': '1'})
                    ], style={'width':'50%', 'display':'inline-block'}),  # Added display:flex and align-items:center for vertical alignment
                    
                    html.Div([
                        html.Div([
                            html.B("Visible data points:")
                        ], style={'margin-right': '10px', 'width':'30%'}),
                        html.Div([
                            dcc.Slider(
                                min=0,
                                max=df.shape[0],
                                value=round(df.shape[0]/3,2),
                                id="size_slider",
                                tooltip={"placement": "bottom", "always_visible": False}
                            )
                        ], style={'flex': '1'})
                    ], style={'width':'50%', 'display':'inline-block'})  # Added display:flex and align-items:center for vertical alignment
                ], style={'display':'flex', 'width':'100%', 'background-color':'white', 'padding-left':'220px'})

            ],width=9),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H3(["Filter"], style={'text-align':'center'}),
                    
                        html.H5("Building"),
                        dcc.Dropdown(options=[
                            {'label':'Building 1', 'value':1},
                            {'label':'Building 2', 'value':2},
                            {'label':'Building 3', 'value':3}], 
                            value=3, 
                            id='building_no', 
                            placeholder="Select a Building"),
                        html.Br(),
                        
                        html.H5("Zones"),
                        dcc.Dropdown(
                            options=[{'label':val, 'value':val} for val in sorted(df.Zone_name.unique())],
                            id='zone',
                            placeholder='Select a Zone',
                            multi=True
                        ),
                        html.Br(),
                        
                        html.H5("Season"),
                        dcc.Dropdown(
                            options=[{'label':status, 'value':value} for status,value in zip(df['Season_string'].unique(), df.Season.unique())],
                            id='season',
                            placeholder='Select a Season',
                            multi=True
                        ),
                        html.Br(),
                        
                        html.H5("Date Range"),
                        dcc.DatePickerRange(
                            min_date_allowed=df.Date.min(),
                            max_date_allowed=df.Date.max(),
                            start_date=df.Date.min(),
                            end_date=df.Date.max(),
                            id='date-picker-range',
                            clearable=True,
                            updatemode='bothdates',
                            display_format="MMM Do, YY",
                            start_date_placeholder_text="MMM Do, YY",
                            end_date_placeholder_text="MMM Do, YY"
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
                    ], 
                    style={'background-color':'#D9E3F1',
                        'padding':5}),
                ], 
                style={
                    'background-color':'white',
                    'padding':10
                })
            ],width=3),
        ]),
    ],fluid=True)


    @callback(
        Output('size_slider','max'),
        Output('size_slider','value'), 
        Input('building_no','value'),
        Input('zone', 'value'),
        Input('season','value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date')    
    )
    def update_slider(building_no, zone,season,start_date, end_date):
        return update_slider_values(building_no, zone,season,start_date, end_date)

    @callback(
        Output('date-picker-range', 'start_date'),
        Output('date-picker-range', 'end_date'),
        Input('building_no', 'value'),
        Input('zone', 'value'),
        Input('season', 'value'),
    )
    def update_date_range(building_no, zone, season):
        return update_date_range_values(building_no, zone, season)

    @callback(
        Output(component_id='scatter', component_property='figure'),

        Input('colorby','value'), 
        Input('building_no','value'),
        Input('zone', 'value'),
        Input('season','value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('time_slider', 'value'),
        Input('size_slider', 'value'),
        

    )
    def update(colorby, building_no, zone, season, start_date, end_date, time_slider, size_slider):
        dff = df.copy()
        if (building_no):
            dff = dff[dff.building_no==int(building_no)]
        if (zone):
            dff = dff[dff.Zone_name.isin(zone)]
        if (season):
            dff = dff[dff.Season.isin(season)]
        if (start_date) and (end_date):
            dff = dff[(dff.Date>=start_date)&(dff.Date<=end_date)]
        if (time_slider):
            dff = dff[(dff.Datetime.dt.hour>=time_slider[0])&(dff.Datetime.dt.hour<time_slider[1])]
        if (size_slider):
            dff = dff.iloc[:size_slider]
        
        return design_scatterplot(dff, colorby)
    return ml_evaluation_app