import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime
from sklearn.metrics import confusion_matrix

class ConfusionMatrixCreator:
    def __init__(self, df):
        self.df = df.copy()
        self.fig = self.update_fig()
        
    def update_plot_styling(self):
        pass
    
    def update_fig(self):
        # df = self.df.copy()
        cm = confusion_matrix(self.df.Fan_status, self.df.Predicted, labels=['On', 'Off'])

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
            plot_bgcolor='#181A20',
            paper_bgcolor='#181A20',
            yaxis=dict(
                title='<b>Actual</b>',
                autorange='reversed',
                titlefont=dict(
                    color='white'
                ),
                tickfont=dict(
                    color='white',
                    size=15)
            ),
            xaxis=dict(
                side='top',
                title='<b>Predicted</b>',
                tickfont=dict(
                    color='white',
                    size=15),
                titlefont=dict(color='white')
            )
        )

        return fig


class FeatureImportanceCreator:
    def __init__(self, model):
        self.model = model
        self.fig = self.update_fig()
    
    def update_fig(self):
        fis = pd.DataFrame(zip(self.model.feature_names_in_, self.model.feature_importances_), columns=['Features','Score'])
        fis = fis.sort_values('Score')
        fig = px.bar(fis, x='Score', y='Features', orientation='h')
        fig.update_layout(
            # title='<b>Feature Importance</b>', 
            plot_bgcolor='#181A20',
            paper_bgcolor='#181A20',
            xaxis=dict(
                tickfont=dict(color='white'),
                titlefont=dict(color='white')), 
            yaxis=dict(
                tickfont=dict(color='white'),
                titlefont=dict(color='white')),
            height=400,
            margin=dict(t=10,l=10,r=10,b=10))
        
        return fig


class ScatterPlotCreator:
    def __init__(self, df, date):
        self.df = df.copy()
        self.fig = self.update_fig(date)
        
    @staticmethod
    def generate_pastel_colors_plotly(num_colors):
        # Get the 'pastel' color scale from Plotly
        pastel_colors_scale = px.colors.qualitative.Pastel
        
        # Calculate the number of colors to sample from the scale
        num_samples = int((num_colors + len(pastel_colors_scale) - 1) / len(pastel_colors_scale))
        
        # Sample colors from the scale
        pastel_colors = pastel_colors_scale * num_samples
        
        return pastel_colors[:num_colors]

    @staticmethod
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

    def update_fig(self, date, zone=None, time=None):
        if not isinstance(date, datetime.date):
            date = pd.Timestamp(date).date()
            
        dfo = self.df[(self.df.Datetime.dt.date==date)]
        
        zone = [zone] if type(zone) == str else zone
        if (zone):
            dfo = dfo[(dfo.Zone_name.isin(zone))]
        if (time):
            dfo = dfo[(dfo.Datetime.dt.hour>=time[0])&(dfo.Datetime.dt.hour<time[1])]

        # break df down into all possible outcomes
        a = dfo[(dfo.Fan_status=='On')&(dfo.Predicted=='On')]
        b = dfo[(dfo.Fan_status=='Off')&(dfo.Predicted=='Off')]
        c = dfo[(dfo.Fan_status=='On')&(dfo.Predicted=='Off')]
        d = dfo[(dfo.Fan_status=='Off')&(dfo.Predicted=='On')]

        # initalise figure
        fig = go.Figure()
        
        # create symbol and color maps
        symbols_map = {'On':'circle', 'Off':'cross'}
        colors = ScatterPlotCreator.generate_pastel_colors_plotly(dfo.Zone_name.nunique())
        colors_map = {val:color for val,color in zip(dfo.Zone_name.unique(), colors)}
        colorby_dict = {val:val for val in dfo.Zone_name.unique()}
        
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
                            width=1.5 if i>1 else 0.5,
                            color='white' if i>1 else 'lightgrey'
                        )
                    )
                )
            )
            # add trace with data to plot
            for val in dfs.Zone_name.unique():            
                temp = dfs[dfs.Zone_name==val]
                hover_text = ScatterPlotCreator.get_hover_text(temp)
                fig.add_trace(
                    go.Scattergl(
                        x=temp.x,
                        y=temp.y,
                        text=hover_text,
                        hoverinfo='text',
                        name=f'Fan: {dfs.Fan_status.unique()[0]}, Prediction: {dfs.Predicted.unique()[0]}',
                        showlegend=False,
                        mode='markers',
                        opacity=0.6,
                        marker=dict(
                            size=10,
                            color=colors_map[val],
                            symbol=symbols_map[dfs.Fan_status.unique()[0]],
                            line=dict(
                                width=1.5 if i>1 else 0.5,
                                color='black' if i>1 else 'lightgrey'
                            )
                        )
                    )
                )
            # end loop

        # for each of the different colors used in the plot
        for val in sorted(dfo.Zone_name.unique()):
            # add empty trace to apply color info to legend
            fig.add_trace(
                go.Scattergl(
                    x=[None],
                    y=[None],
                    mode='markers',
                    name=f'{colorby_dict[val]}',
                    # legendgroup=f'{colorby_dict[val]}',
                    legendgroup='Zone_name',
                    legendgrouptitle_text='<b>Zone Name</b>',
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
            plot_bgcolor='#181A20',
            paper_bgcolor='#181A20',
            margin=dict(t=10,l=10,r=10,b=10),
            hoverlabel=dict(align='left'), 
            height=600,
            legend=dict(
                font=dict(color='white'),
                x=-0.01, 
                xanchor='right',)
            )
        
        return fig

class BarChartCreator:
    def __init__(self, df, building_no, date_range=None):
        self._df = df.copy()
        self._building_no = building_no
        self._date_range = date_range
        
        self.mark1_color = px.colors.qualitative.Plotly[1]
        self.mark2_color = px.colors.qualitative.Plotly[2]
        self.line_color = 'black'
        self.line_thick = 0.5
        self.opacity1 = 0.6
        self.opacity2 = 0.6
        
        self._fig = self.update_fig(df, building_no, date_range)
        
        
    @property
    def df(self):
        return self._df
    
    @property
    def building_no(self):
        return self._building_no
    
    @property
    def date_range(self):
        return self._date_range
    
    @property
    def fig(self):
        return self._fig
    
    def update_plot_styling(self, mark1_color=None, mark2_color=None, line_color=None, line_thick=None, opacity1=None, opacity2=None):
        if mark1_color is not None:
            self.mark1_color = mark1_color
        if mark2_color is not None:
            self.mark2_color = mark2_color
        if line_color is not None:
            self.line_color = line_color
        if line_thick is not None:
            self.line_thick = line_thick
        if opacity1 is not None:
            self.opacity1 = opacity1
        if opacity2 is not None:
            self.opacity2 = opacity2
    
    def update_fig(self, df, building_no, date_range=None):
        df = df[(df.building_no==building_no)&(df.Fan_status=='On')]
        if date_range is not None:
            temp = pd.Series(pd.to_datetime(df.Datetime, errors='coerce'), index=df.index, name='Datetime')
            df = df[(temp.dt.date>=date_range[0])&(temp.dt.date<date_range[1])]
           
        zones = df.Zone_name.unique()
        fig = go.Figure()
        
        for zone in zones:
            temp = df[df.Zone_name==zone]
            
            # Add main plots
            fig.add_trace(
                go.Bar(
                    x=[zone],
                    y=[temp.Fan_time_diff.sum()/3600],
                    marker=dict(
                        cornerradius=5,
                        opacity=self.opacity1,
                        color=self.mark1_color,
                        line=dict(
                            color=self.line_color,
                            width=self.line_thick
                        )
                    ),
                    opacity=self.opacity1,
                    name=zone,
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[zone],
                    y=[temp.Predicted.sum()/3600],
                    width=0.65,
                    marker=dict(
                        cornerradius=5,
                        opacity=self.opacity2,
                        color=self.mark2_color,
                        line=dict(
                            color=self.line_color,
                            width=self.line_thick
                        )
                    ),
                    opacity=self.opacity2,
                    name=zone,
                    showlegend=False,
                )
            )
        
        # Add empty plots for legend names
        fig.add_trace(
            go.Bar(x=[None],
            y=[None],
            name='Actual',
            marker=dict(color=self.mark1_color)
        ))
        fig.add_trace(
            go.Bar(x=[None],
            y=[None],
            name='Predicted',
            marker=dict(color=self.mark2_color)
        ))

        # Add axis titles
        fig.update_layout(
            xaxis=dict(
                title=f'<b>Building {self.building_no} Zones</b>',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')), 
            yaxis=dict(
                title=f'<b>Total Time Fan On (hours)</b>',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')), 
            plot_bgcolor='#181A20',
            paper_bgcolor='#181A20',
            barmode='overlay',
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                font=dict(color='white'),
                orientation='h',
                x=1,
                y=1,
                xanchor='right',
                yanchor='bottom'
            ))
        
        return fig
    
class LineChartCreator:
    def __init__(self, df, building_no, zone_name, date_range=None, agg='day'):
        self._df = df.copy()
        self._building_no = building_no
        self._date_range = date_range
        self._zone_name = zone_name
        
        self.line1_color = px.colors.qualitative.Plotly[1]
        self.line2_color = px.colors.qualitative.Plotly[2]
        self.marker_thick = 6
        self.marker_line_color = 'black'
        self.line_thick = 0.5
        self.opacity1 = 1
        self.mode = 'markers+lines'
        
        self.shading_color = px.colors.qualitative.Pastel[5]
        
        self._fig = self.update_fig(df, building_no, zone_name, date_range, agg)
        
        
    @property
    def df(self):
        return self._df
    
    @property
    def building_no(self):
        return self._building_no
    
    @property
    def date_range(self):
        return self._date_range
    
    @property
    def zone_name(self):
        return self._zone_name
    
    @property
    def fig(self):
        return self._fig
    
    def update_plot_styling(self, line1_color=None, line2_color=None, marker_line_color=None, marker_thick=None, line_thick=None, opacity1=None, mode=None, shading_color=None):
        if line1_color is not None:
            self.line1_color = line1_color
        if line2_color is not None:
            self.line2_color = line2_color
        if marker_line_color is not None:
            self.marker_line_color = marker_line_color
        if marker_thick is not None:
            self.marker_thick = marker_thick
        if line2_color is not None:
            self.line2_color = line2_color
        if line_thick is not None:
            self.line_thick = line_thick
        if opacity1 is not None:
            self.opacity1 = opacity1
        if mode is not None:
            self.mode = mode
        if shading_color is not None:
            self.shading_color = shading_color
    
    def update_fig(self, df, building_no, zone_name, date_range=None, agg='day'):
        # Suppress Plotly Deprecated dt warning
        import warnings
        warnings.simplefilter("ignore", category=FutureWarning)

        # Subset data
        df = df[(df.building_no==building_no)&(df.Fan_status=='On')&(df.Zone_name==zone_name)]
        
        # Aggregate data
        temp = pd.Series(pd.to_datetime(df.Datetime, errors='coerce').values, index=df.index, name='Datetime')
        # grouping checkbox values
        group_mapping = {
            'day': temp.dt.date,
            'week': temp.dt.to_period('W').apply(lambda r: r.start_time).dt.date,
            'month': temp.dt.to_period('M').apply(lambda r: r.start_time).dt.date
        }
        df = df.groupby(['building_no','Zone_name','Season','Faulty','Fan_status',group_mapping[agg]])[['Fan_time_diff','Predicted']].sum().reset_index().sort_values(by='Datetime')
        
        fig = go.Figure()
            
        # Add main plots
        fig.add_trace(
            go.Scattergl(
                x=df.Datetime,
                y=df.Fan_time_diff,
                marker=dict(
                    color=self.line1_color,
                    size=self.marker_thick,
                    line=dict(
                        color=self.marker_line_color,
                        width=self.line_thick
                    )
                ),
                mode=self.mode,
                name='Actual'
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=df.Datetime,
                y=df.Predicted,
                marker=dict(
                    color=self.line2_color,
                    size=self.marker_thick,
                    line=dict(
                        color=self.marker_line_color,
                        width=self.line_thick
                    )
                ),
                mode=self.mode,
                name='Predicted'
            )
        )
        
        # Add background of Date Range selection
        if date_range is None:
            date_range = [df.Datetime.min(), df.Datetime.max()]
            
        fig.add_shape(
            type='rect',
            xref='x', yref='paper',
            x0=date_range[0], x1=date_range[1], y0=0, y1=1,
            fillcolor=self.shading_color, opacity=0.2, layer='below', line_width=0,
        )

        # Add axis titles
        agg_map = {'day':'Daily', 'week':'Weekly', 'month':'Monthly'}
        fig.update_layout(
            xaxis=dict(title=f'<b>{zone_name} All time Fan Usage</b>',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')), 
            yaxis=dict(title=f'<b>Total {agg_map[agg]} Fan On Time (mins)</b>',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')),
            hovermode='x unified',
            plot_bgcolor='#181A20',
            paper_bgcolor='#181A20',
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                font=dict(color='white'),
                orientation='h',
                x=1,
                y=1,
                xanchor='right',
                yanchor='bottom'
            ))
        
        return fig
    
class PieChartCreator:
    def __init__(self, df, building_no, zone_name, date_range=None):
        self._df = df.copy()
        self._building_no = building_no
        self._date_range = date_range
        self._zone_name = zone_name    
        self._fig = self.update_fig(df, building_no, zone_name, date_range)
        
        
    @property
    def df(self):
        return self._df
    
    @property
    def building_no(self):
        return self._building_no
    
    @property
    def date_range(self):
        return self._date_range
    
    @property
    def zone_name(self):
        return self._zone_name
    
    @property
    def fig(self):
        return self._fig
    
    def update_fig(self, df, building_no, zone_name, date_range=None):
        # Suppress Plotly Deprecated dt warning
        import warnings
        warnings.simplefilter("ignore", category=FutureWarning)

        # Subset data
        df = df[(df.building_no==building_no)&(df.Fan_status=='On')&(df.Zone_name==zone_name)]
        if date_range is not None:
            temp = pd.Series(pd.to_datetime(df.Datetime, errors='coerce'), index=df.index, name='Datetime')
            df = df[(temp.dt.date>=date_range[0])&(temp.dt.date<date_range[1])]

        fig = go.Figure()
            
        # Add main plots
        fig.add_trace(
            go.Pie(
                labels=['Savings','Regular Usage'],
                values=[df.Fan_time_diff.sum()-df.Predicted.sum(),df.Predicted.sum()],
                hole=0.2,
                pull=[0, 0.1]
            )
        )
        
        fig.update_layout(
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='#181A20',
            paper_bgcolor='#181A20',
            legend=dict(
                font=dict(color='white'),
                orientation='h',
                x=0.5,
                y=1,
                xanchor='center',
                yanchor='bottom'
            ))
        
        return fig

