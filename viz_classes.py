import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class BarChartCreator:
    def __init__(self, df, building_no, date_range=None):
        self._df = df.copy()
        self._building_no = building_no
        self._date_range = date_range
        
        self.mark1_color = px.colors.qualitative.Plotly[1]
        self.mark2_color = px.colors.qualitative.Plotly[2]
        self.line_color = 'black'
        self.line_thick = 0.5
        self.opacity1 = 1
        self.opacity2 = 1
        
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
            df = df[(df.Datetime.dt.date>=date_range[0])&(df.Datetime.dt.date<date_range[1])]
           
        zones = df.Zone_name.unique()
         
        fig = go.Figure()
        
        for zone in zones:
            temp = df[df.Zone_name==zone]
            
            # Add main plots
            fig.add_trace(
                go.Bar(
                    x=[zone],
                    y=[temp.Fan_time_diff.sum()],
                    marker=dict(
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
                    y=[temp.Predicted.sum()],
                    marker=dict(
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
            xaxis=dict(title=f'<b>Building {self.building_no} Zones</b>'), 
            yaxis=dict(title=f'<b>Total Time Fan On (mins)</b>'), 
            barmode='overlay')
        
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
            xaxis=dict(title=f'<b>{self.zone_name} All time Fan Usage</b>'), 
            yaxis=dict(title=f'<b>Total {agg_map[agg]} Fan On Time (mins)</b>'),
            hovermode='x unified'
            )
        
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
                labels=['Savings','Predicted Fan Usage'],
                values=[df.Fan_time_diff.sum()-df.Predicted.sum(),df.Predicted.sum()],
                hole=0.2,
                pull=[0, 0.1]
            )
        )
        
        return fig