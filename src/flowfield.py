#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
import numpy as np
import plotly as ply
import plotly.graph_objects as go
import plotly.figure_factory as ff
import src.plotly_streamline as strline
import plotly.io as pio
from plotly.subplots import make_subplots
from src.commondicts import TYPE_NAME_DICT, LONG_NAME_DICT
from src.commonfuncs import flow_element_type
import potentialflowvisualizer as pfv

pio.renderers.default = (
    "browser"  # Feel free to disable this if you're running in notebook mode or prefer a different frontend.
)

def line_color(object):
    try:
        color = "green" if object.strength > 0 else "red"
    except AttributeError:
        color = "black"

    return color

def dot_size(object):
    try:
        strength = abs(object.strength) / 10
    except AttributeError:
        strength = 1

    return 10 + np.tanh(strength) * 10

def line_width(object):
    try:
        strength = abs(object.strength) / 10
    except AttributeError:
        strength = 1

    return 10 + np.tanh(strength) * 10

## FlowField Class
class Flowfield:
    def __init__(self, objects={}):
        self.objects = objects

    def draw(self,
             x_points=np.linspace(-10, 10, 200),
             y_points=np.linspace(-10, 10, 200),
             colorscheme="rainbow",
             n_contour_lines=15,
             n_streamline_density=0.5,
             potential_streamline_bool=False
            ):

        ## Create plots
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=(LONG_NAME_DICT["xvel"], LONG_NAME_DICT["pressure"],
                                            LONG_NAME_DICT["potential"], LONG_NAME_DICT["streamfunction"]),
                            shared_xaxes=True,
                            shared_yaxes=True,
                            x_title='x',
                            y_title='y',
                            horizontal_spacing=0.08,
                            vertical_spacing=0.08
                           )
        if len(self.objects) == 0:  # Edge scenario
            return fig

        ## System variables
        X, Y    = np.meshgrid(x_points, y_points)
        X_r     = X.flatten()
        Y_r     = Y.flatten()
        points  = np.vstack((X_r, Y_r)).T

        ## Get initial variables that are written into
        x_vels              = np.zeros_like(X_r)
        y_vels              = np.zeros_like(X_r)
        potential           = np.zeros_like(X_r)
        streamfunction      = np.zeros_like(X_r)
        u_cumulative        = 0
        v_cumulative        = 0
        V2_infty            = 1 ## Default value in case no uniform flow objects

        ## Get plotting values
        for object in self.objects.values():
            x_vels              += object.get_x_velocity_at(points)
            y_vels              += object.get_y_velocity_at(points)
            potential           += object.get_potential_at(points)
            streamfunction      += object.get_streamfunction_at(points)
            if flow_element_type(object) == "Uniform":
                u_cumulative += object.u
                v_cumulative += object.v
                V2_infty      = u_cumulative**2 + v_cumulative**2   ## Gets overwritten
                if V2_infty  == 0: V2_infty = 1                     ## Edge exception in calculation of Cp

            V2      = x_vels**2 + y_vels**2            ## Gets overwritten
            # V       = np.sqrt(V2)                      ## Gets overwritten
            Cp      = 1 - V2/V2_infty                  ## Cp calculation

        #### ================ ####
        #### Plotting Routine ####
        #### ================ ####
        ## Velocity Magnitude
        min = np.nanpercentile(x_vels, 5)
        max = np.nanpercentile(x_vels, 95)
        fig.add_trace(go.Contour(name=LONG_NAME_DICT["xvel"],
                                 x=x_points, y=y_points, z=np.reshape(x_vels, X.shape),
                                 colorscale=colorscheme,
                                 contours=dict(start=min,
                                               end=max,
                                               size=(max - min) / n_contour_lines,
                                              ),
                                 contours_showlines=False,
                                 showscale=True,
                                 hovertemplate='x = %{x:.4f}'+
                                               '<br>y = %{y:.4f}'+
                                               '<br>u = %{z:.4e}'+
                                               '<extra></extra>', ## '<extra></extra>' removes the trace name from hover text
                                 colorbar=dict(title_text= LONG_NAME_DICT["xvel"] + '   [m s<sup>-1</sup>]',
                                               title_side= 'right',
                                               ticks     = "inside",
                                               len       = 0.45,                          ## vertical height of colorbar, expressed in a fraction of graph height, final height reduced by ypad
                                               tickwidth = 2,
                                               ticklen   = 10,
                                               x         = 1.02,
                                               y         = 0.77,
                                               ypad      = 0
                                              ),
                                ),
                      row=1, col=1
                     )

        #### Impose contour lines from the streamfunction field onto the x-velocity field
        #### It is an alternative option to strline.create_streamline()!
        # min = np.nanpercentile(streamfunction, 5)
        # max = np.nanpercentile(streamfunction, 95)
        # fig.add_trace(go.Contour(x=x_points, y=y_points, z=np.reshape(streamfunction, X.shape),
        #                          colorscale=[[0,'#000000'],[1,'#000000']],
        #                          contours=dict(start=min,
        #                                        end=max,
        #                                        size=(max - min) / n_contour_lines,
        #                                       ),
        #                          showscale=False,
        #                          contours_coloring='lines',
        #                          hoverinfo='skip'
        #                         ),
        #               row=1, col=1
        #              )

        ## Pressure Coefficient
        min = np.nanpercentile(Cp, 5)
        max = np.nanpercentile(Cp, 95)
        fig.add_trace(go.Contour(name=LONG_NAME_DICT["pressure"],
                                 x=x_points, y=y_points, z=np.reshape(Cp, X.shape),
                                 colorscale=colorscheme,
                                 contours=dict(start=min,
                                               end=max,
                                               size=(max - min) / n_contour_lines,
                                              ),
                                 contours_showlines=False,
                                 showscale=True,
                                 hovertemplate='x = %{x:.4f}'+
                                               '<br>y = %{y:.4f}'+
                                               '<br>Cp = %{z:.4e}'+
                                               '<extra></extra>',
                                 colorbar=dict(title_text= LONG_NAME_DICT["pressure"] + '   [-]',
                                               title_side= 'right',
                                               ticks     = "inside",
                                               len       = 0.45,                          ## vertical height of colorbar, expressed in a fraction of graph height, final height reduced by ypad
                                               tickwidth = 2,
                                               ticklen   = 10,
                                               x         = 1.17,
                                               y         = 0.77,
                                               ypad      = 0
                                              ),
                                ),
                      row=1, col=2
                     )

        ## Potential Function
        min = np.nanpercentile(potential, 5)
        max = np.nanpercentile(potential, 95)
        fig.add_trace(go.Contour(name=LONG_NAME_DICT['potential'],
                                 x=x_points, y=y_points, z=np.reshape(potential, X.shape),
                                 colorscale=colorscheme,
                                 contours=dict(start=min,
                                               end=max,
                                               size=(max - min) / n_contour_lines,
                                              ),
                                 contours_showlines=False,
                                 showscale=True,
                                 hovertemplate='x = %{x:.4f}'+
                                               '<br>y = %{y:.4f}'+
                                               '<br>phi = %{z:.4e}'+
                                               '<extra></extra>',
                                 colorbar=dict(title_text= LONG_NAME_DICT["potential"] + '   [m<sup>2</sup> s<sup>-1</sup>]',
                                               title_side= 'right',
                                               ticks     = "inside",
                                               len       = 0.45,                          ## vertical height of colorbar, expressed in a fraction of graph height, final height reduced by ypad
                                               tickwidth = 2,
                                               ticklen   = 10,
                                               x         = 1.02,
                                               y         = 0.23,
                                               ypad      = 0
                                              ),
                                ),
                      row=2, col=1
                     )

        ## Streamfunction
        min = np.nanpercentile(streamfunction, 5)
        max = np.nanpercentile(streamfunction, 95)
        fig.add_trace(go.Contour(name=LONG_NAME_DICT['streamfunction'],
                                 x=x_points, y=y_points, z=np.reshape(streamfunction, X.shape),
                                 colorscale=colorscheme,
                                 contours=dict(start=min,
                                               end=max,
                                               size=(max - min) / n_contour_lines,
                                              ),
                                 contours_showlines=False,
                                 showscale=True,
                                 hovertemplate='x = %{x:.4f}'+
                                               '<br>y = %{y:.4f}'+
                                               '<br>psi = %{z:.4e}'+
                                               '<extra></extra>',
                                 colorbar=dict(title_text= LONG_NAME_DICT["streamfunction"] + '   [m<sup>2</sup> s<sup>-1</sup>]',
                                               title_side= 'right',
                                               ticks     = "inside",
                                               len       = 0.45,                          ## vertical height of colorbar, expressed in a fraction of graph height, final height reduced by ypad
                                               tickwidth = 2,
                                               ticklen   = 10,
                                               x         = 1.17,
                                               y         = 0.23,
                                               ypad      = 0
                                              ),
                                 ),
                      row=2, col=2
                     )


        streamlines = strline.create_streamline(x_points, -y_points,                                  # for some reason, we need the x-axis reflection, so we need negative y
                                      np.reshape(x_vels, X.shape), -np.reshape(y_vels, Y.shape), # for some reason, we need the x-axis reflection, so we need negative y
                                      density=n_streamline_density,
                                      hoverinfo='skip',
                                      name='stream_lines',
                                      line=dict(color='rgba(0,0,0,1)',
                                                width=1)
                                     )
        if potential_streamline_bool:
            potentiallines = strline.create_streamline(x_points, y_points,
                                          np.reshape(y_vels, Y.shape), -(np.reshape(x_vels, X.shape)),
                                          density=n_streamline_density,
                                          arrow_scale=0.00001,
                                          hoverinfo='skip',
                                          name='potential_lines',
                                          line=dict(color='rgba(0,0,0,1)',
                                                    width=1)
                                         )
        # https://stackoverflow.com/questions/68187485/subplot-for-go-figure-objects-with-multiple-plots-within-them
        for t in streamlines.data:
            fig.append_trace(t, row=1, col=1)
            fig.append_trace(t, row=1, col=2)
            fig.append_trace(t, row=2, col=2)
        if potential_streamline_bool:
            for t in potentiallines.data:
                fig.append_trace(t, row=2, col=1)

        ## Plot flow element origins
        rows, cols = fig._get_subplot_rows_columns()    ## rows, cols are range, not int
        for row in rows:
            for col in cols:
                for i, object in enumerate(self.objects.values()):
                    ## All flow elements that are described by a point
                    try:
                        fig.add_trace(go.Scatter(name=f"{i + 1}. [{flow_element_type(object)}]",
                                                 x=[object.x], y=[object.y],
                                                 marker=dict(color=line_color(object),
                                                             size=dot_size(object)
                                                            ),
                                                 hovertemplate=f'<b>{i + 1}. [{flow_element_type(object)}]</b>'+
                                                               '<br>x = %{x:.4f}'+
                                                               '<br>y = %{y:.4f}'+
                                                               f'<br>strength = {object.strength:.4e}'+
                                                               '<br>%{text}'
                                                               '<extra></extra>',
                                                               text = [f'alpha = {object.alpha:.4e}' if flow_element_type(object) == "Doublet" else ''],
                                                ),
                                      row=row, col=col
                                     )
                    except AttributeError:
                        pass

                    ## All flow elements that are described by a line
                    try:
                        fig.add_trace(go.Line(name=f"{i + 1}. [{flow_element_type(object)}]",
                                              x=[object.x1, object.x2], y=[object.y1, object.y2],
                                              line=dict(color=line_color(object),
                                                        width=line_width(object)
                                                       ),
                                              hovertemplate=f'<b>{i + 1}. [{flow_element_type(object)}]</b>'+
                                                            '<br>x = %{x:.4f}'+
                                                            '<br>y = %{y:.4f}'+
                                                            f'<br>strength = {object.strength:.4e}'+
                                                            '<extra></extra>',
                                             ),
                                      row=row, col=col
                                     )
                    except AttributeError:
                        pass

        ## Update x-axis properties
        fig.update_xaxes(#title_text='x',
                         title_font_color='#000000',
                         title_standoff=0,
                         gridcolor='rgba(153, 153, 153, 0.75)', #999999 in RGB
                         gridwidth=1,
                         zerolinecolor='#000000',
                         zerolinewidth=2,
                         linecolor='#000000',
                         linewidth=1,
                         ticks='outside',
                         ticklen=10,
                         tickwidth=2,
                         tickcolor='#000000',
                         tickfont_color='#000000',
                         minor_showgrid=True,
                         minor_gridcolor='rgba(221, 221, 221, 0.50)', #DDDDDD in RGB, 0.50 opacity
                         minor_ticks='outside',
                         minor_ticklen=5,
                         minor_tickwidth=2,
                         minor_griddash='dot',
                         hoverformat='.4f',
                         range=[x_points.min(),x_points.max()],
                        )


        ## Update y-axis properties
        fig.update_yaxes(#title_text='y',
                         title_font_color='#000000',
                         title_standoff=0,
                         gridcolor='rgba(153, 153, 153, 0.75)', #999999 in RGB, 0.75 opacity
                         gridwidth=1,
                         zerolinecolor='#000000',
                         zerolinewidth=2,
                         linecolor='#000000',
                         linewidth=1,
                         ticks='outside',
                         ticklen=10,
                         tickwidth=2,
                         tickcolor='#000000',
                         tickfont_color='#000000',
                         minor_showgrid=True,
                         minor_gridcolor='rgba(221, 221, 221, 0.50)', #DDDDDD in RGB, 0.50 opacity
                         minor_ticks='outside',
                         minor_ticklen=5,
                         minor_tickwidth=2,
                         minor_griddash='dot',
                         range=[y_points.min(),y_points.max()],
                        )

        ## Update figure layout
        fig.update_layout(font_color='#000000',
                          plot_bgcolor='rgba(255,255,255,1)',
                          paper_bgcolor='rgba(255,255,255,1)',
                          width=900,
                          height=800,
                          showlegend=False,
                         )

        return fig
