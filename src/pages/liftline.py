import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import numpy as np

def calculate_liftline(discretisation, distribution, V_inf, mesh = 'Uniform', b=2, gamma0=1):
    if mesh == 'Uniform':
        y = np.linspace(-b/2, b/2, discretisation)  
    elif mesh == 'Cosine':
        y = b/2 * (1 - np.cos(np.linspace(0, np.pi, discretisation)))-b/2

    mid_y = (y[1:] + y[:-1]) / 2

    if distribution == 'Elliptic':
        Gamma = gamma0 * np.sqrt(1 - (2 * mid_y / b)**2)
    elif distribution == 'Cosine':
        Gamma = gamma0 * np.cos(np.pi * mid_y / b)
    else:
        raise ValueError('Invalid distribution type')
    
    rho_inf = 1.225


    Gamma_diff = Gamma[1:] - Gamma[:-1]
    Gamma_diff = np.concatenate(([Gamma[0]], Gamma_diff, [-Gamma[-1]]))
    print(Gamma_diff)

    lift_dist = rho_inf * V_inf * Gamma

    Downwash = np.zeros(discretisation)

    for i in range(len(y)):
        downwash = 0
        query = y[i]
        for j in range(len(y)):
            # print("i: ", i, ", j: ", j)
            vort = y[j]
            # print(vort-query)
            if query==vort:
                continue
            else:
                downwash -= Gamma_diff[j] / (4 * np.pi) / (vort-query)
                # print(Gamma_diff[j] / (4 * np.pi) / (vort-query))
            

        Downwash[i] = downwash
    result_dict = {
        'y': y,
        'mid_y': mid_y,
        'Gamma': Gamma,
        'Gamma_diff': Gamma_diff,
        'lift_dist': lift_dist,
        'Downwash': Downwash
    }
    return result_dict

def draw_liftline(Params, x, fig):
    y = Params['y']
    mid_y = Params['mid_y']
    Gamma = Params['Gamma']
    Gamma_diff = Params['Gamma_diff']
    lift_dist = Params['lift_dist']
    Downwash = -Params['Downwash']

    x_arr = np.ones(len(y))*x

    mid_x = np.ones(len(mid_y))*x

    fig.add_trace(go.Scatter3d(x=y, y=x_arr, z=Downwash, mode='lines', name='Downwash', line=dict(width=5, color='red'), legendgroup='Downwash'))
    fig.add_trace(go.Scatter3d(x=mid_y, y=mid_x, z=Gamma, mode='lines', name='Circulation', line=dict(width=5, color='orange'), legendgroup='Gamma'))

    for i in range(len(y)):
        downwash_arrow = go.Cone(x=[y[i]], y=[x], z=[Downwash[i]-0.05*np.sign(Downwash[i])], 
                                 u=[0], v=[0], w=[0.1*np.sign(Downwash[i])], 
                                 showscale=False, sizemode='absolute', sizeref=0.1, legendgroup='Downwash', showlegend=False, colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']])

        fig.add_trace(go.Scatter3d(x=[y[i], y[i]], y=[x, x], z=[0, Downwash[i]], mode='lines', line=dict(width=5, color='red'), legendgroup='Downwash', showlegend=False))
        fig.add_trace(downwash_arrow)

    for i in range(len(mid_y)):
        Gamma_arrow = go.Cone(x=[mid_y[i]], y=[x], z=[Gamma[i]-0.05*np.sign(Gamma[i])], 
                              u=[0], v=[0], w=[0.1*np.sign(Gamma[i])], 
                              showscale=False, sizemode='absolute', sizeref=0.1, legendgroup='Gamma', showlegend=False, colorscale=[[0, 'rgb(255,165,0)'], [1, 'rgb(255,165,0)']])
        fig.add_trace(go.Scatter3d(x=[mid_y[i], mid_y[i]], y=[x, x], z=[0, Gamma[i]], mode='lines', line=dict(width=5, color='orange'), legendgroup='Gamma', showlegend=False))
        fig.add_trace(Gamma_arrow)
    return fig



def liftline():
    return html.Div(children=[
    
    #### ============== ####
    #### BUILDING BLOCK ####
    #### ============== ####
    html.H1("Lifting line theory"),

    dcc.Markdown('''
    \\[...\\] \n
                 
    The velocity around a source/sink is then given by
                 
    $$
    V_r = \\frac{\\Lambda}{2\\pi r}\\,,
    $$
    $$
    V_\\theta = 0\\,.
    $$

    Which as a set of cartesian velocities can be found by      
    EXAMPLE TEXT: \n
    bla bla bla
    ''',mathjax=True),

    html.H2("Lifting Line tool"),
    dcc.Markdown('''
                 ''',mathjax=True),

    html.Br(),

    # Hidden store for Filam object

    html.Label('Discretisation:'),
    dcc.Slider(1, 7, 1, 
               value=1,
               id='num-lines',
              ),
    html.Label('Strength Slider:'),
    dcc.Slider(0.1, 4,
               value=1,
               id='VortexStrength_liftline',
              ),
    html.Label('Freestream Velocity Slider:'),
    dcc.Slider(0.1, 2,
               value=1,
               id='VelInfMag_liftline',
              ),
    html.Div(
        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '10px'},
        children=[
            html.Label('Mesh Type:', style={'marginRight': '5px'}),
            dcc.Dropdown(['Uniform', 'Cosine'], 'Uniform', id='mesh-type-liftline', style={'width': '150px'}),
            html.Label('Distribution Type:', style={'marginLeft': '20px', 'marginRight': '5px'}),
            dcc.Dropdown(['Elliptic', 'Cosine'], 'Elliptic', id='distribution-type-liftline', style={'width': '150px'})
        ]
    ),
    html.Button('Draw', id='draw-button-lift', n_clicks=0),

    ## Graph updated via app.callable() in main.py
    dcc.Graph(id='liftline', config={'clickmode': 'event+select'}),  # Enable clickmode to select points
    

    ])

if __name__ == '__main__':
    result = calculate_liftline(100, 1, 1, mesh='cosine')
    plt.plot(result['mid_y'], result['Gamma'])
    plt.plot(result['y'], result['Downwash'])
    plt.show()