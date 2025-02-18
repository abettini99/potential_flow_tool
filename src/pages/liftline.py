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

    html.H1("Lifting line theory"),

    dcc.Markdown(r'''
    A vortex filament of strength $\Gamma$ bound to a fixed location in flow will experience a lift force, $L\' = \rho_\infty V_\infty \Gamma$. 
    This bound vortex is fixed in space. We simulate this by replacing a finite wing of span **b** with a bound vortex from $y = -b/2$ to $y = b/2$. 
    Due to Helmholtz's Theorem, a bound vortex cannot end in the fluid, so we assume vortex filaments from each of the wingtips extend downstream to infinity. 
    The bound vortex plus the two trailing downstream form a horseshoe shape as shown below:
    ''',mathjax=True),

    html.Img(src="assets/horseshoe.png",
            style={
                    "height": "auto", 
                    "display": "block", 
                    "margin-left": "auto", 
                    "margin-right": "auto"}),

    dcc.Markdown('''
        $$
        \\mathbf{dV} = \\frac{\\Gamma}{4\\pi} \\frac{\\mathbf{dl} \\times \\mathbf{r}}{|\\mathbf{r}|^3}
        $$
    ''' , mathjax=True),
    dcc.Markdown(r'''
        Consider the downwash $w$ induced along the bound vortex from $y = -b/2$ to $y = b/2$ as shown in the figure below. 
        The bound vortex induces no velocity along itself but the two trailing vortices contribute to the velocity along the bound vortex in a downward direction. 
        Hence, the downwash $w(y)$ at any spanwise position $y$ can be expressed as:    
    ''' , mathjax=True),
    dcc.Markdown('''
        $$
        \\mathbf{dV} = \\frac{\\Gamma}{4\\pi} \\frac{\\mathbf{dl} \\times \\mathbf{r}}{|\\mathbf{r}|^3}
        $$

        $$
        w(y) = - \\frac{\\Gamma}{4 \\pi (b/2 + y)} + \\frac{\\Gamma}{4 \\pi (b/2 - y)}
        $$

        which simplifies to: 

        $$
        w(y) = - \\frac{\\Gamma}{4 \\pi} + \\frac{b}{4 \\pi ((b/2)^2 - y^2)}
        $$
    ''' , mathjax=True),
    html.Img(src="assets/horseshoe2.png",
            style={
                    "height": "auto", 
                    "display": "block", 
                    "margin-left": "auto", 
                    "margin-right": "auto"}),
        dcc.Markdown('''

        Note that as $y$ approaches $b/2$ or $-b/2$, $w$ goes to $-\infty$. To address this, the lifting line theory suggests superimposing multiple horseshoe vortices along the span, 
        creating a more realistic distribution of circulation. Each horseshoe vortex contributes a small circulation $d\Gamma$ and the net circulation along the lifting line is the sum 
        of these contributions. This approach is illustrated in Figure 5.14, where the lifting line is represented by a sequence of bound vortices, each trailing a pair of vortices downstream.\n

        The circulation at $y$ is $d\Gamma$, and its change over the y-axis is given by
    ''' , mathjax=True),
    dcc.Markdown('''

        $$
        d \\Gamma(y) = \\frac{d \\Gamma}{dy}  dy
        $$

        Using the Biot-Savart law, the induced velocity $dw$ at $y_0$ due to a segment $dy$ of a trailing vortex at location $y$ is given by:

        $$
        dw = - \\frac{(d\\Gamma / dy)\\, dy}{4 \\pi (y_0 - y)}
        $$

        Noting that $(d\\Gamma / dy)$ is a negative value, the negative sign makes $dw$ positive downwards.\n\n

        Hence, integrating $dw$ over the wingspan yields

        $$
        w(y_0) = - \\frac{1}{4 \\pi} \\int_{-b/2}^{b/2} \\frac{(d\\Gamma / dy)\\, dy}{y_0 - y}
        $$

    ''',mathjax=True),
    html.Img(src="assets/dgamma.png",
            style={
                    "height": "auto", 
                    "display": "block", 
                    "margin-left": "auto", 
                    "margin-right": "auto"}),

    html.H2("Lifting Line tool"),

    dcc.Markdown(r'''
    The tool below allows you to visualise how a lifting line can simulate the lift distribution of a wing. Using vortex filaments and the Biot-Savart law, 
    a representative wing can be drawn out if the strength distribution on the wing is known. This tool allows you to visualise the velocity field around a lifting line model. \m
                 
    As you know from the lectures, The trailing vortices around the lifting line induce a downwash along the wing. You can try to change the strength distribution of the vortices,
    the freestream velocity and the discretisation of the wing to see how the downwash changes. Observe the velocity right brfore it reaches the wing. Do you observe an increased angle of attack?
    
    ''',mathjax=True),
    html.H3("How to use the lifting line tool:"),
    dcc.Markdown(r'''
    1. **Discretisation**: Select the number of vortices to represent the wing with the slider, and the mesh type with the dropdown menu. The more vortices, the more accurate the simulation. Having a discretisation of 1 means it is a horseshoe vortex.
    2. **Strength Slider**: Adjust the strength of the vortices. This will change the circulation around the wing. The distribution of the strength can also be changed with the dropdown menu.
    3. **Freestream Velocity Slider**: Adjust the freestream velocity. This will change the downwash along the wing.
    4.  **Draw!**: Click the draw button to see the velocity field around the wing. This may take around 10-20 seconds depending on how fine your mesh is. Be patient and only press it once please.
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