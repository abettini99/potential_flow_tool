import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


def downwash():
    return html.Div(children=[
        html.H1("Introduction to downwash and induced drag"),
        dcc.Markdown('''
            A true wing is different from an airfoil (an infinite wing) as it has a finite length. Airfoils give a 2-D representation of the flow $(c_l, c_d)$, and these are only functions of the **chord** and **thickness**. However, 3-D wings introduce a new variable, the finite **span** of the wing. Hence, the analyses of this chapter will present the variation of forces of airfoil sections along the span of the wing, leading to the $C_L$ and $C_D$ of the wing. 
            \n\n
            As the pressure under the wing is higher than that above it, flow at the tip "leaks" from the bottom to the top in a curling motion. This circular flow pattern trails downstream of the wing, creating a small downward component of air velocity near the wing. This is called **downwash**, $w$. The downwash modifies the flow by:

            1. Reducing the angle of attack seen by an induced angle of attack $\\alpha_i$, resulting in an effective angle of attack $\\alpha_{eff}$. The downwash combines with the freestream velocity, $V_{\\infty}$, to produce a local relative wind canted downwards by $\\alpha_i$ for each airfoil section.
            
                $$
                \\alpha_i \\approx - \\frac{w}{V_\\infty}
                $$
                $$
                \\alpha_{eff} = \\alpha - \\alpha_i
                $$

            2. Tilting the sectional lift vector, $L'$, back by $\\alpha_i$, creating a sectional induced drag, $D'_i$. This corresponds to the lift vector being always perpendicular to the local relative wind.

                $$
                D'_i \\approx L' \\alpha_i 
                $$

            In this chapter, the sectional lift, drag, and moment will be denoted with an apostrophe: $L', D', M'$. The sectional coefficients are in lowercase: $c_l, c_d, c_m$. The 3-D forces are in uppercase: $L, D, M$, and the corresponding coefficients are also in uppercase: $C_L, C_D, C_M$. Hence, $L$ can be obtained by integrating $L'$ over the span of the wing.

            $$
            L = \\int_{-\\frac{b}{2}}^{\\frac{b}{2}} L'(y) \\, dy
            $$

            $$
            L' = c_l q_{\\infty} c = \\rho_{\\infty} V_{\\infty} \\Gamma(y)
            $$

            The following parameters vary along the wingspan: chord, $c$; aerodynamic twist, $\\alpha$ and $\\alpha_i$. This leads to the variation of the following over the wingspan: $c_l$, $L'$, and $\\Gamma$.

            $$
            c_l (\\alpha_{eff}) = a_0 [\\alpha - \\alpha_i - \\alpha_{L=0}]
            $$

            where $a_0$ is the lift curve slope. Moreover, note that $L' = 0$ at the tips due to pressure equalization.
            ''', mathjax=True)])