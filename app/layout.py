# layout.py
from dash import html, dcc
import dash_bootstrap_components as dbc

# Sidebar navigation for pipeline stages
sidebar = dbc.Nav(
    [
        dbc.NavLink("Preprocessing", href="/preprocessing", active="exact"),
        dbc.NavLink("Random Forest", href="/random-forest", active="exact"),
        dbc.NavLink("Kalman Filter", href="/kalman-filter", active="exact"),
        dbc.NavLink("Simpson’s Rule", href="/simpsons-rule", active="exact"),
        dbc.NavLink("Q-Learning", href="/q-learning", active="exact"),
        dbc.NavLink("Shewhart Control", href="/shewhart-control", active="exact"),
        dbc.NavLink("Visualization", href="/visualization", active="exact"),
    ],
    vertical=True,
    pills=True,
    className="bg-light",
)

# Main content area
content = html.Div(id="page-content", style={"margin-left": "18rem", "margin-right": "2rem", "padding": "2rem 1rem"})

# App layout
app_layout = html.Div([
    dcc.Location(id="url"),
    dbc.Row([
        dbc.Col(sidebar, width=2, style={"position": "fixed", "height": "100vh", "overflow": "auto"}),
        dbc.Col(content, width={"size": 10, "offset": 2}),
    ])
])