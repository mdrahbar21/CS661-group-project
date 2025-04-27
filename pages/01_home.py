import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', title='Cricket Analyzer Home') # Add Title

layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Welcome to the Cricket Player Performance Analyzer", className="text-center my-4"), width=12)), # H1 for main title
    dbc.Row(dbc.Col([
        dbc.Card(dbc.CardBody([
            html.P("This interactive application allows you to explore and analyze international cricket player statistics based on historical match data.", className="lead"), # Lead paragraph
            html.P("Use the navigation panel on the left to access the different analysis tools:"),
            html.Ul([
                html.Li(dcc.Link(html.B("Summary:"), href="/summary", className="text-decoration-none"), className="mb-2"), # Use Link and style
                html.Li(dcc.Link(html.B("Player Analyzer:"), href="/analyzer", className="text-decoration-none"), className="mb-2"), # Use Link and style
            ]),
            html.P("Start by navigating to the Player Analyzer to select players, formats, and opponents."),
        ]), className="shadow-sm") # Add subtle shadow
    ], width=12, md=10, lg=8, className="mx-auto")), # Center the card content
    html.Hr(className="my-5"),
     dbc.Row(dbc.Col([
         html.Small("Data Source: Cricsheet.org (Note: Data accuracy depends on the source dataset)", className="text-muted text-center d-block")
     ]))

], fluid=True)