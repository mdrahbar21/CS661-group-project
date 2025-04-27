import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, name='Summary', title='Application Summary') # Add title

layout = dbc.Container([
     dbc.Row(dbc.Col(html.H1("Application Summary", className="text-center my-4"), width=12)), # Main Title
     dbc.Row(dbc.Col([
         dbc.Card(dbc.CardBody([
            html.H4("Overview", className="card-title"),
            html.P("This tool leverages a dataset of historical international cricket matches (primarily focusing on ODI, T20I, and Test formats available in the dataset) to provide insights into individual player performance against specific opponents and compared to their overall career stats within a format."),
            html.Hr(),
            html.H4("Key Features Implemented", className="card-title"),
             html.Ul([
                html.Li("Selection of individual players from the dataset via a dropdown menu."),
                 html.Li("Choice of match format (ODI, T20, T20I, Test) to filter analysis."),
                html.Li("Interactive world map highlighting countries against which the selected player has played in the chosen format."),
                html.Li("Ability to click on a highlighted country on the map to trigger a detailed statistical analysis."),
                html.Li("Head-to-head statistics display for the player against the selected opponent."),
                html.Li("Comparison visualizations (gauges, bar charts) showing performance metrics against the opponent versus the player's overall career stats in that format."),
                html.Li("Visual analysis of dismissal types and run-scoring breakdown (4s, 6s, others) against the specific opponent."),
                html.Li("Option to download the filtered match data (player vs selected opponent) as a CSV file."),
                html.Li("Option to download a concise summary of the head-to-head performance as a PDF file."),
            ]),
            html.Hr(),
             html.H4("How to Use", className="card-title"),
             html.Ol([
                 html.Li("Navigate to the 'Player Analyzer' page using the sidebar."),
                 html.Li("Select a player from the dropdown list."),
                 html.Li("Choose the desired match format (ODI, T20, etc.)."),
                 html.Li("The world map will update, highlighting opponents played in blue."),
                 html.Li("Click on one of the blue countries on the map."),
                 html.Li("The statistics section below the map will populate with analysis and comparison data."),
                 html.Li("Use the download buttons at the bottom of the statistics section if needed.")
             ]),

        ]), className="shadow-sm mb-4")
     ], width=12, md=10, lg=8, className="mx-auto")) # Center card
 ])