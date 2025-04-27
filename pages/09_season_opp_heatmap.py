import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Register this page with Dash Pages ---
dash.register_page(__name__, name='Season Heatmap',
                   path='/heatmap', title='Season × Opponent Heatmap')

# --- Constants ---
ALLOWED_FORMATS = ["ODI", "T20", "T20I", "Test"]
METRIC_OPTIONS = [
    {'label': 'Batting Average', 'value': 'bat_avg'},
    {'label': 'Strike Rate',    'value': 'strike_rate'},
    {'label': 'Total Runs',     'value': 'total_runs'}
]

# --- Helper to deserialize the stored DataFrame ---


def deserialize_data(store_data):
    if store_data is None:
        return pd.DataFrame()
    df = pd.read_json(store_data, orient='split')
    # Ensure 'start_date' is datetime
    if 'start_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['start_date']):
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    return df


# --- Layout ---
layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Performance Heatmap: Season × Opponent"),
            width=12, className="my-4")),

    dbc.Row([
        dbc.Col([
            html.Label("Select Player:", className="fw-bold"),
            dcc.Dropdown(
                id='heatmap-player-dropdown',
                options=[{'label': 'Loading players...', 'value': ''}],
                value=None,
                clearable=False
            )
        ], width=6),
        dbc.Col([
            html.Label("Select Format:", className="fw-bold"),
            dbc.RadioItems(
                id='heatmap-format-radio',
                options=[{'label': f, 'value': f} for f in ALLOWED_FORMATS],
                value=ALLOWED_FORMATS[0],
                inline=True,
                inputClassName="me-1"
            ),
            dbc.Button("Submit", id='submit-button', n_clicks=0,
                       color='primary', className="mt-2"),
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Metric:", className="fw-bold"),
            dcc.Dropdown(
                id='heatmap-metric-dropdown',
                options=METRIC_OPTIONS,
                value='bat_avg',
                clearable=False
            )
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id='heatmap-graph'), type='circle'), width=12)
    ]),

    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id='heatmap-boxplot'),
                type='circle'), width=12)
    ], className="mt-4")
], fluid=True)

# --- Populate Player Dropdown ---


@callback(
    Output('heatmap-player-dropdown', 'options'),
    Output('heatmap-player-dropdown', 'value'),
    Input('main-data-store', 'data')
)
def update_player_list(store_data):
    df = deserialize_data(store_data)
    if df.empty or 'name' not in df.columns:
        return [{'label': 'No players found', 'value': ''}], ''
    players = sorted(df['name'].dropna().unique())
    opts = [{'label': p, 'value': p} for p in players]
    return opts, opts[0]['value']

# --- Generate Heatmap ---


@callback(
    Output('heatmap-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    State('heatmap-player-dropdown', 'value'),
    State('heatmap-format-radio', 'value'),
    State('heatmap-metric-dropdown', 'value'),
    State('main-data-store', 'data')
)
def update_heatmap(n_clicks, player, fmt, metric, store_data):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    df = deserialize_data(store_data)
    if df.empty or not player or not fmt:
        return go.Figure(layout=go.Layout(
            title="Select a player and format to view the heatmap", title_x=0.5
        ))
    # Filter and extract seasons
    d = df[(df['name'] == player) & (df['match_type'] == fmt)
           & df['start_date'].notna()].copy()
    d['season'] = d['start_date'].dt.year

    # Compute chosen metric per season × opponent
    if metric == 'bat_avg':
        agg = d.groupby(['season', 'opposition_team']).apply(
            lambda g: g['runs_scored'].sum(
            ) / g['player_out'].sum() if g['player_out'].sum() > 0 else np.nan
        ).reset_index(name='value')
        color_label = 'Batting Average'
    elif metric == 'strike_rate':
        agg = d.groupby(['season', 'opposition_team']).apply(
            lambda g: (g['runs_scored'].sum() / g['balls_faced'].sum()
                       * 100) if g['balls_faced'].sum() > 0 else np.nan
        ).reset_index(name='value')
        color_label = 'Strike Rate'
    else:  # total_runs
        agg = d.groupby(['season', 'opposition_team'])[
            'runs_scored'].sum().reset_index(name='value')
        color_label = 'Total Runs'

    # Pivot to matrix
    heat = agg.pivot(index='opposition_team', columns='season', values='value')

    fig = px.imshow(
        heat,
        labels={'x': 'Season', 'y': 'Opponent', 'color': color_label},
        text_auto='.2f',
        aspect='auto'
    )
    fig.update_layout(
        title=f"{player} — {color_label} by Season & Opponent",
        xaxis={'side': 'top'},
        margin={'t': 50, 'b': 50, 'l': 150, 'r': 50}
    )
    return fig

# --- Drill-Down Boxplot on Click ---


@callback(
    Output('heatmap-boxplot', 'figure'),
    Input('heatmap-graph', 'clickData'),
    State('heatmap-player-dropdown', 'value'),
    State('heatmap-format-radio', 'value'),
    State('heatmap-metric-dropdown', 'value'),
    State('main-data-store', 'data')
)
def update_boxplot(clickData, player, fmt, metric, store_data):
    # Default prompt
    if not clickData:
        return go.Figure(layout=go.Layout(
            title="Click a cell in the heatmap to see per-match distribution", title_x=0.5
        ))
    season = clickData['points'][0]['x']
    opponent = clickData['points'][0]['y']

    df = deserialize_data(store_data)
    d = df[
        (df['name'] == player) &
        (df['match_type'] == fmt) &
        (df['start_date'].dt.year == int(season)) &
        (df['opposition_team'] == opponent)
    ].copy()
    if d.empty:
        return go.Figure(layout=go.Layout(
            title=f"No data for {player} vs {opponent} in {season}", title_x=0.5
        ))

    # Compute per-match metric
    if metric == 'bat_avg':
        d['metric'] = d.apply(lambda r: r['runs_scored']/r['player_out']
                              if r['player_out'] > 0 else np.nan, axis=1)
        ylabel = 'Batting Average per Match'
    elif metric == 'strike_rate':
        d['metric'] = d.apply(lambda r: (
            r['runs_scored']/r['balls_faced']*100) if r['balls_faced'] > 0 else np.nan, axis=1)
        ylabel = 'Strike Rate per Match'
    else:
        d['metric'] = d['runs_scored']
        ylabel = 'Runs Scored per Match'

    fig = px.box(
        d,
        y='metric',
        points='all',
        labels={'metric': ylabel},
        title=f"{player} {ylabel} vs {opponent} in {season}"
    )
    fig.update_layout(margin={'t': 50, 'b': 30, 'l': 50, 'r': 30})
    return fig
