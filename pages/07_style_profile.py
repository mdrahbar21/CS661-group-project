# pages/07_recent_form.py

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import functools

# ─── Register page ─────────────────────────────────────────────────────────────
dash.register_page(
    __name__,
    path='/recent-form',
    name='Recent Form',
    title='Recent Form Comparison'
)

# ─── Cache parsing of the JSON store ───────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _cached_df(store_json: str):
    """Parse store JSON once and cache the DataFrame."""
    df = pd.read_json(store_json, orient='split')
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    return df


def _load_df(store_data):
    """Return cached DataFrame or None."""
    if not store_data:
        return None
    try:
        return _cached_df(store_data)
    except Exception:
        return None


def layout():
    return dbc.Container(fluid=True, children=[

        dbc.Row(dbc.Col(html.H2("Recent Form Comparison",
                className="text-center my-4"), width=12)),

        # Controls: Player1, Player2, Format, Last N Innings
        dbc.Row([
            dbc.Col([
                html.Label("Player 1:", className="fw-bold"),
                dcc.Dropdown(id='form-player-1',
                             placeholder="Select player 1"),
            ], width=3),
            dbc.Col([
                html.Label("Player 2:", className="fw-bold"),
                dcc.Dropdown(id='form-player-2',
                             placeholder="Select player 2"),
            ], width=3),
            dbc.Col([
                html.Label("Format:", className="fw-bold"),
                dbc.RadioItems(
                    id='form-format',
                    options=[{'label': f, 'value': f}
                             for f in ["ODI", "T20", "T20I", "Test"]],
                    value="ODI", inline=True
                )
            ], width=3),
            dbc.Col([
                html.Label("Last N Innings:", className="fw-bold"),
                dbc.Input(id='form-n', type='number',
                          min=1, max=50, step=1, value=5)
            ], width=3),
            # add a button to submit the form
            dbc.Col([
                dbc.Button("Submit", id='form-submit', n_clicks=0,
                           color='primary', className="mt-2")
            ], width=3),
        ], className="mb-4"),

        # Shared line chart
        dbc.Row(dbc.Col(dcc.Loading(dcc.Graph(id='form-line')),
                width=12), className="mb-4"),

        # Gauges
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id='form-gauge-1')), width=6),
            dbc.Col(dcc.Loading(dcc.Graph(id='form-gauge-2')), width=6),
        ], className="mb-4"),

        # Pie charts
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id='form-pie-1')), width=6),
            dbc.Col(dcc.Loading(dcc.Graph(id='form-pie-2')), width=6),
        ], className="mb-4"),

        html.P(
            "Line: runs in each of the last N innings.  "
            "Gauges: form average vs career average.  "
            "Pies: dismissal breakdown in those innings.",
            className="text-center text-muted mt-4"
        )
    ])


@callback(
    Output('form-player-1', 'options'),
    Output('form-player-2', 'options'),
    Output('form-player-1', 'value'),
    Output('form-player-2', 'value'),
    Input('main-data-store', 'data'),
)
def _fill_players(store_data):
    df = _load_df(store_data)
    if df is None or 'name' not in df.columns:
        return [], [], None, None
    players = sorted(df['name'].dropna().unique())
    opts = [{'label': p, 'value': p} for p in players]
    default1 = opts[0]['value'] if opts else None
    default2 = opts[1]['value'] if len(opts) > 1 else default1
    return opts, opts, default1, default2


@callback(
    Output('form-line',    'figure'),
    Output('form-gauge-1', 'figure'),
    Output('form-gauge-2', 'figure'),
    Output('form-pie-1',   'figure'),
    Output('form-pie-2',   'figure'),
    Input('form-submit', 'n_clicks'),
    State('form-player-1', 'value'),
    State('form-player-2', 'value'),
    State('form-format',  'value'),
    State('form-n',       'value'),
    State('main-data-store', 'data'),
    prevent_initial_call=True
)
def update_form(n_clicks, p1, p2, fmt, n, store_data):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    df = _load_df(store_data)
    # placeholder if inputs missing
    empty = go.Figure()
    empty.update_layout(title="Select both players, format & N")
    if df is None or not all([p1, p2, fmt, isinstance(n, int) and n > 0]):
        return empty, empty, empty, empty, empty

    # shared filter by format
    df_fmt = df[df['match_type'] == fmt]

    def get_player_data(player):
        d = df_fmt[df_fmt['name'] == player]
        if d.empty:
            return None, None, None, None
        # last n innings by date
        d_recent = d.nlargest(n, 'start_date').sort_values('start_date')
        # line data
        line_df = d_recent[['start_date', 'runs_scored']].assign(Player=player)
        # career average
        total_outs = d['player_out'].sum()
        career_avg = d['runs_scored'].sum(
        ) / total_outs if total_outs > 0 else 0
        # form average
        form_outs = d_recent['player_out'].sum()
        form_avg = d_recent['runs_scored'].sum(
        ) / form_outs if form_outs > 0 else 0
        # dismissal pie
        outs = d_recent[d_recent['player_out'] == 1]
        if not outs.empty:
            pie_df = outs['out_kind'].fillna(
                'Unknown').value_counts().reset_index()
            pie_df.columns = ['Dismissal', 'Count']
        else:
            pie_df = pd.DataFrame({'Dismissal': ['None'], 'Count': [1]})
        return line_df, career_avg, form_avg, pie_df

    l1, car1, f1, pie1 = get_player_data(p1)
    l2, car2, f2, pie2 = get_player_data(p2)
    if l1 is None or l2 is None:
        return empty, empty, empty, empty, empty

    # Combined line chart
    line_all = pd.concat([l1, l2], ignore_index=True)
    fig_line = px.line(
        line_all, x='start_date', y='runs_scored', color='Player',
        markers=True, title=f"Last {n} Innings Runs: {p1} vs {p2}",
        labels={'start_date': 'Date', 'runs_scored': 'Runs'}
    ).update_layout(margin=dict(t=40, b=20))

    # Gauges
    def make_gauge(player, form_avg, career_avg):
        return go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=form_avg,
            delta={'reference': career_avg,
                   'valueformat': '.1f', 'position': 'top'},
            title={'text': f"{player} Form Avg vs Career"},
            gauge={'axis': {'range': [0, max(form_avg, career_avg)*1.5]},
                   'bar': {'color': 'green'}}
        )).update_layout(margin=dict(t=40, b=20))

    fig_g1 = make_gauge(p1, f1, car1)
    fig_g2 = make_gauge(p2, f2, car2)

    # Pie charts
    def make_pie(pie_df, player):
        fig = px.pie(
            pie_df, names='Dismissal', values='Count',
            title=f"{player} Dismissals", hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig.update_layout(margin=dict(t=40, b=20))

    fig_p1 = make_pie(pie1, p1)
    fig_p2 = make_pie(pie2, p2)

    return fig_line, fig_g1, fig_g2, fig_p1, fig_p2
