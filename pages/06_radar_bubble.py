# ---------- pages/06_radar_bubble.py ----------

import dash
from dash import dcc, html, Input, Output, State, callback, register_page
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import traceback

# ---- page registration ----------------------------------------------------
register_page(__name__,
              name='Player Radar & Timeline',
              path='/radar-bubble',
              title='Player Radar & Timeline')

# ---- helpers --------------------------------------------------------------


def deserialize_data_radar(stored):
    if stored is None:
        print("Radar-Bubble: no store data")
        return None
    try:
        df = pd.read_json(stored, orient="split")
        if 'start_date' not in df.columns:
            return None
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        return df
    except Exception as e:
        traceback.print_exc()
        return None


def calculate_player_stats_aggregate(df_pl_fmt):
    if df_pl_fmt is None or df_pl_fmt.empty:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runs = int(df_pl_fmt['runs_scored'].sum())
        wkts = int(df_pl_fmt['wickets_taken'].sum())
        balls_faced = int(df_pl_fmt['balls_faced'].sum())
        outs = int(df_pl_fmt['player_out'].sum())
        balls_bowled = int(df_pl_fmt['balls_bowled'].sum())
        runs_conceded = int(df_pl_fmt['runs_conceded'].sum())
        return {
            'total_runs': runs,
            'total_wickets': wkts,
            'batting_avg': runs/outs if outs else 0,
            'batting_sr': runs/balls_faced*100 if balls_faced else 0,
            'bowling_avg': runs_conceded/wkts if wkts else np.inf,
            'bowling_econ': runs_conceded/balls_bowled*6 if balls_bowled else np.inf,
        }


def calculate_format_max_stats(df_fmt):
    if df_fmt is None or df_fmt.empty or 'player_id' not in df_fmt.columns:
        return {}
    g = df_fmt.groupby('player_id').agg(
        total_runs=('runs_scored', 'sum'),
        total_wickets=('wickets_taken', 'sum'),
        balls_faced=('balls_faced', 'sum'),
        outs=('player_out', 'sum'),
        balls_bowled=('balls_bowled', 'sum'),
        runs_conceded=('runs_conceded', 'sum')
    ).reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g['batting_avg'] = g['total_runs']/g['outs'].replace(0, np.nan)
        g['batting_sr'] = g['total_runs'] / \
            g['balls_faced'].replace(0, np.nan)*100
        g['bowling_avg'] = g['runs_conceded'] / \
            g['total_wickets'].replace(0, np.nan)
        g['bowling_econ'] = g['runs_conceded'] / \
            g['balls_bowled'].replace(0, np.nan)*6
    max_stats = {
        'total_runs': g['total_runs'].max(),
        'batting_avg': g['batting_avg'].max(),
        'batting_sr': g['batting_sr'].max(),
        'total_wickets': g['total_wickets'].max(),
        'bowling_avg': g['bowling_avg'].dropna().max() or 100,
        'bowling_econ': g['bowling_econ'].dropna().max() or 20,
    }
    max_stats = {k: (v if (pd.notna(v) and np.isfinite(v) and v > 0) else 1)
                 for k, v in max_stats.items()}
    return max_stats


def normalize_stat(v, vmax, lower_is_better=False):
    if pd.isna(v) or not np.isfinite(v) or vmax <= 0:
        return 0
    val = (v/vmax)*100
    if lower_is_better:
        val = 100-val
    return max(0, min(val, 100))


# ---- placeholder ----------------------------------------------------------
radar_placeholder = dbc.Alert(
    "Select a player and format, then click **Submit**.",
    color="info", className="text-center mt-4",
    id="radar-placeholder-message"
)

# ---- layout ---------------------------------------------------------------


def layout():
    ALLOWED_FORMATS = ["ODI", "T20", "T20I", "Test", "All"]
    init_opts = [{'label': 'Loading...', 'value': '', 'disabled': True}]
    return dbc.Container([
        dbc.Row(dbc.Col(
            html.H2("Player Radar, Timeline & Boundary Analysis"), className="mb-4")),
        # selection panel
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Selections"),
                dbc.CardBody([
                    html.Label("Player:", className="fw-bold"),
                    dcc.Dropdown(id='radar-player-dropdown', options=init_opts,
                                 placeholder="Select player...", clearable=False, className="mb-3"),
                    html.Label("Format:", className="fw-bold"),
                    dbc.RadioItems(id='radar-format-radio',
                                   options=[{'label': f, 'value': f}
                                            for f in ALLOWED_FORMATS],
                                   value="ODI", inline=True, inputClassName="me-1"),
                    dbc.Button("Submit", id='radar-submit-button',
                               n_clicks=0, color="primary", className="mt-3 w-100")
                ])
            ]), md=4, className="mb-3"),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Performance Radar (0-100)"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='radar-plot', style={'height': '380px'},
                                                   config={'displayModeBar': False})))
            ]), md=8)
        ], className="mb-4"),
        # time series row
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Runs per Match"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='bubble-runs-plot',
                                                   style={'height': '320px'},
                                                   config={'displayModeBar': False})))
            ]), md=6, className="mb-3"),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Wickets per Match"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='bubble-wickets-plot',
                                                   style={'height': '320px'},
                                                   config={'displayModeBar': False})))
            ]), md=6)
        ], className="mb-4"),
        # boundary scatter
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(
                    "Boundary % vs Balls Faced vs Time  (colour = Strike-rate, size = Balls)"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='boundary-scatter-plot',
                                                   style={'height': '350px'},
                                                   config={'displayModeBar': False})))
            ]), width=12)
        ], className="mb-4"),
        # yearly & histogram
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Boundary % by Year"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='boundary-year-line',
                                                   style={'height': '320px'},
                                                   config={'displayModeBar': False})))
            ]), md=6, className="mb-3"),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Boundary % Distribution"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='boundary-histogram',
                                                   style={'height': '320px'},
                                                   config={'displayModeBar': False})))
            ]), md=6)
        ], className="mb-4"),
        html.Div(id='radar-plots-output-area', children=radar_placeholder)
    ], fluid=True)

# ---- callbacks ------------------------------------------------------------


@callback(
    Output('radar-player-dropdown', 'options'),
    Output('radar-player-dropdown', 'value'),
    Input('main-data-store', 'data'),
    prevent_initial_call=False)
def populate_players(store):
    df = deserialize_data_radar(store)
    if df is None or df.empty or 'name' not in df.columns:
        return [{'label': 'No data', 'value': '', 'disabled': True}], None
    opts = [{'label': p, 'value': p}
            for p in sorted(df['name'].dropna().unique())]
    return opts, (opts[0]['value'] if opts else None)


@callback(
    Output('radar-plot', 'figure'),
    Output('bubble-runs-plot', 'figure'),
    Output('bubble-wickets-plot', 'figure'),
    Output('boundary-scatter-plot', 'figure'),
    Output('boundary-year-line', 'figure'),
    Output('boundary-histogram', 'figure'),
    Output('radar-plots-output-area', 'children'),
    Input('radar-submit-button', 'n_clicks'),
    State('radar-player-dropdown', 'value'),
    State('radar-format-radio', 'value'),
    State('main-data-store', 'data'),
    prevent_initial_call=True)
def make_all_plots(n_clicks, player, fmt, store):
    if not n_clicks:
        raise PreventUpdate
    placeholder = ""

    def empty_fig(title): return go.Figure(layout=go.Layout(
        title=title, title_x=0.5, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'))

    df = deserialize_data_radar(store)
    if df is None or df.empty or not player or not fmt:
        return *(empty_fig("No data"),)*6, radar_placeholder

    # allow "All" = no format filter
    if fmt == "All":
        player_df = df[df["name"] == player].copy()
        fmt_df = df.copy()
        fmt_label = "All Formats"
    else:
        player_df = df[(df["name"] == player) & (
            df["match_type"] == fmt)].copy()
        fmt_df = df[df["match_type"] == fmt].copy()
        fmt_label = fmt

    if player_df.empty:
        msg = f"No {fmt_label} data for {player}"
        return *(empty_fig(msg),)*6, ""

    # ---------------- radar -----------------
    radar_max = calculate_format_max_stats(fmt_df)
    radar_stat = calculate_player_stats_aggregate(player_df)

    cat = ['Total Runs', 'Batting Avg', 'Batting SR',
           'Total Wickets', 'Bowling Avg', 'Bowling Econ']
    raw = [radar_stat.get('total_runs', 0),
           radar_stat.get('batting_avg', 0),
           radar_stat.get('batting_sr', 0),
           radar_stat.get('total_wickets', 0),
           radar_stat.get('bowling_avg', np.inf),
           radar_stat.get('bowling_econ', np.inf)]
    norm = [normalize_stat(raw[0], radar_max.get('total_runs')),
            normalize_stat(raw[1], radar_max.get('batting_avg')),
            normalize_stat(raw[2], radar_max.get('batting_sr')),
            normalize_stat(raw[3], radar_max.get('total_wickets')),
            normalize_stat(raw[4], radar_max.get('bowling_avg'), True),
            normalize_stat(raw[5], radar_max.get('bowling_econ'), True)]

    hover = [f"{c}: {('N/A' if pd.isna(v) or not np.isfinite(v) else f'{v:.2f}' if isinstance(v, float) else v)}"
             for c, v in zip(cat, raw)]
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=norm, theta=cat, fill='toself',
                                        hoverinfo='text', text=hover, name=player))
    radar_fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])),
                            showlegend=False, title=f"{player} – Normalized Radar ({fmt_label})",
                            title_x=0.5, margin=dict(l=50, r=50, t=70, b=40))

    # ------------- time-series bubbles -------------
    player_df_sorted = player_df.sort_values('start_date')
    runs_bubble = px.scatter(
        player_df_sorted, x='start_date', y='runs_scored',
        hover_data=['opposition_team', 'match_id', 'balls_faced',
                    'fours_scored', 'sixes_scored', 'out_kind'],
        labels={'start_date': 'Match Date', 'runs_scored': 'Runs'},
        title=f"Runs per Match ({fmt_label})")
    runs_bubble.update_traces(marker=dict(size=8))
    runs_bubble.update_layout(title_x=0.5, margin=dict(t=45, b=30, l=30, r=30))

    bowl_df = player_df_sorted[player_df_sorted['balls_bowled'] > 0]
    wickets_bubble = (px.scatter(
        bowl_df, x='start_date', y='wickets_taken',
        hover_data=['opposition_team', 'match_id',
                    'balls_bowled', 'runs_conceded'],
        labels={'start_date': 'Match Date', 'wickets_taken': 'Wickets'},
        title=f"Wickets per Match ({fmt_label})")
        if not bowl_df.empty else empty_fig("No Bowling Data"))
    if not bowl_df.empty:
        wickets_bubble.update_traces(marker=dict(size=8))
        wickets_bubble.update_layout(
            title_x=0.5, margin=dict(t=45, b=30, l=30, r=30))

    # ------------- boundary % calculations ----------------
    player_df_sorted['boundary_runs'] = player_df_sorted['fours_scored'] * \
        4 + player_df_sorted['sixes_scored']*6
    player_df_sorted['boundary_pct'] = np.where(player_df_sorted['runs_scored'] > 0,
                                                player_df_sorted['boundary_runs'] /
                                                player_df_sorted['runs_scored']*100,
                                                np.nan)
    player_df_sorted['strike_rate_match'] = np.where(player_df_sorted['balls_faced'] > 0,
                                                     player_df_sorted['runs_scored'] /
                                                     player_df_sorted['balls_faced']*100,
                                                     np.nan)

    # ------ scatter: boundary % vs balls vs time ----------
    scatter_fig = (px.scatter(
        player_df_sorted, x='start_date', y='boundary_pct',
        color='strike_rate_match', size='balls_faced',
        color_continuous_scale='viridis', size_max=18,
        labels={'start_date': 'Match Date', 'boundary_pct': 'Boundary %',
                'strike_rate_match': 'Strike Rate'},
        title=f"Boundary % vs Balls Faced vs Time ({fmt_label})")
        if not player_df_sorted.empty else empty_fig("No Data"))
    scatter_fig.update_layout(title_x=0.5, margin=dict(t=55, b=40, l=40, r=40))

    # ------ yearly line ----------
    yearly = player_df_sorted.dropna(subset=['boundary_pct']).copy()
    yearly['year'] = yearly['start_date'].dt.year
    yearly_line = (px.line(yearly.groupby('year')['boundary_pct'].mean().reset_index(),
                           x='year', y='boundary_pct', markers=True,
                           labels={'boundary_pct': 'Boundary %',
                                   'year': 'Year'},
                           title=f"Average Boundary % per Year ({fmt_label})")
                   if not yearly.empty else empty_fig("No Yearly Data"))
    yearly_line.update_layout(title_x=0.5, margin=dict(t=55, b=40, l=40, r=40))

    # ------ histogram ----------
    hist_fig = (px.histogram(player_df_sorted.dropna(subset=['boundary_pct']),
                             x='boundary_pct', nbins=25,
                             labels={'boundary_pct': 'Boundary %'},
                             title=f"Boundary % Distribution ({fmt_label})",
                             opacity=0.85, marginal='rug')
                if not player_df_sorted.empty else empty_fig("No Histogram Data"))
    hist_fig.update_layout(title_x=0.5, margin=dict(t=55, b=40, l=40, r=40))

    return radar_fig, runs_bubble, wickets_bubble, scatter_fig, yearly_line, hist_fig, placeholder