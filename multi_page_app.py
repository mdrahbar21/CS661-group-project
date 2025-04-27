import os
import traceback
import threading
from flask_caching import Cache
from google.cloud import bigquery
import dash
from dash import dcc, html, Input, Output, callback, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
#  App & Server Setup
# -----------------------------------------------------------------------------

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./primordial-veld-456613-n6-c5dd57e4037a.json"

app = dash.Dash(__name__, use_pages=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# -----------------------------------------------------------------------------
#  Set up Flask-Caching
# -----------------------------------------------------------------------------

cache = Cache(server, config={
    'CACHE_TYPE': 'simple',            # in-memory cache; swap for redis/memcached in prod
    'CACHE_DEFAULT_TIMEOUT': 60*60     # 1 hour
})

# -----------------------------------------------------------------------------
#  BigQuery client
# -----------------------------------------------------------------------------

bq_client = bigquery.Client()

# -----------------------------------------------------------------------------
#  Your existing load functions (unchanged)
# -----------------------------------------------------------------------------


def load_table_from_bigquery(table_name: str) -> pd.DataFrame:
    """
    Loads an entire table from BigQuery into a pandas DataFrame.
    """
    print(f"--- Loading BigQuery table: {table_name} ---")
    query = f"""
        SELECT *
        FROM `primordial-veld-456613-n6.cs661_gr2_discovered_001.{table_name}`
    """
    try:
        df = bq_client.query(query).to_dataframe()
        print(f"Loaded '{table_name}' from BigQuery. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load '{table_name}' from BigQuery: {e}")
        traceback.print_exc()
        return None


def load_main_data() -> pd.DataFrame:
    print("--- Running load_main_data() [BigQuery: total_data] ---")
    df = load_table_from_bigquery('total_data')
    if df is None or df.empty:
        print("ERROR: BigQuery 'total_data' returned no rows.")
        return None

    # --- Data Cleaning & Standardization ---
    # Numeric conversions
    numeric_cols_main = [
        'runs_scored', 'balls_faced', 'wickets_taken', 'balls_bowled',
        'runs_conceded', 'bowled_done', 'lbw_done', 'player_out',
        'fours_scored', 'sixes_scored', 'runs_off_bat', 'extras',
        'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'win_by_runs', 'win_by_wickets',
        'caught_done', 'stumped_done', 'run_out_direct', 'run_out_throw',
        'run_out_involved', 'dot_balls_as_bowler', 'maidens', 'catches_taken',
        'stumpings_done', 'innings', 'over', 'delivery', 'dot_balls_as_batsman'
    ]
    for col in numeric_cols_main:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col in ['win_by_runs', 'win_by_wickets', 'innings', 'over', 'delivery']:
                df[col] = df[col].astype('Int64')
            else:
                df[col] = df[col].fillna(0)
                # Convert to int if no decimals
                if (df[col] % 1 == 0).all():
                    df[col] = df[col].astype(int)
                else:
                    df[col] = df[col].astype(float)
        else:
            print(
                f"Warning (Main Data): '{col}' missing; skipped numeric conversion.")

    # Categorical / String cleaning
    if 'out_kind' in df.columns:
        df['out_kind'] = df['out_kind'].fillna(
            'not out').str.lower().str.strip()
    else:
        df['out_kind'] = 'not out'

    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    else:
        df['start_date'] = pd.NaT

    if 'match_type' in df.columns:
        allowed = ["ODI", "T20", "TEST", "T20I"]
        df['match_type'] = df['match_type'].astype(str).str.upper()
        df = df[df['match_type'].isin(allowed)].copy()

    # Team name standardization
    team_cols = [
        'player_team', 'opposition_team', 'batting_team',
        'bowling_team', 'winner', 'toss_winner'
    ]
    replacements = {
        'U.A.E.': 'United Arab Emirates', 'UAE': 'United Arab Emirates',
        'P.N.G.': 'Papua New Guinea', 'PNG': 'Papua New Guinea',
        'USA': 'United States of America',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Delhi Daredevils': 'Delhi Capitals'
    }
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(
                str).replace(replacements)
        else:
            df[col] = 'Unknown'

    # Fill other string columns
    for col in ['venue', 'city', 'toss_decision', 'result', 'batsman', 'non_striker',
                'bowler', 'fielder', 'bowler_involved_in_out', 'name', 'bowling_style', 'role', 'event_name']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
        else:
            df[col] = 'Unknown'

    print(f"--- load_main_data() done, shape: {df.shape} ---")

    return df


def load_tournament_data() -> pd.DataFrame:
    print("--- Running load_tournament_data() [BigQuery: mw_overall] ---")
    df = load_table_from_bigquery('mw_overall')
    if df is None or df.empty:
        print("ERROR: BigQuery 'mw_overall' returned no rows.")
        return None

    # Ensure required columns exist; create defaults if missing
    required = [
        'match_id', 'event_name', 'city', 'venue', 'winner', 'toss_winner',
        'toss_decision', 'batting_team', 'runs_off_bat', 'player_dismissed',
        'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'out_kind',
        'season', 'start_date', 'match_type', 'umpire1', 'umpire2'
    ]
    for col in required:
        if col not in df.columns:
            print(f"WARNING (Tournament): '{col}' missing; filling default.")
            df[col] = 0 if col in ['runs_off_bat', 'wides', 'noballs',
                                   'byes', 'legbyes', 'penalty', 'player_dismissed'] else 'Unknown'

    # Numeric conversions
    for col in ['runs_off_bat', 'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'player_dismissed']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Categorical cleaning
    for col in ['event_name', 'winner', 'season', 'batting_team', 'match_id', 'city', 'venue', 'toss_winner', 'toss_decision', 'out_kind']:
        df[col] = df[col].fillna('Unknown').astype(str).str.strip()
    df['out_kind'] = df['out_kind'].str.lower()

    # Map city→country
    city_to_country = {
        # Keep the extensive dictionary from the original multi_page_app2.py
        'Dubai': 'United Arab Emirates', 'Sharjah': 'United Arab Emirates', 'Abu Dhabi': 'United Arab Emirates',
        'London': 'United Kingdom', 'Manchester': 'United Kingdom', 'Birmingham': 'United Kingdom', 'Cardiff': 'United Kingdom', 'Southampton': 'United Kingdom', 'Leeds': 'United Kingdom', 'Chester-le-Street': 'United Kingdom', 'Nottingham': 'United Kingdom', 'Bristol': 'United Kingdom', 'Taunton': 'United Kingdom', 'Hove': 'United Kingdom', "Lord's": 'United Kingdom',
        'Sydney': 'Australia', 'Melbourne': 'Australia', 'Adelaide': 'Australia', 'Perth': 'Australia', 'Brisbane': 'Australia', 'Hobart': 'Australia', 'Canberra': 'Australia', 'Geelong': 'Australia', 'Launceston': 'Australia',
        'Mumbai': 'India', 'Delhi': 'India', 'Kolkata': 'India', 'Chennai': 'India', 'Bengaluru': 'India', 'Bangalore': 'India', 'Hyderabad': 'India', 'Mohali': 'India', 'Nagpur': 'India', 'Pune': 'India', 'Ahmedabad': 'India', 'Dharamsala': 'India', 'Visakhapatnam': 'India', 'Indore': 'India', 'Rajkot': 'India', 'Ranchi': 'India', 'Cuttack': 'India', 'Guwahati': 'India', 'Lucknow': 'India', 'Kanpur': 'India', 'Jaipur': 'India', 'Chandigarh': 'India',
        'Cape Town': 'South Africa', 'Johannesburg': 'South Africa', 'Durban': 'South Africa', 'Centurion': 'South Africa', 'Port Elizabeth': 'South Africa', 'Gqeberha': 'South Africa', 'Paarl': 'South Africa', 'Bloemfontein': 'South Africa', 'East London': 'South Africa', 'Potchefstroom': 'South Africa', 'Kimberley': 'South Africa', 'Benoni': 'South Africa',
        'Auckland': 'New Zealand', 'Wellington': 'New Zealand', 'Christchurch': 'New Zealand', 'Hamilton': 'New Zealand', 'Napier': 'New Zealand', 'Dunedin': 'New Zealand', 'Mount Maunganui': 'New Zealand', 'Queenstown': 'New Zealand', 'Nelson': 'New Zealand',
        'Karachi': 'Pakistan', 'Lahore': 'Pakistan', 'Rawalpindi': 'Pakistan', 'Multan': 'Pakistan', 'Faisalabad': 'Pakistan',
        'Colombo': 'Sri Lanka', 'Kandy': 'Sri Lanka', 'Galle': 'Sri Lanka', 'Hambantota': 'Sri Lanka', 'Dambulla': 'Sri Lanka', 'Pallekele': 'Sri Lanka',
        'Chattogram': 'Bangladesh', 'Chittagong': 'Bangladesh', 'Dhaka': 'Bangladesh', 'Sylhet': 'Bangladesh', 'Mirpur': 'Bangladesh', 'Khulna': 'Bangladesh', 'Fatullah': 'Bangladesh',
        'Harare': 'Zimbabwe', 'Bulawayo': 'Zimbabwe', 'Kwekwe': 'Zimbabwe', 'Mutare': 'Zimbabwe',
        'Bridgetown': 'Barbados', 'Gros Islet': 'Saint Lucia', 'Port of Spain': 'Trinidad and Tobago', 'Kingston': 'Jamaica', 'Providence': 'Guyana', 'North Sound': 'Antigua and Barbuda', 'Basseterre': 'Saint Kitts and Nevis', 'Kingstown': 'Saint Vincent and the Grenadines', 'Roseau': 'Dominica', 'Lauderhill': 'United States',
        'Dublin': 'Ireland', 'Belfast': 'United Kingdom', 'Malahide': 'Ireland', 'Bready': 'United Kingdom',
        'Edinburgh': 'United Kingdom', 'Glasgow': 'United Kingdom', 'Aberdeen': 'United Kingdom',
        'Amstelveen': 'Netherlands', 'Rotterdam': 'Netherlands', 'The Hague': 'Netherlands',
        'Windhoek': 'Namibia', 'Nairobi': 'Kenya', 'Kampala': 'Uganda',
        'Muscat': 'Oman', 'Al Amerat': 'Oman', 'Kathmandu': 'Nepal', 'Kirtipur': 'Nepal',
        'Singapore': 'Singapore', 'Kuala Lumpur': 'Malaysia', 'Hong Kong': 'Hong Kong',
        # Add venue mappings if city is often missing/unreliable
        'Old Trafford': 'United Kingdom',  # Example
    }
    df['country'] = df['city'].map(city_to_country).fillna(
        df['venue'].map(city_to_country)).fillna('Unknown')

    # Compute match_teams_list & bowling_team
    grouped = df[df['batting_team'] != 'Unknown'].groupby(
        'match_id')['batting_team'].unique().apply(lambda x: sorted(set(x)))
    teams_df = grouped.reset_index()
    teams_df = teams_df[
        teams_df['batting_team'].apply(
            lambda t: isinstance(t, list) and len(t) == 2)
    ]
    teams_df.rename(columns={'batting_team': 'match_teams_list'}, inplace=True)

    df = df.merge(teams_df, on='match_id', how='left')

    def derive_bowling(row):
        t = row['match_teams_list']
        b = row['batting_team']
        return t[1] if isinstance(t, list) and len(t) == 2 and t[0] == b else (t[0] if isinstance(t, list) and len(t) == 2 else 'Unknown')
    df['bowling_team'] = df.apply(
        derive_bowling, axis=1).replace({'Unknown': 'Unknown'})

    # Date conversion
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')

    print(f"--- load_tournament_data() done, shape: {df.shape} ---")
    return df


def load_player_analysis_data() -> pd.DataFrame:
    print("--- Running load_player_analysis_data() [BigQuery: app6] ---")
    df = load_table_from_bigquery('app6')
    if df is None or df.empty:
        print("ERROR: BigQuery 'app6' returned no rows.")
        return None

    df['name'] = df['name'].fillna('Unknown').astype(str).str.strip()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    numeric = [
        'runs_scored', 'balls_faced', 'fours_scored', 'sixes_scored', 'catches_taken',
        'run_out_direct', 'run_out_throw', 'stumpings_done', 'player_out', 'balls_bowled',
        'runs_conceded', 'wickets_taken', 'bowled_done', 'lbw_done', 'maidens',
        'dot_balls_as_batsman', 'dot_balls_as_bowler'
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if (df[col] % 1 == 0).all():
            df[col] = df[col].astype(int)
    df.dropna(subset=['name', 'start_date'], inplace=True)
    df.sort_values('start_date', inplace=True)
    print(f"--- load_player_analysis_data() done, shape: {df.shape} ---")
    return df


def load_betting_data():
    print(
        "--- Running load_betting_data() [BigQuery: style_data_with_start_date] ---")
    df = load_table_from_bigquery('style_data_with_start_date')
    if df is None or df.empty:
        print("ERROR: BigQuery 'style_data_with_start_date' returned no rows.")
        return None

    df['name'] = df['name'].fillna('Unknown').astype(str)
    df['match_type'] = df['match_type'].fillna('Unknown').astype(str)
    stat_cols = [
        'balls_against_spin', 'runs_against_spin', 'outs_against_spin',
        'balls_against_right_fast', 'runs_against_right_fast', 'outs_against_right_fast',
        'balls_against_left_fast', 'runs_against_left_fast', 'outs_against_left_fast'
    ]
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    grouped = df.groupby(['name', 'match_type'], as_index=False)[
        stat_cols].sum()
    records = []
    config = {
        'Spin': ('balls_against_spin', 'runs_against_spin', 'outs_against_spin'),
        'Right Fast': ('balls_against_right_fast', 'runs_against_right_fast', 'outs_against_right_fast'),
        'Left Fast': ('balls_against_left_fast', 'runs_against_left_fast', 'outs_against_left_fast')
    }
    for _, row in grouped.iterrows():
        for label, (bcol, rcol, ocol) in config.items():
            if row[bcol] > 0:
                records.append({
                    'name': row['name'], 'match_type': row['match_type'],
                    'Bowling Type': label, 'Total Runs': row[rcol], 'Total Balls': row[bcol], 'Total Outs': row[ocol]
                })
    if not records:
        return pd.DataFrame(columns=['name', 'match_type', 'Bowling Type', 'Total Runs', 'Total Balls', 'Total Outs', 'run_rate', 'out_rate'])

    proc = pd.DataFrame(records)
    proc['run_rate'] = np.where(
        proc['Total Balls'] > 0, proc['Total Runs']*100 / proc['Total Balls'], 0.0)
    proc['out_rate'] = np.where(
        proc['Total Balls'] > 0, proc['Total Outs']*100 / proc['Total Balls'], 0.0)
    print(f"--- load_betting_data() done, shape: {proc.shape} ---")
    return proc


# -----------------------------------------------------------------------------
#  Cached JSON getters
# -----------------------------------------------------------------------------


@cache.memoize()
def get_main_data_json():
    df = load_main_data()
    return df.to_json(orient='split', date_format='iso', date_unit='ms') if df is not None else None


@cache.memoize()
def get_tournament_data_json():
    df = load_tournament_data()
    return df.to_json(orient='split', date_format='iso', date_unit='ms') if df is not None else None


@cache.memoize()
def get_player_analysis_data_json():
    df = load_player_analysis_data()
    return df.to_json(orient='split', date_format='iso', date_unit='ms') if df is not None else None


@cache.memoize()
def get_betting_analyzer_data_json():
    df = load_betting_data()
    return df.to_json(orient='split', date_format='iso', date_unit='ms') if df is not None else None


# -----------------------------------------------------------------------------
# Threading to avoid blocking the main thread
# -----------------------------------------------------------------------------
def warm_up_cache():
    print("\n--- Warming up cache in background ---")
    try:
        get_main_data_json()
        get_tournament_data_json()
        get_player_analysis_data_json()
        get_betting_analyzer_data_json()
        print("--- Cache warm-up done ---\n")
    except Exception as e:
        print(f"WARNING: Cache warm-up failed: {e}")


threading.Thread(target=warm_up_cache, daemon=True).start()


# -----------------------------------------------------------------------------
#  Layout
# -----------------------------------------------------------------------------

app.layout = dbc.Container([

    dcc.Location(id='url', refresh=False),

    dcc.Loading(
        id="full-page-loading",
        type="cube",
        fullscreen=True,
        children=[
            # --- All dcc.Stores ---
            dcc.Store(id='main-data-store'),
            dcc.Store(id='tournament-data-store'),
            dcc.Store(id='player-analysis-store'),
            dcc.Store(id='betting-analyzer-data-store'),

            dbc.Row([
                # --- Sidebar ---
                dbc.Col([
                    html.Div([
                        html.H2("🏏 Dashboard", className="display-6",
                                style={"padding": "20px 10px"}),
                        html.Hr(),
                        dbc.Nav([
                            dbc.NavLink(
                                page["name"],
                                href=page["path"],
                                active="exact",
                                style={"margin": "5px 0"}
                            ) for page in dash.page_registry.values()
                        ],
                            vertical=True,
                            pills=True,
                            className="flex-column",
                            style={"padding": "10px"}
                        )
                    ], style={
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "bottom": 0,
                        "width": "16rem",
                        "backgroundColor": "#f8f9fa",
                        "padding": "2rem 1rem",
                        "overflowY": "auto"
                    })
                    # Sidebar occupies 2/12 grid
                ], width=2, style={"padding": "0"}),

                # --- Page Content ---
                dbc.Col([
                    dash.page_container
                ], style={
                    "marginLeft": "16rem",
                    "padding": "2rem 1rem",
                    "overflowX": "hidden"
                })
            ])
        ]
    )

], fluid=True)


# -----------------------------------------------------------------------------
#  Lazy‐load callbacks
# -----------------------------------------------------------------------------

# when user navigates to any of these pages, populate main-data-store


@callback(
    Output('main-data-store', 'data'),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def load_main_store(pathname):
    if pathname in [
        '/analyzer', '/heatmap', '/recent-form',
        '/radar-bubble', '/performance-analysis'
    ]:
        return get_main_data_json()
    return no_update

# when user navigates to any of these pages, populate tournament-data-store


@callback(
    Output('tournament-data-store', 'data'),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def load_tournament_store(pathname):
    if pathname in ['/tournaments', '/match-explorer']:
        return get_tournament_data_json()
    return no_update

# when user navigates to /player-analysis, populate player-analysis-data-store


@callback(
    Output('player-analysis-store', 'data'),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def load_player_analysis_store(pathname):
    if pathname == '/player-analysis':
        return get_player_analysis_data_json()
    return no_update


@callback(
    Output('betting-analyzer-data-store', 'data'),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def load_betting_analyzer_store(pathname):
    if pathname == '/betting-analyzer':
        return get_betting_analyzer_data_json()
    return no_update


# -----------------------------------------------------------------------------
#  Run
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(debug=False, port=8050)
