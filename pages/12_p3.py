# pages/07_all_time_rankings.py

import dash
from dash import dcc, html, Input, Output, callback, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import traceback
from collections import OrderedDict # For ordered column definitions

# --- Register Page ---
dash.register_page(
    __name__,
    name='All-Time Rankings',             # Name for the sidebar
    path='/all-time-rankings',           # URL path
    title='All-Time Top 100 Rankings'   # Browser tab title
)

# --- Helper Function: Deserialize Data ---
def deserialize_data(stored_data):
    """Deserializes JSON data (from dcc.Store) back into a Pandas DataFrame."""
    if stored_data is None:
        return None
    try:
        df = pd.read_json(stored_data, orient='split')
        # Ensure relevant columns are numeric after deserialization if needed
        numeric_cols = ['runs_scored', 'balls_faced', 'player_out', 'fours_scored', 'sixes_scored',
                        'wickets_taken', 'runs_conceded', 'balls_bowled']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert date if needed (less critical here)
        if 'start_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['start_date']):
             df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')

        # Handle potential boolean conversion if player_out was stored differently
        if 'player_out' in df.columns and df['player_out'].dtype == 'bool':
             df['player_out'] = df['player_out'].astype(int)

        if df.empty:
            return df
        return df
    except Exception as e:
        print(f"ERROR: Failed to deserialize data (All-Time Rankings): {e}")
        traceback.print_exc()
        return None

# --- Configuration ---
TOP_N = 100 # Number of top performers to show in table
GRAPH_TOP_N = 20 # Number of performers to show in graphs

# Define Ranking Metrics and Configurations
# Structure: {metric_key: {'label': 'Display Label', 'higher_is_better': True/False, 'format': 'plotly_format_string or None'}}
# Batting Metrics
BATSMAN_METRICS = OrderedDict([
    ('total_runs', {'label': 'Total Runs', 'higher_is_better': True, 'format': ',.0f'}),
    ('average', {'label': 'Batting Average', 'higher_is_better': True, 'format': '.2f'}),
    ('strike_rate', {'label': 'Strike Rate', 'higher_is_better': True, 'format': '.2f'}),
    ('total_fours', {'label': 'Total Fours', 'higher_is_better': True, 'format': ',.0f'}),
    ('total_sixes', {'label': 'Total Sixes', 'higher_is_better': True, 'format': ',.0f'}),
    ('innings', {'label': 'Innings Played', 'higher_is_better': True, 'format': ',.0f'}),
])
MIN_INNINGS_BATTED = 20 # Qualification threshold

# Bowling Metrics
BOWLER_METRICS = OrderedDict([
    ('wickets', {'label': 'Total Wickets', 'higher_is_better': True, 'format': ',.0f'}),
    ('bowling_average', {'label': 'Bowling Average', 'higher_is_better': False, 'format': '.2f'}), # Lower is better
    ('economy', {'label': 'Economy Rate', 'higher_is_better': False, 'format': '.2f'}), # Lower is better
    ('bowling_strike_rate', {'label': 'Bowling Strike Rate', 'higher_is_better': False, 'format': '.2f'}), # Lower is better
    ('matches_bowled', {'label': 'Matches Bowled', 'higher_is_better': True, 'format': ',.0f'}),
])
MIN_BALLS_BOWLED = 600 # Qualification threshold (e.g., 100 overs)

# Team Metrics (Using Win % as the primary example)
TEAM_METRICS = OrderedDict([
    ('win_percentage', {'label': 'Win Percentage', 'higher_is_better': True, 'format': '.1f'}),
    ('total_wins', {'label': 'Total Wins', 'higher_is_better': True, 'format': ',.0f'}),
    ('matches_played', {'label': 'Matches Played', 'higher_is_better': True, 'format': ',.0f'}),
])
MIN_MATCHES_PLAYED = 25 # Qualification threshold for teams

# All metrics combined for dynamic dropdown population
ALL_METRICS = {
    'Batsman': BATSMAN_METRICS,
    'Bowler': BOWLER_METRICS,
    'Team': TEAM_METRICS
}

# Define secondary metrics for scatter plots {primary_metric: (secondary_metric_key, secondary_metric_label)}
SECONDARY_SCATTER_METRIC = {
    'Batsman': {
        'total_runs': ('average', 'Batting Average'),
        'average': ('strike_rate', 'Strike Rate'),
        'strike_rate': ('average', 'Batting Average'),
        'total_fours': ('total_sixes', 'Total Sixes'),
        'total_sixes': ('strike_rate', 'Strike Rate'),
        'innings': ('total_runs', 'Total Runs'),
    },
    'Bowler': {
        'wickets': ('bowling_average', 'Bowling Average'),
        'bowling_average': ('economy', 'Economy Rate'),
        'economy': ('bowling_strike_rate', 'Bowling Strike Rate'),
        'bowling_strike_rate': ('wickets', 'Total Wickets'),
        'matches_bowled': ('wickets', 'Total Wickets'),
    },
    'Team': {
        'win_percentage': ('total_wins', 'Total Wins'),
        'total_wins': ('matches_played', 'Matches Played'),
        'matches_played': ('win_percentage', 'Win Percentage'),
    }
}


# --- Layout Definition ---
layout = dbc.Container([
    html.H1(f"All-Time Top {TOP_N} Performers", className="text-center my-4"),

    # Control Row
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                # Category Selection (Batsman/Bowler/Team)
                dbc.Col([
                    html.Label("Select Category:", className="fw-bold"),
                    dbc.RadioItems(
                        id='atr-category-select',
                        options=[
                            {'label': 'Batsman', 'value': 'Batsman'},
                            {'label': 'Bowler', 'value': 'Bowler'},
                            {'label': 'Team', 'value': 'Team'},
                        ],
                        value='Batsman', # Default selection
                        inline=True,
                        inputClassName="me-1",
                        labelStyle={'margin-right': '15px'}
                    )
                ], width=12, md=4, className="mb-3"),

                # Ranking Metric Selection
                dbc.Col([
                    html.Label("Rank By Metric:", className="fw-bold"),
                    dcc.Dropdown(
                        id='atr-metric-select',
                        # Options populated dynamically
                        clearable=False
                    )
                ], width=12, md=4, className="mb-3"),

                 # Match Type Filter
                dbc.Col([
                    html.Label("Filter by Match Type:", className="fw-bold"),
                    dcc.Dropdown(
                        id='atr-match-type-filter',
                        options=[
                            {'label': 'All International (ODI, T20I, Test)', 'value': 'All International'},
                            {'label': 'ODI', 'value': 'ODI'},
                            {'label': 'T20I', 'value': 'T20I'},
                            {'label': 'Test', 'value': 'Test'},
                            {'label': 'All Formats', 'value': 'All Formats'}, # Include domestic T20 etc.
                        ],
                        value='All International', # Default filter
                        clearable=False
                    )
                ], width=12, md=4, className="mb-3"),
            ]),
            # Row to display qualification criteria
            dbc.Row(dbc.Col(
                 html.Small(id='atr-qualification-info', className="text-muted fst-italic")
            ))
        ]), className="shadow-sm mb-4"
    ),

    # Visualizations Row
    html.Div(id='atr-visualizations', children=[
        dbc.Row([
            # Bar Chart
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5(id='atr-bar-chart-title', className="card-title text-center"),
                        dcc.Loading(dcc.Graph(id='atr-ranking-bar-chart', figure=go.Figure())) # Initialize with empty figure
                    ])
                ), width=12, lg=6, className="mb-3" # Use half width on large+ screens
            ),
            # Scatter Plot
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5(id='atr-scatter-plot-title', className="card-title text-center"),
                        dcc.Loading(dcc.Graph(id='atr-ranking-scatter-plot', figure=go.Figure())) # Use the correct ID here
                    ])
                ), width=12, lg=6, className="mb-3" # Use half width on large+ screens
            )
        ], className="mb-4"), # Add some bottom margin
    ]),

    # Results Table
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4(f"Detailed Ranking Table (Top {TOP_N})", className="text-center mb-3"),
                    dcc.Loading(
                        dash_table.DataTable(
                            id='atr-ranking-table',
                            columns=[], # Populated by callback
                            data=[],    # Populated by callback
                            page_size=20,
                            style_table={'overflowX': 'auto'}, # Enable horizontal scroll
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            },
                            style_cell={
                                'textAlign': 'left',
                                'padding': '5px',
                                'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'Rank'},
                                    'fontWeight': 'bold',
                                    'backgroundColor': 'rgb(240, 240, 240)'
                                }
                            ],
                            sort_action="native",
                            filter_action="native",
                        )
                    )
                ])
            ), width=12
        )
    ])

], fluid=True)

# --- Callback Helper Functions ---

def create_empty_figure(message="No data to display"):
    """Creates an empty Plotly figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="grey")
    )
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def get_metric_config(category, metric_key):
    """Safely get the config dict for a metric."""
    return ALL_METRICS.get(category, {}).get(metric_key, {})

def get_metric_label(category, metric_key):
    """Safely get the display label for a metric."""
    return get_metric_config(category, metric_key).get('label', metric_key.replace('_', ' ').title())

# --- Callbacks ---

# Callback 1: Update Metric Dropdown and Qualification Info based on Category
@callback(
    Output('atr-metric-select', 'options'),
    Output('atr-metric-select', 'value'),
    Output('atr-qualification-info', 'children'),
    Input('atr-category-select', 'value')
)
def update_metric_options_and_info(selected_category):
    """Updates the ranking metric dropdown and qualification text."""
    metrics = ALL_METRICS.get(selected_category, {})
    options = [{'label': v['label'], 'value': k} for k, v in metrics.items()]
    default_value = list(metrics.keys())[0] if metrics else None

    qual_text = "Qualification Criteria: "
    if selected_category == 'Batsman':
        qual_text += f"Minimum {MIN_INNINGS_BATTED} innings batted."
    elif selected_category == 'Bowler':
        qual_text += f"Minimum {MIN_BALLS_BOWLED} balls bowled."
    elif selected_category == 'Team':
        qual_text += f"Minimum {MIN_MATCHES_PLAYED} matches played."
    else:
        qual_text = ""

    return options, default_value, qual_text

# Callback 2: Update Ranking Table AND VISUALIZATIONS based on selections
@callback(
    Output('atr-ranking-table', 'columns'),
    Output('atr-ranking-table', 'data'),
    Output('atr-ranking-bar-chart', 'figure'),
    Output('atr-ranking-scatter-plot', 'figure'), # Corrected ID
    Output('atr-bar-chart-title', 'children'),
    Output('atr-scatter-plot-title', 'children'),
    Output('atr-visualizations', 'style'), # To hide graphs if no data
    Input('atr-category-select', 'value'),
    Input('atr-metric-select', 'value'),
    Input('atr-match-type-filter', 'value'),
    State('main-data-store', 'data'),
    prevent_initial_call=True
)
def update_rankings_and_plots(category, metric, match_type_filter, stored_main_data):
    """Calculates rankings and updates the DataTable and associated plots."""
    print(f"ATR Callback: Updating table & plots for Category={category}, Metric={metric}, Filter={match_type_filter}")

    # --- Default empty outputs ---
    empty_table_cols = [{"name": "Info", "id": "info"}]
    empty_table_data = [{"info": "Select options to generate rankings."}]
    empty_fig = create_empty_figure()
    no_data_fig = create_empty_figure("No qualifying data found for these selections.")
    error_fig = create_empty_figure("An error occurred during analysis.")
    hidden_style = {'display': 'none'}
    visible_style = {'display': 'block'}
    default_bar_title = "Top Performers"
    default_scatter_title = "Metric Comparison"

    # --- Initial Checks ---
    if not category or not metric:
        print("ATR Warning: Category or Metric not selected.")
        return empty_table_cols, empty_table_data, empty_fig, empty_fig, default_bar_title, default_scatter_title, hidden_style

    df_main = deserialize_data(stored_main_data)
    if df_main is None or df_main.empty:
        print("ATR Error: Main data not available.")
        error_cols = [{"name": "Error", "id": "error"}]
        error_data = [{"error": "Data not loaded or empty"}]
        error_fig_data = create_empty_figure("Error: Data not loaded")
        return error_cols, error_data, error_fig_data, error_fig_data, "Error", "Error", visible_style

    # Check if the selected metric exists for the category
    metric_config = get_metric_config(category, metric)
    if not metric_config:
         print(f"ATR Error: Invalid metric '{metric}' for category '{category}'.")
         error_cols = [{"name": "Error", "id": "error"}]
         error_data = [{"error": f"Invalid Metric '{metric}' for {category}"}]
         error_fig_metric = create_empty_figure("Error: Invalid Metric Selection")
         return error_cols, error_data, error_fig_metric, error_fig_metric, "Error", "Error", visible_style

    metric_label = metric_config.get('label', metric.replace('_', ' ').title())
    metric_format = metric_config.get('format', '') # Use empty string if no format

    # --- Filtering by Match Type ---
    df_filtered = df_main.copy()
    if 'match_type' in df_filtered.columns:
        if match_type_filter == 'All International':
            allowed_types = ['ODI', 'T20I', 'Test']
            df_filtered = df_filtered[df_filtered['match_type'].astype(str).isin(allowed_types)]
        elif match_type_filter != 'All Formats': # Specific format selected
            df_filtered = df_filtered[df_filtered['match_type'].astype(str) == match_type_filter]
        # Else ('All Formats'), no filtering needed
    else:
        print("ATR Warning: 'match_type' column missing, cannot filter by format.")

    if df_filtered.empty:
        print(f"ATR Info: No data found after filtering for match type: {match_type_filter}")
        no_data_cols = [{"name": "Info", "id": "info"}]
        no_data_data = [{"info": f"No data found for Match Type: {match_type_filter}"}]
        no_data_fig_filter = create_empty_figure(f"No data for {match_type_filter}")
        return no_data_cols, no_data_data, no_data_fig_filter, no_data_fig_filter, f"No Data ({match_type_filter})", f"No Data ({match_type_filter})", visible_style

    # --- Aggregation and Ranking ---
    df_ranked = pd.DataFrame()
    required_columns_map = {} # Stores {col_id: {'label': 'Label', 'format': 'fmt'}}
    entity_name_col = 'name' # Default for player, change for team

    try:
        if category == 'Batsman':
            req_cols = ['name', 'match_id', 'runs_scored', 'balls_faced', 'player_out', 'fours_scored', 'sixes_scored']
            if not all(c in df_filtered.columns for c in req_cols):
                missing = [c for c in req_cols if c not in df_filtered.columns]
                raise ValueError(f"Missing required columns for Batsman analysis: {missing}")

            # Filter out rows where name is NaN/None if necessary before grouping
            df_filtered.dropna(subset=['name'], inplace=True)
            if df_filtered.empty: raise ValueError("No valid batsman data after filtering.")

            # Calculate innings ensuring a player bats (balls_faced > 0)
            batted_in_match = df_filtered[df_filtered['balls_faced'] > 0].groupby(['name', 'match_id']).size().reset_index()
            innings_count = batted_in_match.groupby('name')['match_id'].nunique().reset_index(name='innings')

            player_stats = df_filtered.groupby('name', as_index=False).agg(
                total_runs=('runs_scored', 'sum'),
                balls_faced=('balls_faced', 'sum'),
                outs=('player_out', 'sum'),
                total_fours=('fours_scored', 'sum'),
                total_sixes=('sixes_scored', 'sum')
            )
            player_stats = pd.merge(player_stats, innings_count, on='name', how='left').fillna({'innings': 0})
            player_stats = player_stats[player_stats['innings'] >= MIN_INNINGS_BATTED].copy() # Use copy to avoid SettingWithCopyWarning

            if player_stats.empty: raise ValueError(f"No batsmen qualify with >= {MIN_INNINGS_BATTED} innings.")

            # Calculate derived metrics carefully checking for division by zero
            player_stats['average'] = np.where(player_stats['outs'] > 0, player_stats['total_runs'] / player_stats['outs'], np.nan)
            player_stats['strike_rate'] = np.where(player_stats['balls_faced'] > 0, (player_stats['total_runs'] / player_stats['balls_faced']) * 100, np.nan)

            required_columns_map = {
                 'Rank': {'label': 'Rank'}, 'name': {'label': 'Player'},
                 'total_runs': {'label': 'Runs', 'format': ',.0f'}, 'average': {'label': 'Avg', 'format': '.2f'},
                 'strike_rate': {'label': 'SR', 'format': '.2f'}, 'innings': {'label': 'Inns', 'format': ',.0f'},
                 'total_fours': {'label': '4s', 'format': ',.0f'}, 'total_sixes': {'label': '6s', 'format': ',.0f'},
            }
            df_ranked = player_stats

        elif category == 'Bowler':
            req_cols = ['name', 'match_id', 'wickets_taken', 'runs_conceded', 'balls_bowled']
            if not all(c in df_filtered.columns for c in req_cols):
                 missing = [c for c in req_cols if c not in df_filtered.columns]
                 raise ValueError(f"Missing required columns for Bowler analysis: {missing}")

            df_filtered.dropna(subset=['name'], inplace=True)
            if df_filtered.empty: raise ValueError("No valid bowler data after filtering.")

            # Calculate matches bowled ensuring player bowls (balls_bowled > 0)
            bowled_in_match = df_filtered[df_filtered['balls_bowled'] > 0].groupby(['name', 'match_id']).size().reset_index()
            matches_bowled_count = bowled_in_match.groupby('name')['match_id'].nunique().reset_index(name='matches_bowled')

            player_stats = df_filtered.groupby('name', as_index=False).agg(
                wickets=('wickets_taken', 'sum'),
                runs_conceded=('runs_conceded', 'sum'),
                balls_bowled=('balls_bowled', 'sum')
            )
            player_stats = pd.merge(player_stats, matches_bowled_count, on='name', how='left').fillna({'matches_bowled': 0})
            player_stats = player_stats[player_stats['balls_bowled'] >= MIN_BALLS_BOWLED].copy()

            if player_stats.empty: raise ValueError(f"No bowlers qualify with >= {MIN_BALLS_BOWLED} balls bowled.")

            player_stats['bowling_average'] = np.where(player_stats['wickets'] > 0, player_stats['runs_conceded'] / player_stats['wickets'], np.nan)
            player_stats['economy'] = np.where(player_stats['balls_bowled'] > 0, (player_stats['runs_conceded'] / player_stats['balls_bowled']) * 6, np.nan)
            player_stats['bowling_strike_rate'] = np.where(player_stats['wickets'] > 0, player_stats['balls_bowled'] / player_stats['wickets'], np.nan)

            required_columns_map = {
                 'Rank': {'label': 'Rank'}, 'name': {'label': 'Player'},
                 'wickets': {'label': 'Wkts', 'format': ',.0f'}, 'bowling_average': {'label': 'Avg', 'format': '.2f'},
                 'economy': {'label': 'Econ', 'format': '.2f'}, 'bowling_strike_rate': {'label': 'SR', 'format': '.2f'},
                 'matches_bowled': {'label': 'Matches', 'format': ',.0f'}, 'balls_bowled': {'label': 'Balls', 'format': ',.0f'},
            }
            df_ranked = player_stats

        elif category == 'Team':
            entity_name_col = 'team_name' # Use a distinct name for the team column
            req_cols = ['match_id', 'winner', 'batting_team', 'bowling_team']
            if not all(c in df_filtered.columns for c in req_cols):
                 missing = [c for c in req_cols if c not in df_filtered.columns]
                 raise ValueError(f"Missing required columns for Team analysis: {missing}")

            # Ensure team names are consistent and handle NaNs
            df_filtered['batting_team'] = df_filtered['batting_team'].astype(str)
            df_filtered['bowling_team'] = df_filtered['bowling_team'].astype(str)
            df_filtered['winner'] = df_filtered['winner'].astype(str)
            df_filtered.dropna(subset=['batting_team', 'bowling_team', 'winner', 'match_id'], inplace=True)

            match_info = df_filtered.drop_duplicates(subset='match_id')[['match_id', 'winner']].copy()
            teams_in_matches = pd.concat([
                df_filtered[['match_id', 'batting_team']].rename(columns={'batting_team': entity_name_col}),
                df_filtered[['match_id', 'bowling_team']].rename(columns={'bowling_team': entity_name_col})
            ]).drop_duplicates()
            # Filter out potential placeholder/NaN team names if needed
            teams_in_matches = teams_in_matches[teams_in_matches[entity_name_col].str.lower() != 'nan']

            if teams_in_matches.empty: raise ValueError("No valid team data found.")

            team_stats = pd.merge(teams_in_matches, match_info, on='match_id', how='left')
            team_stats['is_winner'] = np.where(team_stats[entity_name_col] == team_stats['winner'], 1, 0)

            agg_stats = team_stats.groupby(entity_name_col).agg(
                matches_played=('match_id', 'nunique'),
                total_wins=('is_winner', 'sum')
            ).reset_index()

            agg_stats = agg_stats[agg_stats['matches_played'] >= MIN_MATCHES_PLAYED].copy()

            if agg_stats.empty: raise ValueError(f"No teams qualify with >= {MIN_MATCHES_PLAYED} matches played.")

            agg_stats['win_percentage'] = np.where(agg_stats['matches_played'] > 0, (agg_stats['total_wins'] / agg_stats['matches_played']) * 100, 0.0)

            required_columns_map = {
                 'Rank': {'label': 'Rank'}, entity_name_col: {'label': 'Team'}, # Use the entity name col here
                 'win_percentage': {'label': 'Win %', 'format': '.1f'},
                 'total_wins': {'label': 'Wins', 'format': ',.0f'},
                 'matches_played': {'label': 'Played', 'format': ',.0f'},
            }
            df_ranked = agg_stats.rename(columns={entity_name_col: 'name'}) # Rename to 'name' for internal consistency AFTER map defined
            entity_name_col = 'name' # Now use 'name' internally

        # --- Sorting and Ranking (Common Logic) ---
        if metric not in df_ranked.columns:
             # This case should ideally be caught by the initial metric check, but good failsafe
             raise ValueError(f"Metric column '{metric}' not found after aggregation for {category}.")

        # Drop rows where the ranking metric itself is NaN (essential for ranking)
        df_ranked.dropna(subset=[metric], inplace=True)
        if df_ranked.empty:
             raise ValueError(f"No players/teams remain after removing NaN values for ranking metric '{metric_label}'.")

        ascending_order = not metric_config.get('higher_is_better', True) # Default to True if missing
        df_ranked.sort_values(by=metric, ascending=ascending_order, inplace=True, na_position='last')

        # Add Rank column AFTER sorting
        df_ranked.insert(0, 'Rank', range(1, len(df_ranked) + 1))

        # Select Top N for Table
        df_top_n_table = df_ranked.head(TOP_N).copy()

        # --- Prepare for Dash DataTable ---
        dt_columns = []
        display_cols_order = list(required_columns_map.keys()) # Get order from map

        # Ensure the entity column name matches what's in df_ranked ('name')
        if 'team_name' in display_cols_order and category == 'Team':
            display_cols_order[display_cols_order.index('team_name')] = 'name'
            required_columns_map['name'] = required_columns_map.pop('team_name')


        df_display_table = df_top_n_table[[col for col in display_cols_order if col in df_top_n_table.columns]]

        for col_id in df_display_table.columns:
            col_info = required_columns_map.get(col_id, {})
            col_label = col_info.get('label', col_id.replace('_', ' ').title())
            col_format = col_info.get('format')
            col_def = {"name": col_label, "id": col_id}

            # Check if column exists and is numeric before applying numeric format
            if col_id in df_display_table.columns and pd.api.types.is_numeric_dtype(df_display_table[col_id]):
                 col_def["type"] = "numeric"
                 if col_format:
                    col_def["format"] = {'specifier': col_format}
            dt_columns.append(col_def)

        dt_data = df_display_table.to_dict('records')

        # --- Prepare Data for Graphs (Top N for Graphs) ---
        df_graph_data = df_ranked.head(GRAPH_TOP_N).copy()
        graph_entity_label = required_columns_map.get(entity_name_col, {}).get('label', 'Entity')

        # --- Create Bar Chart ---
        bar_fig = create_empty_figure(f"Not enough data for {metric_label} bar chart.")
        bar_title = f"Top {min(GRAPH_TOP_N, len(df_graph_data))} {category}s by {metric_label}"
        if not df_graph_data.empty:
            try:
                # Define columns needed for hover data (exclude name and the main metric)
                bar_hover_cols = [col for col in required_columns_map if col in df_graph_data.columns and col not in [entity_name_col, metric, 'Rank']]
                custom_data_bar = df_graph_data[bar_hover_cols]

                # Construct the hover template string
                hovertemplate_bar = f"<b>%{{customdata[0]}}</b><br><br>" # Use customdata[0] for name
                hovertemplate_bar += f"{metric_label}=%{{y:{metric_format}}}<br>" # Main metric (y-axis)

                # Add Rank to hover data
                hovertemplate_bar += f"Rank=%{{customdata[1]}}<br>"

                # Start custom data index from 2 for other stats
                for i, col in enumerate(bar_hover_cols[1:], 2): # Skip 'name' already added
                    label = required_columns_map[col]['label']
                    fmt = required_columns_map[col].get('format', '') # Get format specifier
                    hovertemplate_bar += f"{label}=%{{customdata[{i}]:{fmt}}}<br>"

                hovertemplate_bar += "<extra></extra>" # Hide the trace info

                # Prepare customdata list for px.bar (must match template indices)
                bar_custom_data_cols = [entity_name_col, 'Rank'] + bar_hover_cols[1:]


                bar_fig = px.bar(
                    df_graph_data,
                    x=entity_name_col,
                    y=metric,
                    title=None,
                    labels={entity_name_col: graph_entity_label, metric: metric_label},
                    text=metric,
                    custom_data=bar_custom_data_cols, # Pass the columns needed for the template
                )
                bar_fig.update_traces(
                    texttemplate=f'%{{text:{metric_format}}}',
                    textposition='outside',
                    hovertemplate=hovertemplate_bar # Apply the custom hover template
                )
                bar_fig.update_layout(
                     xaxis={'categoryorder':'total descending' if metric_config.get('higher_is_better', True) else 'total ascending'},
                     yaxis_title=metric_label,
                     xaxis_title=graph_entity_label,
                     margin=dict(t=20, b=10, l=10, r=10),
                     hovermode='x unified',
                     title=None
                )
            except Exception as e:
                print(f"ATR Error creating bar chart: {e}")
                traceback.print_exc() # Print traceback for debugging hover issues
                bar_fig = create_empty_figure(f"Error generating bar chart for {metric_label}")


        # --- Create Scatter Plot ---
        scatter_fig = create_empty_figure("Cannot generate scatter plot.")
        scatter_title = "Comparison Plot"
        sec_metric_key, sec_metric_label = SECONDARY_SCATTER_METRIC.get(category, {}).get(metric, (None, None))

        if sec_metric_key and sec_metric_key in df_graph_data.columns:
            sec_metric_config = get_metric_config(category, sec_metric_key)
            sec_metric_format = sec_metric_config.get('format', '')

            # Determine color/size metric (use a tertiary stat if available)
            color_metric, color_label = None, None
            color_metric_format = ''
            if category == 'Batsman' and 'innings' in df_graph_data.columns: color_metric, color_label = 'innings', get_metric_label('Batsman', 'innings')
            elif category == 'Bowler' and 'matches_bowled' in df_graph_data.columns: color_metric, color_label = 'matches_bowled', get_metric_label('Bowler', 'matches_bowled')
            elif category == 'Team' and 'matches_played' in df_graph_data.columns: color_metric, color_label = 'matches_played', get_metric_label('Team', 'matches_played')
            if color_metric: color_metric_format = get_metric_config(category, color_metric).get('format', '')


            # Drop NaNs for scatter plot axes AND color metric before proceeding
            scatter_req_cols = [metric, sec_metric_key]
            if color_metric: scatter_req_cols.append(color_metric)
            df_graph_data_scatter = df_graph_data.dropna(subset=scatter_req_cols).copy()


            if not df_graph_data_scatter.empty:
                scatter_title = f"{metric_label} vs. {sec_metric_label} (Top {min(GRAPH_TOP_N, len(df_graph_data_scatter))} {category}s)"

                # Define columns for scatter hover, excluding x, y, hover_name, color, size
                scatter_exclude_cols = [entity_name_col, metric, sec_metric_key, 'Rank']
                if color_metric: scatter_exclude_cols.append(color_metric)

                scatter_hover_cols = [col for col in required_columns_map if col in df_graph_data_scatter.columns and col not in scatter_exclude_cols]

                # Construct the scatter hover template
                hovertemplate_scatter = f"<b>%{{customdata[0]}}</b><br><br>" # Name
                hovertemplate_scatter += f"Rank=%{{customdata[1]}}<br>" # Rank
                hovertemplate_scatter += f"{metric_label}=%{{x:{metric_format}}}<br>" # x-axis
                hovertemplate_scatter += f"{sec_metric_label}=%{{y:{sec_metric_format}}}<br>" # y-axis
                if color_metric:
                    # Use marker.size for size value, marker.color for color value if needed (check syntax)
                    # For simplicity, using customdata if color is also passed there. Check if color needs separate handling.
                     hovertemplate_scatter += f"{color_label}=%{{customdata[2]:{color_metric_format}}}<br>" # color axis value

                # Index starts from 3 if color metric is present, else 2
                start_idx = 3 if color_metric else 2
                for i, col in enumerate(scatter_hover_cols, start_idx):
                    label = required_columns_map[col]['label']
                    fmt = required_columns_map[col].get('format', '')
                    hovertemplate_scatter += f"{label}=%{{customdata[{i}]:{fmt}}}<br>"

                hovertemplate_scatter += "<extra></extra>"

                # Prepare customdata list for px.scatter
                scatter_custom_data_cols = [entity_name_col, 'Rank']
                if color_metric: scatter_custom_data_cols.append(color_metric)
                scatter_custom_data_cols.extend(scatter_hover_cols)


                try:
                    scatter_fig = px.scatter(
                        df_graph_data_scatter, # Use NaN-dropped data
                        x=metric,
                        y=sec_metric_key,
                        title=None,
                        labels={metric: metric_label, sec_metric_key: sec_metric_label, color_metric: color_label},
                        custom_data=scatter_custom_data_cols, # Pass list of columns
                        color=color_metric,
                        color_continuous_scale=px.colors.sequential.Blues_r if category == 'Bowler' and ('average' in sec_metric_key or 'economy' in sec_metric_key or 'strike_rate' in sec_metric_key) else px.colors.sequential.Blues, # Reverse scale if lower is better for Y
                        size=color_metric if color_metric else None,
                        size_max=15
                    )
                    scatter_fig.update_traces(
                        hovertemplate=hovertemplate_scatter # Apply custom template
                    )
                    scatter_fig.update_layout(
                        xaxis_title=metric_label,
                        yaxis_title=sec_metric_label,
                        coloraxis_colorbar=dict(title=color_label) if color_label else None,
                        margin=dict(t=20, b=10, l=10, r=10),
                        title=None
                    )
                    # Apply specific axis tick formatting
                    if metric_format: scatter_fig.update_xaxes(tickformat=metric_format)
                    if sec_metric_format: scatter_fig.update_yaxes(tickformat=sec_metric_format)

                except Exception as e:
                    print(f"ATR Error creating scatter plot: {e}")
                    traceback.print_exc()
                    scatter_fig = create_empty_figure(f"Error generating plot for {metric_label} vs {sec_metric_label}")
                    scatter_title = f"Plot Error"
            else:
                 scatter_fig = create_empty_figure(f"Not enough valid data points for {metric_label} vs {sec_metric_label} plot.")
                 scatter_title = f"{metric_label} vs {sec_metric_label} (No Data)"
        else:
            scatter_fig = create_empty_figure(f"No secondary metric defined or available for {metric_label}.")
            scatter_title = f"Comparison Plot (Not Available for {metric_label})"


        print(f"ATR Update: Successfully generated Top {len(df_top_n_table)} table and plots for {category} by {metric}")
        return dt_columns, dt_data, bar_fig, scatter_fig, bar_title, scatter_title, visible_style

    except ValueError as ve: # Catch qualification errors or missing data errors specifically
        print(f"ATR Info: {ve}")
        info_cols = [{"name": "Info", "id": "info"}]
        info_data = [{"info": str(ve)}]
        info_fig = create_empty_figure(str(ve))
        return info_cols, info_data, info_fig, info_fig, "No Data", "No Data", visible_style # Show viz area but with no data message
    except Exception as e:
        print(f"ATR Error during processing: {e}")
        traceback.print_exc()
        error_cols = [{"name": "Error", "id": "error"}]
        error_data = [{"error": "An error occurred during analysis. Check logs."}]
        return error_cols, error_data, error_fig, error_fig, "Error", "Error", visible_style