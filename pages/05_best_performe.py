# pages/04_performance_analysis.py

import dash
from dash import dcc, html, Input, Output, callback, State, Patch
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import traceback  # For detailed error printing

# --- Register Page ---
# Registers the page with the Dash Pages system
dash.register_page(
    __name__,
    name='Performance Analysis',  # Name displayed in the navigation
    path='/performance-analysis',  # URL path for this page
    title='Player Performance Analysis'  # Browser tab title
)

# --- Helper Function ---


def deserialize_data(stored_data):
    """
    Deserializes JSON data (expected from dcc.Store) back into a Pandas DataFrame.
    Handles potential errors during deserialization.
    """
    if stored_data is None:
        print("Warning: No data found in store (performance analysis).")
        return None
    try:
        # Assumes data was stored using df.to_json(orient='split')
        df = pd.read_json(stored_data, orient='split')
        if df.empty:
            print("Warning: Deserialized DataFrame is empty (performance analysis).")
            return df  # Return empty df, let downstream handle it
        # print(f"Data deserialized successfully for performance analysis. Shape: {df.shape}")
        return df
    except Exception as e:
        print(
            f"ERROR: Failed to deserialize data from store (performance analysis): {e}")
        traceback.print_exc()  # Print full error details for debugging
        return None


# --- Constants ---
# Defines the metrics available for ranking batsmen
BATSMAN_RANKING_STATS = {
    'total_runs': {'label': 'Total Runs Scored', 'higher_is_better': True},
    'average': {'label': 'Batting Average', 'higher_is_better': True},
    'strike_rate': {'label': 'Batting Strike Rate', 'higher_is_better': True},
    'matches_batted': {'label': 'Matches Batted In', 'higher_is_better': True},
    'innings': {'label': 'Innings Batted', 'higher_is_better': True},
    'fours': {'label': 'Total Fours', 'higher_is_better': True},
    'sixes': {'label': 'Total Sixes', 'higher_is_better': True},
}

# Defines the metrics available for ranking bowlers
BOWLER_RANKING_STATS = {
    'wickets': {'label': 'Wickets Taken', 'higher_is_better': True},
    # Lower is better
    'bowling_average': {'label': 'Bowling Average', 'higher_is_better': False},
    # Lower is better
    'economy': {'label': 'Economy Rate', 'higher_is_better': False},
    # Lower is better
    'bowling_strike_rate': {'label': 'Bowling Strike Rate', 'higher_is_better': False},
    'matches_bowled': {'label': 'Matches Bowled In', 'higher_is_better': True},
}

# Minimum thresholds to qualify for ranking (can be adjusted)
MIN_BALLS_FACED = 50        # Minimum balls faced for batsman ranking
MIN_INNINGS_BATTED = 5    # Minimum innings batted for batsman ranking
# Minimum balls bowled (e.g., 10 overs) for bowler ranking
MIN_BALLS_BOWLED = 60

# --- Layout Definition ---
# Defines the visual structure of the page using Dash Bootstrap Components
layout = dbc.Container([
    html.H1("Player Performance Ranking", className="text-center my-4"),

    # Control Row containing dropdowns, radio items, and slider
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                # Dropdown for selecting Player's Team (Country or Global)
                dbc.Col([
                    html.Label("Filter by Player's Team:",
                               className="fw-bold"),
                    dcc.Dropdown(
                        id='perf-country-dropdown',  # Unique ID for this component
                        # Initial option
                        options=[
                            {'label': 'Global (All Teams)', 'value': 'Global'}],
                        value='Global',  # Default selection
                        clearable=False  # User cannot clear the selection
                    )
                ], width=12, md=3, className="mb-3"),  # Responsive width

                # Radio items to select between Batsman or Bowler analysis
                dbc.Col([
                    html.Label("Select Analysis Type:", className="fw-bold"),
                    dbc.RadioItems(
                        id='perf-analysis-type-selector',  # Unique ID
                        options=[
                            {'label': 'Batsman', 'value': 'Batsman'},
                            {'label': 'Bowler', 'value': 'Bowler'},
                        ],
                        value='Batsman',  # Default selection
                        inline=True,  # Display options horizontally
                        inputClassName="me-1",  # Margin for input elements
                        labelStyle={'margin-right': '15px'}  # Style for labels
                    )
                ], width=12, md=3, className="mb-3"),

                # Dropdown for selecting the specific ranking metric (dynamic options)
                dbc.Col([
                    html.Label("Select Ranking Metric:", className="fw-bold"),
                    dcc.Dropdown(
                        id='perf-stat-dropdown',  # Unique ID
                        # Options are populated dynamically by a callback
                        options=[{'label': v['label'], 'value': k}
                                 for k, v in BATSMAN_RANKING_STATS.items()],
                        # Default value (first batsman stat)
                        value=list(BATSMAN_RANKING_STATS.keys())[0],
                        clearable=False
                    )
                ], width=12, md=3, className="mb-3"),

                # Slider to select the number of top/bottom players (N)
                dbc.Col([
                    html.Label("Number of Players (N):", className="fw-bold",
                               id='perf-slider-label'),  # Unique ID for label
                    dcc.Slider(
                        id='perf-n-slider',  # Unique ID
                        min=3, max=15, step=1, value=5,  # Slider parameters
                        # Marks displayed on the slider
                        marks={i: str(i) for i in range(3, 16, 2)},
                        # Tooltip showing value
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=12, md=3, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        "Submit",
                        id="perf-submit-button",
                        n_clicks=0,
                        color="primary",
                        className="w-100"  # Full width button inside column
                    ),
                    width=12, md=3, className="mb-4"  # Adjust width if needed
                ),
            ], justify="center"),
            # Row to display text about the ranking thresholds being applied
            dbc.Row(dbc.Col(
                # Text updated by callback
                html.Small(id='perf-threshold-info',
                           className="text-muted fst-italic")
            ))
        ]), className="shadow-sm mb-4"  # Styling for the card
    ),


    # Row for displaying the output charts
    dbc.Row([
        # Left column for the Top N players pie chart
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='perf-top-pie')))),  # Loading indicator wraps graph
                width=12, lg=6, className="mb-3"),  # Responsive width
        # Right column for the Bottom N players pie chart
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='perf-bottom-pie')))),  # Loading indicator wraps graph
                width=12, lg=6, className="mb-3"),  # Responsive width
    ]),

], fluid=True)  # Makes the container use the full width


# --- Callbacks ---
# Functions that link inputs and outputs to make the page interactive

# Callback 1: Populate Country Dropdown
@callback(
    # Output: updates the dropdown options
    Output('perf-country-dropdown', 'options'),
    # Input: triggered when data store changes
    Input('main-data-store', 'data'),
    prevent_initial_call=False  # Run this callback when the app starts
)
def update_perf_country_options(stored_data):
    """Populates the country/team dropdown based on available data in the main store."""
    print("Callback triggered: update_perf_country_options")
    df = deserialize_data(stored_data)  # Load data
    default_options = [{'label': 'Global (All Teams)', 'value': 'Global'}]

    # Check if data and required column exist
    if df is None or df.empty or 'player_team' not in df.columns:
        print("  - Data unavailable or 'player_team' column missing for country options.")
        return default_options  # Return only 'Global' if data is bad

    try:
        # Find unique, non-null team names and sort them alphabetically
        teams = sorted(df["player_team"].dropna().unique())
        if not teams:
            print("  - No unique player teams found.")
            return default_options

        # Create list of dictionaries for dropdown options
        country_options = [{'label': team, 'value': team} for team in teams]
        # print(f"  - Generated {len(country_options)} team options.")
        return default_options + country_options  # Combine 'Global' with team list

    except Exception as e:
        print(f"  - Error generating country options: {e}")
        traceback.print_exc()  # Log the error
        return default_options  # Return default on error


# Callback 2: Update Stat Dropdown Options based on Analysis Type
@callback(
    # Output 1: updates stat options
    Output('perf-stat-dropdown', 'options'),
    # Output 2: updates default selected stat
    Output('perf-stat-dropdown', 'value'),
    # Output 3: updates threshold info text
    Output('perf-threshold-info', 'children'),
    # Input: triggered by analysis type change
    Input('perf-analysis-type-selector', 'value')
)
def update_stat_dropdown_options(analysis_type):
    """Updates the ranking metric dropdown options and threshold info text
       when the analysis type (Batsman/Bowler) changes."""
    print(
        f"Callback triggered: update_stat_dropdown_options (Type: {analysis_type})")
    if analysis_type == 'Batsman':
        # Set options, default value, and threshold text for Batsman
        options = [{'label': v['label'], 'value': k}
                   for k, v in BATSMAN_RANKING_STATS.items()]
        default_value = list(BATSMAN_RANKING_STATS.keys())[0]
        threshold_text = f"(Batsmen must have faced at least {MIN_BALLS_FACED} balls and batted in at least {MIN_INNINGS_BATTED} innings to be ranked)"
    elif analysis_type == 'Bowler':
        # Set options, default value, and threshold text for Bowler
        options = [{'label': v['label'], 'value': k}
                   for k, v in BOWLER_RANKING_STATS.items()]
        default_value = list(BOWLER_RANKING_STATS.keys())[0]
        threshold_text = f"(Bowlers must have bowled at least {MIN_BALLS_BOWLED} balls to be ranked)"
    else:
        # Fallback for unexpected analysis type
        options = []
        default_value = None
        threshold_text = "Please select analysis type"

    print(
        f" - Setting stat options for {analysis_type}, default: {default_value}")
    return options, default_value, threshold_text


# Callback 3: Update Pie Charts based on selections
@callback(
    Output('perf-top-pie', 'figure'),       # Output 1: Top N Pie Chart
    Output('perf-bottom-pie', 'figure'),    # Output 2: Bottom N Pie Chart
    # Output 3: Slider label text (showing actual N)
    Output('perf-slider-label', 'children'),
    # Input 1: Selected Country/Global
    Input('perf-submit-button', 'n_clicks'),  # <-- CHANGED to Submit button!
    State('perf-country-dropdown', 'value'),
    State('perf-analysis-type-selector', 'value'),
    State('perf-stat-dropdown', 'value'),
    State('perf-n-slider', 'value'),
    State('main-data-store', 'data'),
    prevent_initial_call=True                # Input 5: Main data
)
def update_performance_charts(n_clicks, selected_country, analysis_type, selected_stat, n_players, stored_data):
    """
    Updates the top and bottom player ranking pie charts based on all user selections.
    Calculates stats, applies filters/thresholds, sorts players, and generates charts
    with an annotation showing the best/worst player in the displayed group.
    """
    print(
        f"Callback triggered: update_performance_charts (Country: {selected_country}, Type: {analysis_type}, Stat: {selected_stat}, N: {n_players})")
    # Default slider label
    slider_label_text = f"Number of Players (N={n_players}):"

    # --- Initial Validations ---
    # Ensure all necessary inputs have values
    if not all([selected_country, analysis_type, selected_stat, n_players is not None]):
        print("  - Preventing update: Missing inputs.")
        raise PreventUpdate  # Stop callback execution if inputs are missing

    # Load and validate the main data
    df_full = deserialize_data(stored_data)
    if df_full is None or df_full.empty:
        print("  - Preventing update: Main data not loaded or empty.")
        fig_empty = go.Figure(layout=go.Layout(
            title="Data not loaded", title_x=0.5))
        return fig_empty, fig_empty, slider_label_text  # Return empty charts

    # Determine required columns based on analysis type
    base_cols = ['name', 'player_team', 'match_id']
    if analysis_type == 'Batsman':
        required_cols = base_cols + \
            ['runs_scored', 'balls_faced', 'player_out',
                'fours_scored', 'sixes_scored']
        stat_config_dict = BATSMAN_RANKING_STATS
    elif analysis_type == 'Bowler':
        required_cols = base_cols + \
            ['wickets_taken', 'runs_conceded', 'balls_bowled']
        stat_config_dict = BOWLER_RANKING_STATS
    else:
        # Handle invalid analysis type selection
        print(f"  - Invalid analysis type selected: {analysis_type}")
        fig_err = go.Figure(layout=go.Layout(
            title="Invalid Analysis Type Selected", title_x=0.5))
        return fig_err, fig_err, slider_label_text + " Error"

    # Check if all required columns are present in the DataFrame
    if not all(col in df_full.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_full.columns]
        print(
            f"  - Preventing update: Missing required columns for {analysis_type}: {missing}")
        error_message = f"Error: Missing data columns ({', '.join(missing)})"
        fig_err = go.Figure(layout=go.Layout(title=error_message, title_x=0.5))
        return fig_err, fig_err, slider_label_text + " Error"

    # --- Main Calculation Block ---
    try:
        # --- 1. Filter Data by Country ---
        if selected_country == 'Global':
            df_filtered = df_full.copy()  # Use all data
            country_label = "Global"
        else:
            # Check if the selected team actually exists in the data
            if selected_country not in df_full['player_team'].unique():
                print(
                    f"  - Selected country '{selected_country}' not found in data.")
                fig_err = go.Figure(layout=go.Layout(
                    title=f"Team '{selected_country}' not found", title_x=0.5))
                return fig_err, fig_err, slider_label_text + " Error"
            # Filter the DataFrame for the selected team
            df_filtered = df_full[df_full['player_team']
                                  == selected_country].copy()
            country_label = selected_country

        # Check if filtering resulted in an empty DataFrame
        if df_filtered.empty:
            print(f"  - No data found for the selected scope: {country_label}")
            fig_empty = go.Figure(layout=go.Layout(
                title=f"No player data for {country_label}", title_x=0.5))
            return fig_empty, fig_empty, slider_label_text

        # --- 2. Aggregate Stats per Player (Conditional Logic) ---
        print(f"  - Aggregating {analysis_type} stats...")
        player_stats = pd.DataFrame()      # Initialize DataFrame for aggregated stats
        # Initialize DataFrame for players meeting thresholds
        qualified_players = pd.DataFrame()

        # Aggregate and calculate metrics based on whether Batsman or Bowler is selected
        if analysis_type == 'Batsman':
            # Calculate innings and matches batted
            batted_in_match = df_filtered[df_filtered['balls_faced'] > 0].groupby(
                ['name', 'match_id']).size().reset_index(name='batted_flag')
            innings_count = batted_in_match.groupby(
                'name')['match_id'].nunique().reset_index(name='innings')
            matches_batted_count = batted_in_match.groupby(
                'name')['match_id'].nunique().reset_index(name='matches_batted')

            # Aggregate basic batting stats
            player_stats = df_filtered.groupby('name', as_index=False).agg(
                total_runs=('runs_scored', 'sum'), balls_faced=('balls_faced', 'sum'),
                outs=('player_out', 'sum'), fours=('fours_scored', 'sum'), sixes=('sixes_scored', 'sum')
            )
            # Merge calculated innings/matches
            player_stats = pd.merge(
                player_stats, innings_count, on='name', how='left')
            player_stats = pd.merge(
                player_stats, matches_batted_count, on='name', how='left')
            player_stats['innings'] = player_stats['innings'].fillna(
                0).astype(int)
            player_stats['matches_batted'] = player_stats['matches_batted'].fillna(
                0).astype(int)

            # Calculate derived batting metrics (average, strike rate)
            player_stats['average'] = player_stats.apply(
                lambda r: r['total_runs'] / r['outs'] if r['outs'] > 0 else np.nan, axis=1)
            player_stats['strike_rate'] = player_stats.apply(lambda r: (
                r['total_runs'] / r['balls_faced']) * 100 if r['balls_faced'] > 0 else 0.0, axis=1)

            # Apply batting thresholds to get qualified players
            qualified_players = player_stats[
                (player_stats['balls_faced'] >= MIN_BALLS_FACED) &
                (player_stats['innings'] >= MIN_INNINGS_BATTED)
            ].copy()

        elif analysis_type == 'Bowler':
            # Calculate matches bowled
            bowled_in_match = df_filtered[df_filtered['balls_bowled'] > 0].groupby(
                ['name', 'match_id']).size().reset_index(name='bowled_flag')
            matches_bowled_count = bowled_in_match.groupby(
                'name')['match_id'].nunique().reset_index(name='matches_bowled')

            # Aggregate basic bowling stats
            player_stats = df_filtered.groupby('name', as_index=False).agg(
                wickets=('wickets_taken', 'sum'), runs_conceded=('runs_conceded', 'sum'),
                balls_bowled=('balls_bowled', 'sum')
            )
            # Merge calculated matches bowled
            player_stats = pd.merge(
                player_stats, matches_bowled_count, on='name', how='left')
            player_stats['matches_bowled'] = player_stats['matches_bowled'].fillna(
                0).astype(int)

            # Calculate derived bowling metrics (average, economy, strike rate)
            player_stats['bowling_average'] = player_stats.apply(
                lambda r: r['runs_conceded'] / r['wickets'] if r['wickets'] > 0 else np.nan, axis=1)
            player_stats['economy'] = player_stats.apply(lambda r: (
                r['runs_conceded'] / r['balls_bowled']) * 6 if r['balls_bowled'] > 0 else np.nan, axis=1)
            player_stats['bowling_strike_rate'] = player_stats.apply(
                lambda r: r['balls_bowled'] / r['wickets'] if r['wickets'] > 0 else np.nan, axis=1)

            # Apply bowling thresholds to get qualified players
            qualified_players = player_stats[
                player_stats['balls_bowled'] >= MIN_BALLS_BOWLED
            ].copy()

        # Check if any players qualify after applying thresholds
        if qualified_players.empty:
            print(
                f"  - No players qualify for ranking in {country_label} for {analysis_type} based on thresholds.")
            fig_empty = go.Figure(layout=go.Layout(
                title=f"No players qualify for ranking ({analysis_type})", title_x=0.5))
            return fig_empty, fig_empty, slider_label_text

        # --- 3. Prepare for Ranking ---
        # Validate the selected statistic
        if selected_stat not in stat_config_dict or selected_stat not in qualified_players.columns:
            print(
                f"  - Invalid or missing ranking statistic: {selected_stat} for {analysis_type}")
            fig_err = go.Figure(layout=go.Layout(
                title=f"Invalid statistic selected: {selected_stat}", title_x=0.5))
            return fig_err, fig_err, slider_label_text + " Error"

        # Get configuration for the selected statistic (label, higher is better)
        stat_config = stat_config_dict[selected_stat]
        stat_label = stat_config['label']
        higher_is_better = stat_config['higher_is_better']
        ranking_col = selected_stat  # Column name to rank by

        # Remove players with NaN values for the specific ranking statistic
        qualified_players.dropna(subset=[ranking_col], inplace=True)

        # Check if any players remain after dropping NaN values
        if qualified_players.empty:
            print(
                f"  - No players with valid '{stat_label}' data after dropping NaNs.")
            fig_empty = go.Figure(layout=go.Layout(
                title=f"No valid '{stat_label}' data for ranking", title_x=0.5))
            return fig_empty, fig_empty, slider_label_text

        # --- 4. Sort and Select Top/Bottom N ---
        # Sort players based on the ranking column (always ascending initially)
        qualified_players_sorted = qualified_players.sort_values(
            by=ranking_col, ascending=True, na_position='last')

        # Determine the actual number of players to show (cannot exceed available players)
        actual_n = min(n_players, len(qualified_players_sorted))
        # Update slider label
        slider_label_text = f"Number of Players (N={n_players}, Showing {actual_n}):"

        # Select the top N and bottom N players based on the 'higher_is_better' flag
        if higher_is_better:
            # If higher is better, top N are the tail (highest values), bottom N are the head (lowest values)
            top_n = qualified_players_sorted.tail(actual_n)
            bottom_n = qualified_players_sorted.head(actual_n)
        else:
            # If lower is better, top N are the head (lowest values), bottom N are the tail (highest values)
            top_n = qualified_players_sorted.head(actual_n)
            bottom_n = qualified_players_sorted.tail(actual_n)

        # --- 5. Create Pie Charts with Best/Worst Annotation ---
        chart_colors_top = px.colors.sequential.Blues_r    # Color scheme for top chart
        chart_colors_bottom = px.colors.sequential.Reds    # Color scheme for bottom chart

        # --- Helper function for formatting single stat values ---
        def format_stat_value(value, stat_key, analysis_type):
            """Formats a single stat value based on its type for display."""
            if pd.isna(value):
                return "N/A"  # Handle NaN values
            # Define which stats should be formatted with decimals
            decimal_stats_batsman = ['average', 'strike_rate']
            decimal_stats_bowler = ['bowling_average',
                                    'economy', 'bowling_strike_rate']
            is_decimal = (analysis_type == 'Batsman' and stat_key in decimal_stats_batsman) or \
                         (analysis_type == 'Bowler' and stat_key in decimal_stats_bowler)
            if is_decimal:
                return f"{value:.2f}"  # Format as float with 2 decimals
            else:
                try:
                    return f"{int(value):,}"  # Format as integer with commas
                except (ValueError, TypeError):
                    return f"{value:.2f}"  # Fallback formatting

        # --- Top N Pie Chart ---
        fig_top = go.Figure()  # Initialize an empty figure
        if not top_n.empty:
            # Find the single best player WITHIN the displayed top_n group
            if higher_is_better:
                # Find row with max value
                best_player_row = top_n.loc[top_n[ranking_col].idxmax()]
            else:
                # Find row with min value
                best_player_row = top_n.loc[top_n[ranking_col].idxmin()]
            # Get best player's name
            best_name = best_player_row['name']
            # Get best player's stat value
            best_value = best_player_row[ranking_col]
            formatted_best_value = format_stat_value(
                best_value, selected_stat, analysis_type)  # Format value

            # Create the pie chart using Plotly Express
            fig_top = px.pie(top_n, names='name', values=ranking_col,
                             title=f"Top {actual_n} {analysis_type}s by {stat_label} ({country_label})",
                             hole=0.4,  # Create a hole for the annotation
                             color_discrete_sequence=chart_colors_top)
            # Update trace properties (text on slices, hover info)
            fig_top.update_traces(textposition='inside', textinfo='percent+label',
                                  hovertemplate=f"<b>%{{label}}</b><br>{stat_label}: %{{value:.2f}}<extra></extra>")
            # Add annotation in the center showing the best player
            fig_top.add_annotation(
                # Text content
                text=f"<b>Best:</b><br>{best_name}<br>{formatted_best_value}",
                align='center', showarrow=False,  # Styling
                xref='paper', yref='paper', x=0.5, y=0.5,  # Positioning
                font=dict(size=14)  # Font size
            )
        else:
            # If top_n DataFrame is empty, display a message
            fig_top = go.Figure(layout=go.Layout(
                title=f"No Top {actual_n} {analysis_type} Data Available", title_x=0.5))

        # --- Bottom N Pie Chart ---
        fig_bottom = go.Figure()  # Initialize an empty figure
        if not bottom_n.empty:
            # Find the single worst player WITHIN the displayed bottom_n group
            if higher_is_better:  # Worst has the minimum value
                worst_player_row = bottom_n.loc[bottom_n[ranking_col].idxmin()]
            else:  # Worst has the maximum value
                worst_player_row = bottom_n.loc[bottom_n[ranking_col].idxmax()]
            # Get worst player's name
            worst_name = worst_player_row['name']
            # Get worst player's stat value
            worst_value = worst_player_row[ranking_col]
            formatted_worst_value = format_stat_value(
                worst_value, selected_stat, analysis_type)  # Format value

            # Create the pie chart
            fig_bottom = px.pie(bottom_n, names='name', values=ranking_col,
                                title=f"Bottom {actual_n} {analysis_type}s by {stat_label} ({country_label})",
                                hole=0.4, color_discrete_sequence=chart_colors_bottom)
            # Update trace properties
            fig_bottom.update_traces(textposition='inside', textinfo='percent+label',
                                     hovertemplate=f"<b>%{{label}}</b><br>{stat_label}: %{{value:.2f}}<extra></extra>")
            # Add annotation in the center showing the worst player
            fig_bottom.add_annotation(
                text=f"<b>Worst:</b><br>{worst_name}<br>{formatted_worst_value}",
                align='center', showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5,
                font=dict(size=14)
            )
        else:
            # If bottom_n DataFrame is empty, display a message
            fig_bottom = go.Figure(layout=go.Layout(
                title=f"No Bottom {actual_n} {analysis_type} Data Available", title_x=0.5))

        # Common layout updates for both charts (like hiding legend, setting margins, centering title)
        for fig in [fig_top, fig_bottom]:
            if fig.data:  # Apply only if the figure has data/annotations
                fig.update_layout(showlegend=False, margin=dict(
                    t=60, b=20, l=20, r=20), title_x=0.5)

        print(
            f"  - Charts generated successfully for Top/Bottom {actual_n} {analysis_type}s.")
        # Return the generated figures and updated slider label
        return fig_top, fig_bottom, slider_label_text

    # --- Error Handling Block ---
    except Exception as e:
        print(f"ERROR during {analysis_type} chart generation: {e}")
        traceback.print_exc()  # Print detailed error traceback
        error_message = "An error occurred during analysis"
        fig_err = go.Figure(layout=go.Layout(title=error_message, title_x=0.5))
        # Ensure slider label indicates an error state
        slider_label_text_err = f"Number of Players (N={n_players}): Error"
        # Return error figures and updated label
        return fig_err, fig_err, slider_label_text_err
