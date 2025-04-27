# pages/betting_analyzer.py

import dash
# Make sure ALL is imported if needed, callback definitely needed
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import traceback
import json  # Needed for deserializing data from store

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Register Page ---
dash.register_page(__name__, path='/betting-analyzer',
                   name='Pre-Betting Analyzer')  # Choose path and name

# --- Constants ---
# Keep constants used by the layout and callbacks here
BOWLING_TYPE_COLORS = {
    "Spin": "#EF553B",      # Reddish-Orange
    "Right Fast": "#636EFA",  # Blue
    "Left Fast": "#00CC96"   # Green
}
ALL_BOWLING_TYPES = "--- ALL ---"
ALL_OPTION = "--- ALL ---"
# Not used currently, but keep if planned
AVERAGE_PLAYER_NAME = "Average Player"

# --- Helper Functions specific to this page's logic (if any) ---
# Data loading will be handled by multi_page_app.py, but keep calculation logic if complex
# Note: The original load_and_process_data is moved to multi_page_app.py

# --- App Layout Definition ---
# Assign the layout to a variable named 'layout'
layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("🏏 Pre-Betting Analyzer (Aggression vs Risk + Odds)"),
            width=12), className="mb-4 mt-2"),  # Renamed slightly

    dbc.Row([
        # --- Control Column ---
        dbc.Col([
            html.H4("Select Mode"),
            dcc.RadioItems(id='ba-mode-selector',  # Prefix IDs to avoid clashes
                           options=[{'label': 'Single Player', 'value': 'single'},
                                    {'label': 'Compare Players', 'value': 'compare'}],
                           value='compare', labelStyle={'display': 'inline-block', 'margin-right': '15px'}),
            html.Hr(),

            # --- Player 1 Section ---
            dbc.Card([
                dbc.CardHeader("Player 1"),
                dbc.CardBody([
                    # Dropdown options will be populated by callback
                    dcc.Dropdown(id='ba-player-dropdown-1', options=[], value=None,
                                 clearable=False, placeholder="Select Player 1..."),
                    html.Div([
                        html.Label("Odds (e.g., 10 for 10:1):",
                                   className="mt-2"),
                        dcc.Input(id='ba-player-1-odds', type='number', min=1,
                                  step=0.1, value=1.0, className="form-control")
                    ], style={'marginTop': '10px'})
                ])
            ], className="mb-3"),

            # --- Player 2 Section (Conditional) ---
            html.Div(id='ba-player-2-section', children=[
                dbc.Card([
                     dbc.CardHeader("Player 2"),
                     dbc.CardBody([
                        # Dropdown options will be populated by callback
                        dcc.Dropdown(id='ba-player-dropdown-2', options=[], value=None,
                                     clearable=True, placeholder="Select Player 2..."),
                        html.Div([
                            html.Label("Odds (e.g., 10 for 10:1):",
                                       className="mt-2"),
                            dcc.Input(id='ba-player-2-odds', type='number', min=1,
                                      step=0.1, value=1.0, className="form-control")
                        ], style={'marginTop': '10px'})
                     ])
                     ], className="mb-3"),
                # Initial style, controlled by callback
            ], style={'display': 'block'}
            ),

            # --- Filters ---
            html.H4("Filters"),
            html.Label("Match Type:"),
            # Dropdown options will be populated by callback
            dcc.Dropdown(id='ba-match-type-filter', options=[],
                         value=ALL_OPTION, clearable=False),
            html.Label("Bowling Type:", className="mt-2"),
            dcc.Dropdown(id='ba-bowling-type-filter',
                         options=[{'label': ALL_BOWLING_TYPES, 'value': ALL_BOWLING_TYPES}] + [
                             {'label': bt, 'value': bt} for bt in BOWLING_TYPE_COLORS.keys()],
                         value=ALL_BOWLING_TYPES, clearable=False),
            dbc.Button("Apply Filters", id='ba-apply-filters',
                       n_clicks=0, color='primary', className="mt-2"),

        ], width=12, md=4, lg=3),

        # --- Plot & Recommendation Column ---
        dbc.Col([
            # Dynamic title
            html.H4(id="ba-plot-title", children="Select Options"),
            dcc.Loading(id="ba-loading-scatter", type="circle",
                        children=dcc.Graph(id='ba-aggression-risk-scatter')),
            html.Hr(),
            dbc.Card(id="ba-recommendation-card", children=[  # Area for recommendation
                dbc.CardHeader("Betting Suggestion"),
                dbc.CardBody(id="ba-recommendation-text",
                             children="Select players and odds to see a suggestion.")
            ])
        ], width=12, md=8, lg=9),
    ], className="mb-4"),

    html.Hr(),
    dbc.Row(dbc.Col(html.P("Aggression (Runs/100 Balls) vs Risk (Outs/100 Balls). Color=Bowling Type, Symbol=Player, Size=Balls Faced.", className="text-muted small")))

], fluid=True)


# --- Callbacks ---

# Callback 0: Populate Dropdowns from Store Data
@callback(
    Output('ba-player-dropdown-1', 'options'),
    Output('ba-player-dropdown-2', 'options'),
    Output('ba-match-type-filter', 'options'),
    Output('ba-player-dropdown-1', 'value'),  # Set initial player 1 value
    # Triggered when data arrives from the store
    Input('betting-analyzer-data-store', 'data')
)
def update_dropdown_options(betting_data_json):
    if betting_data_json is None:
        print("Betting Analyzer: No data from store yet for dropdowns.")
        return [], [], [], None  # Return empty options and no initial selection

    try:
        df_processed_data = pd.read_json(betting_data_json, orient='split')
        print(
            f"Betting Analyzer: Loaded data from store for dropdowns. Shape: {df_processed_data.shape}")

        if df_processed_data.empty or 'name' not in df_processed_data.columns:
            print("Betting Analyzer: Data from store is empty or missing 'name' column.")
            return [], [], [], None

        player_options_list = [{'label': p, 'value': p}
                               for p in sorted(df_processed_data['name'].unique())]

        match_type_opts = [{'label': ALL_OPTION, 'value': ALL_OPTION}]
        if 'match_type' in df_processed_data.columns:
            match_type_opts.extend([{'label': mt, 'value': mt} for mt in sorted(
                df_processed_data['match_type'].unique())])
        else:
            print(
                "Betting Analyzer: Warning - 'match_type' column missing in store data.")

        # Set initial value for Player 1 dropdown if options are available
        initial_player_1 = player_options_list[0]['value'] if player_options_list else None

        print("Betting Analyzer: Dropdown options updated.")
        return player_options_list, player_options_list, match_type_opts, initial_player_1

    except Exception as e:
        print(f"ERROR in update_dropdown_options (Betting Analyzer): {e}")
        traceback.print_exc()
        return [], [], [], None  # Return empty on error


# Callback 1: Show/Hide Player 2 Section
@callback(
    Output('ba-player-2-section', 'style'),
    Input('ba-mode-selector', 'value')  # Use prefixed ID
)
def toggle_player_2_visibility(selected_mode):
    return {'display': 'block'} if selected_mode == 'compare' else {'display': 'none'}

# Callback 2: Update Plot and Recommendation


@callback(
    Output('ba-aggression-risk-scatter', 'figure'),
    Output('ba-plot-title', 'children'),
    Output('ba-recommendation-text', 'children'),
    Input('ba-apply-filters', 'n_clicks'),  # Triggered by button click
    State('betting-analyzer-data-store', 'data'),  # Input data from the store
    State('ba-mode-selector', 'value'),
    State('ba-player-dropdown-1', 'value'),
    State('ba-player-dropdown-2', 'value'),
    State('ba-player-1-odds', 'value'),
    State('ba-player-2-odds', 'value'),
    State('ba-match-type-filter', 'value'),
    State('ba-bowling-type-filter', 'value')
    # Removed State('betting-analyzer-data-store', 'data') - use Input if always needed
)
def update_scatter_plot_and_recommendation(n_clicks, betting_data_json, mode, selected_player_1, selected_player_2,
                                           odds_p1, odds_p2,
                                           selected_match_type, selected_bowling_type):
    if not n_clicks:
        raise PreventUpdate

    print(
        f"Callback Betting Analyzer: Mode='{mode}', P1='{selected_player_1}'({odds_p1}), P2='{selected_player_2}'({odds_p2}), Match='{selected_match_type}', Bowl='{selected_bowling_type}'")

    # --- Get Data from Store ---
    if betting_data_json is None:
        print("Betting Analyzer: No data received from store yet.")
        return go.Figure(layout=go.Layout(title="Waiting for data...")), "Loading...", "Waiting for data..."
    try:
        df_processed_data = pd.read_json(betting_data_json, orient='split')
        if df_processed_data is None or df_processed_data.empty:
            print(
                "Betting Analyzer: Failed to load/deserialize data from store or data is empty.")
            return go.Figure(layout=go.Layout(title="Error: Data could not be loaded")), "Data Error", "Failed to load data."
        print(
            f"Betting Analyzer: Data successfully loaded from store. Shape: {df_processed_data.shape}")
    except Exception as e:
        print(f"ERROR deserializing betting data: {e}")
        traceback.print_exc()
        return go.Figure(layout=go.Layout(title="Error: Data processing failed")), "Data Error", "Error processing data from store."

    # --- Basic Validation & Player Selection ---
    selected_players = []
    recommendation = "Enter selections and valid odds."
    if mode == 'single':
        if selected_player_1:
            selected_players = [selected_player_1]
        else:
            # PreventUpdate might be better if dropdowns are loading
            # raise PreventUpdate
            return go.Figure(layout=go.Layout(title="Please select Player 1")), "Select Player", recommendation
    elif mode == 'compare':
        selected_players = [p for p in [
            selected_player_1, selected_player_2] if p is not None]
        if not selected_players:
            # raise PreventUpdate
            return go.Figure(layout=go.Layout(title="Please select at least one player")), "Select Player(s)", recommendation
        if len(selected_players) == 1 and selected_player_1 and not selected_player_2:
            odds_p2 = None  # Ensure odds p2 is None if p2 is deselected
    else:
        return go.Figure(layout=go.Layout(title="Invalid Mode Selected")), "Error", recommendation

    # Validate odds
    odds = {}
    try:
        if selected_player_1:
            odds[selected_player_1] = float(
                odds_p1) if odds_p1 is not None and float(odds_p1) >= 1 else 1.0
        if mode == 'compare' and selected_player_2:
            odds[selected_player_2] = float(
                odds_p2) if odds_p2 is not None and float(odds_p2) >= 1 else 1.0
    except (ValueError, TypeError):
        # Check if figure exists before updating layout
        fig_odds_error = go.Figure(
            layout=go.Layout(title="Invalid Odds Input"))
        return fig_odds_error, "Invalid Odds", "Odds must be numbers >= 1."

    # --- Filter Data for Plotting (using df_processed_data from store) ---
    # (The rest of the filtering, plotting, and recommendation logic remains largely the same
    # as in your original 08_pre_betting_analyzer.py, just ensure it uses df_processed_data)

    player_base_data = df_processed_data[df_processed_data['name'].isin(
        selected_players)].copy()
    data_for_plotting = pd.DataFrame()  # Initialize as empty

    # --- Aggregation Logic (Simplified) ---
    if selected_match_type == ALL_OPTION:
        if not player_base_data.empty:
            agg_cols = ['Total Runs', 'Total Balls', 'Total Outs']
            grouping = ['name', 'Bowling Type']
            # Check if necessary columns exist
            if all(col in player_base_data.columns for col in agg_cols + grouping):
                # Direct aggregation
                aggregated_data = player_base_data.groupby(
                    grouping, as_index=False)[agg_cols].sum()

                # Check if aggregation produced results
                if not aggregated_data.empty and 'Total Balls' in aggregated_data.columns:
                    # Calculate rates only if cols exist after aggregation
                    if 'Total Runs' in aggregated_data.columns:
                        aggregated_data['run_rate'] = np.where(aggregated_data['Total Balls'] > 0, (
                            aggregated_data['Total Runs'] * 100) / aggregated_data['Total Balls'], 0)
                    if 'Total Outs' in aggregated_data.columns:
                        aggregated_data['out_rate'] = np.where(aggregated_data['Total Balls'] > 0, (
                            aggregated_data['Total Outs'] * 100) / aggregated_data['Total Balls'], 0)

                    # Filter out rows with zero balls after rate calculation
                    data_for_plotting = aggregated_data[aggregated_data['Total Balls'] > 0].copy(
                    )
                else:
                    print(
                        "Betting Analyzer: Aggregation for ALL match types resulted in empty data or missing 'Total Balls'.")
            else:
                print(
                    f"Betting Analyzer: ERROR - Missing columns for ALL aggregation. Need: {agg_cols + grouping}, Have: {player_base_data.columns.tolist()}")
        else:
            print(
                "Betting Analyzer: No base data for selected players (ALL match types).")

    else:  # Specific Match Type Selected
        if 'match_type' in player_base_data.columns:
            plot_data_filtered = player_base_data[player_base_data['match_type'] == selected_match_type].copy(
            )
            # Check required columns exist before filtering by balls
            required_plotting_cols = ['Total Balls', 'Total Runs',
                                      'Total Outs', 'Bowling Type', 'name', 'run_rate', 'out_rate']
            if all(col in plot_data_filtered.columns for col in required_plotting_cols):
                data_for_plotting = plot_data_filtered[plot_data_filtered['Total Balls'] > 0].copy(
                )
                print(
                    f"Betting Analyzer: Filtered for specific match type: {selected_match_type}. Shape: {data_for_plotting.shape}")
            else:
                print(
                    f"Betting Analyzer: Warning - Missing columns for plotting specific match type: {selected_match_type}. Have: {plot_data_filtered.columns.tolist()}")
        else:
            print("Betting Analyzer: Warning - 'match_type' column missing in data.")

    # Apply Bowling Type Filter
    if not data_for_plotting.empty and selected_bowling_type != ALL_BOWLING_TYPES:
        if 'Bowling Type' in data_for_plotting.columns:
            data_for_plotting = data_for_plotting[data_for_plotting['Bowling Type']
                                                  == selected_bowling_type]
        else:
            print(
                "Betting Analyzer: Warning - 'Bowling Type' column missing for filtering.")
            data_for_plotting = pd.DataFrame()  # Clear data if column is missing

    # --- Generate Plot ---
    plot_title_text = "Plot"
    heading_title = "Analysis"
    fig = go.Figure()  # Initialize empty figure

    if data_for_plotting.empty:
        player_names = " and ".join(
            selected_players) if selected_players else "selected players"
        match_desc = f" for {selected_match_type}" if selected_match_type != ALL_OPTION else " across all match types"
        bowl_desc = f" vs {selected_bowling_type}" if selected_bowling_type != ALL_BOWLING_TYPES else ""
        print(
            f"Betting Analyzer: No plottable data found for {player_names}{match_desc}{bowl_desc}.")
        fig = go.Figure(layout=go.Layout(title=f"No plottable data available"))
        heading_title = "No Data Available"
    else:
        # Ensure necessary columns exist before plotting
        required_plot_cols = ['out_rate', 'run_rate', 'Total Balls',
                              'Bowling Type', 'name', 'Total Runs', 'Total Outs']
        if not all(col in data_for_plotting.columns for col in required_plot_cols):
            print(
                f"Betting Analyzer: ERROR - Plotting columns missing. Have: {data_for_plotting.columns.tolist()}, Need: {required_plot_cols}")
            fig = go.Figure(layout=go.Layout(
                title="Plotting Error: Missing Data Columns"))
            heading_title = "Plotting Error"
        else:
            title_players = " vs ".join(selected_players) if len(
                selected_players) > 1 else (selected_players[0] if selected_players else "N/A")
            title_match = f"({selected_match_type})" if selected_match_type != ALL_OPTION else "(All Match Types)"
            title_bowling = f" vs {selected_bowling_type}" if selected_bowling_type != ALL_BOWLING_TYPES else ""
            plot_title_text = f"Aggression vs Risk: {title_players} {title_match}{title_bowling}"
            heading_title = f"Performance Scatter Plot {title_match}{title_bowling}"

            try:
                fig = px.scatter(
                    data_for_plotting, x='out_rate', y='run_rate', size='Total Balls',
                    color='Bowling Type', symbol='name', hover_name='name',
                    hover_data={'Bowling Type': True, 'name': False, 'Total Balls': True,
                                'Total Runs': True, 'Total Outs': True, 'run_rate': ':.2f', 'out_rate': ':.2f'},
                    color_discrete_map=BOWLING_TYPE_COLORS, title=plot_title_text,
                    labels={'run_rate': '<b>Aggression</b> (Runs/100 Balls) ↑', 'out_rate': '<b>Risk</b> (Outs/100 Balls) →', 'Total Balls': 'Balls Faced', 'name': 'Player'}, size_max=50
                )
                fig.update_layout(xaxis_title_font_size=14, yaxis_title_font_size=14, legend=dict(title="Legend", traceorder='grouped'), xaxis=dict(
                    gridcolor='rgba(230, 230, 230, 0.5)', zeroline=True), yaxis=dict(gridcolor='rgba(230, 230, 230, 0.5)', zeroline=True), margin=dict(l=60, r=30, t=50, b=60), hovermode='closest')
                fig.update_traces(marker=dict(
                    sizemin=5, line_width=1), selector=dict(mode='markers'))
                print(
                    f"Betting Analyzer: Generated plot for {', '.join(selected_players)}")
            except Exception as plot_err:
                print(
                    f"Betting Analyzer: ERROR during plot generation: {plot_err}")
                traceback.print_exc()
                fig = go.Figure(layout=go.Layout(
                    title="Plot Generation Error"))
                heading_title = "Plotting Error"

    # --- Calculate Recommendation ---
    # Use the initially filtered data (before aggregation for plotting)
    recommendation_data = player_base_data.copy()

    # Apply filters relevant for recommendation scoring
    if selected_match_type != ALL_OPTION:
        if 'match_type' in recommendation_data.columns:
            recommendation_data = recommendation_data[recommendation_data['match_type']
                                                      == selected_match_type]
        else:
            recommendation_data = pd.DataFrame()  # Clear if column missing

    if not recommendation_data.empty and selected_bowling_type != ALL_BOWLING_TYPES:
        if 'Bowling Type' in recommendation_data.columns:
            recommendation_data = recommendation_data[recommendation_data['Bowling Type']
                                                      == selected_bowling_type]
        else:
            recommendation_data = pd.DataFrame()  # Clear if column missing

    player_scores = {}
    if not recommendation_data.empty:
        rec_agg_cols = ['Total Runs', 'Total Balls', 'Total Outs']
        # Aggregate filtered data by player name only for overall scoring in the selected context
        if all(col in recommendation_data.columns for col in rec_agg_cols + ['name']):
            rec_aggregated = recommendation_data.groupby(
                'name', as_index=False)[rec_agg_cols].sum()

            if not rec_aggregated.empty and 'Total Balls' in rec_aggregated.columns:
                # Ensure non-zero balls
                rec_aggregated = rec_aggregated[rec_aggregated['Total Balls'] > 0]
                if not rec_aggregated.empty:
                    # Calculate rates for scoring
                    if 'Total Runs' in rec_aggregated.columns:
                        rec_aggregated['run_rate'] = (
                            rec_aggregated['Total Runs'] * 100) / rec_aggregated['Total Balls']
                    if 'Total Outs' in rec_aggregated.columns:
                        rec_aggregated['out_rate'] = (
                            rec_aggregated['Total Outs'] * 100) / rec_aggregated['Total Balls']

                    # Check if rates were calculated successfully
                    if 'run_rate' in rec_aggregated.columns and 'out_rate' in rec_aggregated.columns:
                        k = 20
                        MIN_RUN_RATE = 1
                        MAX_OUT_RATE = 50  # Scoring parameters
                        for _, row in rec_aggregated.iterrows():
                            player_name = row['name']
                            rr = row['run_rate']
                            ort = row['out_rate']
                            balls = row['Total Balls']
                            # Use validated odds from earlier
                            player_odds = odds.get(player_name)
                            if player_odds is None:
                                print(
                                    f"Betting Analyzer: Warning - Odds missing for {player_name} during scoring.")
                                continue  # Skip scoring if odds are missing

                            if rr < MIN_RUN_RATE or ort > MAX_OUT_RATE:
                                score = -np.inf
                                adj_score = -np.inf
                                print(
                                    f"Betting Analyzer: {player_name} failed base checks (RR:{rr:.1f}, OR:{ort:.1f})")
                            else:
                                score = rr - k * ort
                                adj_score = score * player_odds
                                print(
                                    f"Betting Analyzer: {player_name}: RR={rr:.1f}, OR={ort:.1f}, Balls={balls}, Score={score:.1f}, Odds={player_odds:.1f}, AdjScore={adj_score:.1f}")
                            player_scores[player_name] = {
                                'score': score, 'adj_score': adj_score, 'rr': rr, 'ort': ort}
                    else:
                        print(
                            "Betting Analyzer: Could not calculate rates for recommendation.")
                else:
                    print(
                        "Betting Analyzer: No players with >0 balls after filtering for recommendation.")
            else:
                print(
                    "Betting Analyzer: Aggregated recommendation data empty or missing 'Total Balls'.")
        else:
            print(
                f"Betting Analyzer: Missing columns for recommendation aggregation. Have: {recommendation_data.columns.tolist()}")

    # --- Generate Recommendation Text ---
    # (This logic remains the same, using player_scores)
    if not player_scores:
        recommendation = "No valid player data for recommendation in this context (check filters and player stats)."
    elif mode == 'single' and selected_player_1:
        p1 = selected_player_1
        score_p1 = player_scores.get(p1)
        if score_p1:
            p1_odds = odds.get(p1, "N/A")
            if score_p1['adj_score'] > 0:
                recommendation = f"Consider Bet on {p1}: Decent stats (RR {score_p1['rr']:.1f}, Risk {score_p1['ort']:.1f}) and odds ({p1_odds:.1f}) give positive adjusted score ({score_p1['adj_score']:.1f})."
            elif score_p1['score'] > -np.inf:
                recommendation = f"Avoid Bet on {p1}: Stats OK, but odds ({p1_odds:.1f}) aren't high enough (AdjScore: {score_p1['adj_score']:.1f})."
            else:
                recommendation = f"Avoid Bet on {p1}: Performance (RR {score_p1['rr']:.1f}, Risk {score_p1['ort']:.1f}) likely too poor."
        else:
            recommendation = f"No scoring data for {p1} in this context."
    elif mode == 'compare' and selected_player_1 and selected_player_2:
        p1 = selected_player_1
        p2 = selected_player_2
        score_p1 = player_scores.get(p1)
        score_p2 = player_scores.get(p2)
        p1_odds = odds.get(p1, "N/A")
        p2_odds = odds.get(p2, "N/A")

        if score_p1 and score_p2:
            if score_p1['adj_score'] > score_p2['adj_score'] and score_p1['adj_score'] > 0:
                # Handle division by zero
                ratio = score_p1['adj_score'] / \
                    score_p2['adj_score'] if score_p2['adj_score'] != 0 else "inf"
                recommendation = f"Suggest Bet on {p1}: Higher AdjScore ({score_p1['adj_score']:.1f} @{p1_odds:.1f}) vs {p2} ({score_p2['adj_score']:.1f} @{p2_odds:.1f}). (Ratio: {ratio if isinstance(ratio, str) else f'{ratio:.2f}x'})."
                if score_p1['score'] < score_p2['score']:
                    recommendation += f" Note: {p2} raw perf better."
            elif score_p2['adj_score'] > score_p1['adj_score'] and score_p2['adj_score'] > 0:
                # Handle division by zero
                ratio = score_p2['adj_score'] / \
                    score_p1['adj_score'] if score_p1['adj_score'] != 0 else "inf"
                recommendation = f"Suggest Bet on {p2}: Higher AdjScore ({score_p2['adj_score']:.1f} @{p2_odds:.1f}) vs {p1} ({score_p1['adj_score']:.1f} @{p1_odds:.1f}). (Ratio: {ratio if isinstance(ratio, str) else f'{ratio:.2f}x'})."
                if score_p2['score'] < score_p1['score']:
                    recommendation += f" Note: {p1} raw perf better."
            elif score_p1['adj_score'] <= 0 and score_p2['adj_score'] <= 0:
                recommendation = f"Avoid Bet: Neither {p1} ({score_p1['adj_score']:.1f}) nor {p2} ({score_p2['adj_score']:.1f}) offers positive adjusted return at these odds."
            else:
                recommendation = f"No Clear Edge: Adjusted Scores close or only one positive ({p1}: {score_p1['adj_score']:.1f}, {p2}: {score_p2['adj_score']:.1f})."
        elif score_p1:  # Only P1 has data
            if score_p1['adj_score'] > 0:
                recommendation = f"Consider Bet on {p1}: Only player with valid data, positive AdjScore ({score_p1['adj_score']:.1f} @{p1_odds:.1f}). No data for {p2} in context."
            else:
                recommendation = f"Avoid Bet on {p1}: Only player, AdjScore not positive ({score_p1['adj_score']:.1f} @{p1_odds:.1f}). No data for {p2}."
        elif score_p2:  # Only P2 has data
            if score_p2['adj_score'] > 0:
                recommendation = f"Consider Bet on {p2}: Only player with valid data, positive AdjScore ({score_p2['adj_score']:.1f} @{p2_odds:.1f}). No data for {p1} in context."
            else:
                recommendation = f"Avoid Bet on {p2}: Only player, AdjScore not positive ({score_p2['adj_score']:.1f} @{p2_odds:.1f}). No data for {p1}."
        else:
            recommendation = f"No scoring data for either {p1} or {p2} in this context."
    # Handle case where only one player is selected in compare mode (or one fails to load)
    elif mode == 'compare' and selected_players:
        # Only one player ended up being selected/having data
        p_avail = selected_players[0]
        score_p = player_scores.get(p_avail)
        p_odds = odds.get(p_avail, "N/A")
        if score_p:
            if score_p['adj_score'] > 0:
                recommendation = f"Consider Bet on {p_avail}: Only player selected/with data, positive AdjScore ({score_p['adj_score']:.1f} @{p_odds:.1f})."
            else:
                recommendation = f"Avoid Bet on {p_avail}: Only player selected/with data, AdjScore not positive ({score_p['adj_score']:.1f} @{p_odds:.1f})."
        else:
            recommendation = f"No scoring data for {p_avail} in this context."
    elif not selected_players:
        recommendation = "Please select player(s) first."

    return fig, heading_title, recommendation

# --- END OF FILE pages/betting_analyzer.py ---
