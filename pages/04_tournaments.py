# --- START OF MODIFIED pages/04_tournaments.py ---

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import traceback  # For detailed error printing

# --- Register Page ---
dash.register_page(__name__, name='Tournament Analysis', path='/tournaments')

# --- Define Page Layout --- (Layout remains the same)


def layout():
    # ... (Layout code identical to previous version) ...
    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("Tournament Winner Analysis"), width=12)),
        dbc.Row([
            dbc.Col([
                html.Label("Select Tournament:",
                           htmlFor='tournament-dropdown'),
                dcc.Dropdown(id='tournament-dropdown',
                             options=[
                                 {'label': 'Loading tournaments...', 'value': '', 'disabled': True}],
                             value=None,
                             clearable=False,
                             placeholder="Select a tournament"),
            ], width=12, md=6, className="mb-3")
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    id="loading-tournament-graph", type="circle",
                    children=[
                        dcc.Graph(id='tournament-wins-graph', figure=go.Figure())]
                ), width=12
            )
        ]),
        dbc.Row([
            dbc.Col(html.Div(id='tournament-info-message',
                    style={'marginTop': '15px'}), width=12)
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    id="loading-runs-boxplot", type="circle",
                    children=[
                        html.Div(id='runs-boxplot-title', style={
                                 'textAlign': 'center', 'marginTop': '20px', 'fontWeight': 'bold'}),
                        dcc.Graph(id='team-season-runs-boxplot',
                                  figure=go.Figure())
                    ]
                ),
                width=12, md=6,
                className="mt-4"
            ),
            dbc.Col(
                dcc.Loading(
                    id="loading-dismissals-boxplot", type="circle",
                    children=[
                        html.Div(id='dismissals-boxplot-title', style={
                                 'textAlign': 'center', 'marginTop': '20px', 'fontWeight': 'bold'}),
                        dcc.Graph(id='team-season-dismissals-boxplot',
                                  figure=go.Figure())
                    ]
                ),
                width=12, md=6,
                className="mt-4"
            )
        ])
    ], fluid=True)

# --- Callbacks ---

# Callback 1: Populate Tournament Dropdown (Unchanged)


@callback(
    Output('tournament-dropdown', 'options'),
    Output('tournament-dropdown', 'value'),
    Input('tournament-data-store', 'data'),
    prevent_initial_call=False
)
def update_tournament_dropdown(tournament_data_json):
    # ... (Code identical to previous working version) ...
    print("Callback: update_tournament_dropdown triggered")
    if tournament_data_json is None:
        print("  - No tournament data in store.")
        return [{'label': "Error: Tournament data not loaded", 'value': "", 'disabled': True}], None
    try:
        df_tournaments = pd.read_json(tournament_data_json, orient='split')
        if df_tournaments.empty or 'event_name' not in df_tournaments.columns:
            print("  - Tournament data is empty or missing 'event_name' column.")
            return [{'label': "Error: Invalid tournament data format", 'value': "", 'disabled': True}], None
        valid_tournaments = df_tournaments['event_name'].dropna().unique()
        sorted_tournaments = sorted(valid_tournaments)
        if not sorted_tournaments:
            print("  - No valid tournament names found.")
            return [{'label': "No tournaments found", 'value': "", 'disabled': True}], None
        options = [{'label': name, 'value': name}
                   for name in sorted_tournaments]
        default_value = options[0]['value'] if options else None
        print(f"  - Populated dropdown with {len(options)} tournaments.")
        return options, default_value
    except Exception as e:
        print(f"  - Error processing tournament data for dropdown: {e}")
        return [{'label': "Error processing data", 'value': "", 'disabled': True}], None


# Callback 2: Update Win Histogram (Unchanged)
@callback(
    Output('tournament-wins-graph', 'figure'),
    Output('tournament-info-message', 'children'),
    Input('tournament-dropdown', 'value'),
    State('tournament-data-store', 'data'),
    prevent_initial_call=True
)
def update_tournament_wins_graph(selected_tournament, tournament_data_json):
    # ... (Code identical to previous working version) ...
    print(
        f"Callback: update_tournament_wins_graph triggered for '{selected_tournament}'")
    info_message = ""
    if not selected_tournament:
        print("  - No tournament selected for histogram.")
        return go.Figure(), "Please select a tournament."
    if tournament_data_json is None:
        print("  - Tournament data store is empty for histogram.")
        return go.Figure(), dbc.Alert("Error: Tournament data is not available.", color="danger")
    try:
        df_tournaments = pd.read_json(tournament_data_json, orient='split')
        if df_tournaments.empty:
            print(
                "  - Tournament DataFrame is empty after deserialization for histogram.")
            return go.Figure(), dbc.Alert("Error: Tournament data is empty.", color="warning")
        if 'event_name' not in df_tournaments.columns or 'winner' not in df_tournaments.columns:
            print("  - Missing 'event_name' or 'winner' column for histogram.")
            return go.Figure(), dbc.Alert("Error: Data is missing required columns (event_name, winner).", color="danger")
        tournament_df = df_tournaments[
            (df_tournaments['event_name'] == selected_tournament) &
            (df_tournaments['winner'].notna()) &
            (df_tournaments['winner'] != '')
        ].copy()
        if tournament_df.empty:
            print("  - No match data with winners found for this tournament.")
            info_message = dbc.Alert(
                f"No match winner data found for '{selected_tournament}'.", color="info")
            return go.Figure(), info_message
        win_counts = tournament_df['winner'].value_counts().reset_index()
        win_counts.columns = ['Team', 'Wins']
        win_counts = win_counts.sort_values(by='Wins', ascending=False)
        print(
            f"  - Found {len(win_counts)} teams with wins for '{selected_tournament}'.")
        fig = px.bar(
            win_counts, x='Team', y='Wins',
            title=f'Match Wins per Team in "{selected_tournament}"',
            labels={'Team': 'Team Name', 'Wins': 'Number of Matches Won'},
            text='Wins'
        )
        fig.update_layout(
            xaxis_title="Team (Click on any team to see more detailed analysis)", yaxis_title="Number of Matches Won",
            xaxis={'categoryorder': 'total descending'},
            margin=dict(t=50, b=20, l=30, r=20), title_x=0.5
        )
        fig.update_traces(textposition='outside')
        print(f"  - Histogram figure type: {type(fig)}")
        return fig, info_message
    except Exception as e:
        print(f"  - Error generating tournament histogram: {e}")
        return go.Figure(), dbc.Alert(f"An error occurred while generating the histogram: {e}", color="danger")


# Callback 3: Update BOTH Box Plots (REVISED Dismissal Logic using existing numeric column)
@callback(
    Output('team-season-runs-boxplot', 'figure'),
    Output('runs-boxplot-title', 'children'),
    Output('team-season-dismissals-boxplot', 'figure'),
    Output('dismissals-boxplot-title', 'children'),
    Input('tournament-wins-graph', 'clickData'),
    State('tournament-dropdown', 'value'),
    State('tournament-data-store', 'data'),
    prevent_initial_call=True
)
def update_team_season_boxplots(clickData, selected_tournament, tournament_data_json):
    """Generates box plots for runs and dismissals per season for the clicked team."""
    print("\n--- Callback: update_team_season_boxplots triggered ---")
    print(f"Selected Tournament: {selected_tournament}")
    print(f"Received clickData: {json.dumps(clickData, indent=2)}")

    # Initialize outputs
    fig_runs = go.Figure()
    runs_title = ""
    fig_dismissals = go.Figure()
    dismissals_title = ""
    empty_return = fig_runs, runs_title, fig_dismissals, dismissals_title

    # Basic checks
    if clickData is None:
        return empty_return
    if not selected_tournament:
        return empty_return
    if tournament_data_json is None:
        error_msg = dbc.Alert(
            "Error: Tournament data is not available.", color="danger")
        return go.Figure(), error_msg, go.Figure(), error_msg

    try:
        # Extract clicked team
        try:
            clicked_team = clickData['points'][0]['x']
        except (KeyError, IndexError, TypeError):
            return empty_return
        print(f"  - Clicked team: '{clicked_team}'")

        # Load data
        df_tournaments = pd.read_json(tournament_data_json, orient='split')
        if df_tournaments.empty:
            error_msg = dbc.Alert(
                "Error: Tournament data is empty.", color="warning")
            return go.Figure(), error_msg, go.Figure(), error_msg
        print(f"  - Available columns: {df_tournaments.columns.tolist()}")

        # --- Define required columns ---
        runs_req_cols = {'season', 'runs_off_bat'}
        # For Dismissals plot, now assuming 'player_dismissed' is the numeric count
        # Still need 'batting_team' to filter correctly
        dismissals_req_cols = {
            'season', 'player_dismissed', 'match_id', 'batting_team'}

        # --- Check for Batting Team Column ---
        team_batted_col = 'batting_team'
        if team_batted_col not in df_tournaments.columns:
            print(
                f"  - CRITICAL ERROR: Column '{team_batted_col}' is required and missing.")
            error_msg = dbc.Alert(
                f"Error: Required column '{team_batted_col}' missing.", color="danger")
            # Attempt to show runs plot if possible, error for dismissals
            fig_runs, runs_title = generate_runs_plot(
                # Use a helper potentially, or inline
                df_tournaments, selected_tournament, clicked_team, runs_req_cols, team_batted_col)
            return fig_runs, runs_title, go.Figure(), error_msg

        # --- Filter Data for Selected Team (Batting) & Tournament ---
        print(
            f"  - Filtering for event_name='{selected_tournament}', {team_batted_col}='{clicked_team}'")
        team_df_filtered = df_tournaments[
            (df_tournaments['event_name'] == selected_tournament) &
            (df_tournaments[team_batted_col] == clicked_team)
        ].copy()
        print(
            f"  - Records after filtering for team batting & tournament: {len(team_df_filtered)}")

        if team_df_filtered.empty:
            info_msg = dbc.Alert(
                f"No data found where '{clicked_team}' batted in '{selected_tournament}'.", color="info")
            return go.Figure(), info_msg, go.Figure(), info_msg

        # --- Generate Runs Box Plot ---
        if runs_req_cols.issubset(df_tournaments.columns):
            print("  - Preparing Runs Box Plot...")
            runs_df = team_df_filtered.copy()
            # Convert runs to numeric, coercing errors
            runs_df['runs_off_bat'] = pd.to_numeric(
                runs_df['runs_off_bat'], errors='coerce')
            # Drop rows where runs couldn't be converted or are missing season
            runs_df.dropna(subset=['season', 'runs_off_bat'], inplace=True)
            runs_df['season'] = runs_df['season'].astype(str)

            if not runs_df.empty:
                runs_df = runs_df.sort_values(by='season')
                print(
                    f"  - Found {len(runs_df)} valid records for runs boxplot.")
                fig_runs = px.box(runs_df, x='season', y='runs_off_bat', points="all",
                                  labels={'season': 'Season', 'runs_off_bat': 'Runs Scored (off bat)'})
                fig_runs.update_layout(xaxis_title="Season", yaxis_title="Runs Scored",
                                       margin=dict(t=5, b=20, l=30, r=20), xaxis={'type': 'category'})
                runs_title = f"Runs Scored per Season ({clicked_team} Batting) in '{selected_tournament}'"
            else:
                print("  - No valid run data after cleaning.")
                runs_title = dbc.Alert(
                    f"No valid run data found for '{clicked_team}' (batting).", color="info")
        else:
            missing_runs_cols = runs_req_cols - set(df_tournaments.columns)
            print(
                f"  - ERROR: Missing columns for runs plot: {missing_runs_cols}.")
            runs_title = dbc.Alert(
                f"Error: Missing data columns ({', '.join(missing_runs_cols)}) for runs plot.", color="danger")

        # --- Generate Dismissals Box Plot (using numeric 'player_dismissed') ---
        if dismissals_req_cols.issubset(df_tournaments.columns):
            print(
                "  - Preparing Dismissals Box Plot (using numeric 'player_dismissed')...")
            dismissals_df = team_df_filtered.copy()

            # --- REVISED LOGIC: Use 'player_dismissed' directly ---
            # 1. Convert 'player_dismissed' to numeric, handling errors
            dismissals_df['dismissal_count'] = pd.to_numeric(
                dismissals_df['player_dismissed'], errors='coerce')

            # --- DEBUG ---
            print(
                f"  - Sample 'dismissal_count' after conversion: {dismissals_df['dismissal_count'].dropna().unique()[:10]}")
            print(
                f"  - Stats for converted dismissal_count:\n{dismissals_df['dismissal_count'].describe()}")

            # 2. Drop rows where conversion failed or season is missing
            dismissals_df.dropna(
                subset=['season', 'dismissal_count'], inplace=True)

            # 3. Convert season to string
            dismissals_df['season'] = dismissals_df['season'].astype(str)

            # 4. **IMPORTANT**: Decide on aggregation level.
            #    Does each row represent a match? Or an event within a match?
            #    If each row is an EVENT, we likely need the *last* dismissal count per match.
            #    If each row *is* a match summary, we can plot directly.
            #    Let's assume for now each row is *not* a match summary and find the max dismissal count per match.
            print(
                "  - Assuming data is event-level. Finding max dismissal count per match...")
            dismissal_summary_per_match = dismissals_df.loc[
                dismissals_df.groupby(['season', 'match_id'])[
                    'dismissal_count'].idxmax()
            ]
            # If you are *certain* each row in team_df_filtered is already one match summary for that team,
            # you can skip the .loc[...idxmax()] step and use dismissals_df directly:
            # dismissal_summary_per_match = dismissals_df

            print(
                f"  - Final data for dismissals plot (sample):\n{dismissal_summary_per_match[['season', 'match_id', 'dismissal_count']].head()}")

            # 5. Plot the summary data
            if not dismissal_summary_per_match.empty:
                dismissal_summary_per_match = dismissal_summary_per_match.sort_values(
                    by='season')
                print(
                    f"  - Found {len(dismissal_summary_per_match)} matches with dismissal data for boxplot.")

                fig_dismissals = px.box(
                    dismissal_summary_per_match,
                    x='season',
                    # Use the numeric column directly (after potential aggregation)
                    y='dismissal_count',
                    points="all",
                    labels={'season': 'Season',
                            'dismissal_count': 'Dismissals Recorded'}
                )
                fig_dismissals.update_layout(
                    xaxis_title="Season", yaxis_title="Dismissals Recorded",
                    margin=dict(t=5, b=20, l=30, r=20),
                    xaxis={'type': 'category'}
                )
                dismissals_title = f"Dismissals Recorded per Season ({clicked_team} Batting) in '{selected_tournament}'"
            else:
                print("  - No valid dismissal data found after cleaning/aggregation.")
                dismissals_title = dbc.Alert(
                    f"No valid dismissal data found for '{clicked_team}' (batting).", color="info")

        elif not dismissals_req_cols.issubset(df_tournaments.columns):
            missing_dismissal_cols = dismissals_req_cols - \
                set(df_tournaments.columns)
            print(
                f"  - ERROR: Missing columns for dismissals plot: {missing_dismissal_cols}.")
            dismissals_title = dbc.Alert(
                f"Error: Missing data columns ({', '.join(missing_dismissal_cols)}) for dismissals plot.", color="danger")

        # --- Return all figures and titles ---
        print("  - Returning figures and titles for both box plots.")
        return fig_runs, runs_title, fig_dismissals, dismissals_title

    except Exception as e:
        print(
            f"  - !!! UNEXPECTED ERROR in update_team_season_boxplots: {e} !!!")
        print(traceback.format_exc())
        error_msg = dbc.Alert(
            f"An unexpected error occurred generating box plots: {e}", color="danger")
        return go.Figure(), error_msg, go.Figure(), error_msg

# Helper function (Optional, can keep runs logic inline)


def generate_runs_plot(df, tournament, team, req_cols, team_col):
    # Basic implementation - can be expanded
    fig = go.Figure()
    title = ""
    if not req_cols.issubset(df.columns):
        missing = req_cols - set(df.columns)
        title = dbc.Alert(f"Missing runs columns: {missing}", color="danger")
        return fig, title
    if team_col not in df.columns:
        title = dbc.Alert(f"Missing team column: {team_col}", color="danger")
        return fig, title

    # Add filtering and plot generation here... similar to inline version
    # ...
    return fig, title


# --- END OF MODIFIED pages/04_tournaments.py ---
