# --- START OF FILE player_analysis.py ---

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, dash_table # Import ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
from datetime import datetime

# --- Register this page ---
dash.register_page(__name__, path='/player-analysis', name='Player Analysis', title='Player Analysis')

# --- Configuration ---
warnings.filterwarnings("ignore", category=RuntimeWarning) # Suppress specific runtime warnings (e.g., divide by zero)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Constants ---
DEFAULT_ALL_VALUE = "ALL" # Value representing 'All' options in dropdowns

# --- Helper Functions ---
# NOTE: load_data is assumed to be handled centrally in your main app file (e.g., multi_page_app.py)
# and data is passed via dcc.Store.

def calculate_batsman_stats(df):
    """Calculates aggregated batsman statistics from a dataframe."""
    if df is None or df.empty:
        return {}

    stats = {}
    stats['matches'] = df['match_id'].nunique()
    # Count innings where balls were faced (relevant for batting average etc.)
    # Group by match_id first to see if any balls were faced in that match
    match_balls_faced = df.groupby('match_id')['balls_faced'].sum()
    stats['innings'] = match_balls_faced[match_balls_faced > 0].count() # Count matches where player batted

    stats['runs'] = int(df['runs_scored'].sum())
    stats['balls_faced'] = int(df['balls_faced'].sum())
    stats['fours'] = int(df['fours_scored'].sum())
    stats['sixes'] = int(df['sixes_scored'].sum())
    # Count outs specifically where player_out seems to indicate an out (e.g., == 1)
    # Handle potential NaN in 'player_out' by filling with 0 before sum/shape check
    # Make sure we count outs per match_id only once if player can be out multiple times in data anomalies
    outs_per_match = df[df['player_out'].fillna(0) == 1].groupby('match_id').size()
    stats['outs'] = int(outs_per_match.count()) # Count matches where player got out

    stats['average'] = (stats['runs'] / stats['outs']) if stats['outs'] > 0 else np.inf
    stats['strike_rate'] = (stats['runs'] / stats['balls_faced'] * 100) if stats['balls_faced'] > 0 else 0.0

    # Dismissal breakdown
    if 'out_kind' in df.columns:
        # Filter where player_out is 1 and out_kind is not NaN
        dismissals_df = df[(df['player_out'].fillna(0) == 1) & (df['out_kind'].notna())]
        # If multiple entries per match exist for out, take first? Or just count kinds? Count kinds.
        dismissals = dismissals_df['out_kind'].value_counts().to_dict()
    else:
        dismissals = {}
    stats['dismissals'] = dismissals

    # Calculate not outs: Total innings batted - total outs
    stats['not_outs'] = max(0, stats['innings'] - stats['outs']) # Ensure non-negative

    return stats

def calculate_bowler_stats(df):
    """Calculates aggregated bowler statistics from a dataframe."""
    if df is None or df.empty:
        return {}

    # Filter for innings where the player actually bowled
    # Aggregate first to get per-match bowling figures
    match_bowling_stats = df.groupby('match_id').agg(
        balls_bowled=('balls_bowled', 'sum'),
        runs_conceded=('runs_conceded', 'sum'),
        wickets_taken=('wickets_taken', 'sum'),
        maidens=('maidens', 'sum'),
        bowled_done=('bowled_done', 'sum'),
        lbw_done=('lbw_done', 'sum'),
        dot_balls_as_bowler=('dot_balls_as_bowler', 'sum') # Ensure this column exists
    ).reset_index()

    bowling_innings = match_bowling_stats[match_bowling_stats['balls_bowled'] > 0].copy()

    # Handle potential NaN values resulting from aggregation if source cols were missing/all NaN
    numeric_cols = ['balls_bowled', 'runs_conceded', 'wickets_taken', 'maidens', 'bowled_done', 'lbw_done', 'dot_balls_as_bowler']
    for col in numeric_cols:
         if col in bowling_innings.columns:
            bowling_innings[col] = bowling_innings[col].fillna(0)
         else:
             bowling_innings[col] = 0 # Add if missing after aggregation


    if bowling_innings.empty:
        return {
            'matches': df['match_id'].nunique(), 'innings_bowled': 0, 'wickets': 0,
            'average': np.inf, 'economy': np.inf, 'strike_rate': np.inf,
            'balls_bowled': 0, 'runs_conceded': 0, 'maidens': 0, 'bowled_wickets': 0,
            'lbw_wickets': 0, 'dot_balls': 0
        }

    stats = {}
    stats['matches'] = df['match_id'].nunique() # Matches player participated in overall
    stats['innings_bowled'] = bowling_innings.shape[0] # Innings actually bowled
    stats['balls_bowled'] = int(bowling_innings['balls_bowled'].sum())
    stats['runs_conceded'] = int(bowling_innings['runs_conceded'].sum())
    stats['wickets'] = int(bowling_innings['wickets_taken'].sum())
    stats['maidens'] = int(bowling_innings['maidens'].sum())
    stats['bowled_wickets'] = int(bowling_innings['bowled_done'].sum())
    stats['lbw_wickets'] = int(bowling_innings['lbw_done'].sum())
    stats['dot_balls'] = int(bowling_innings['dot_balls_as_bowler'].sum())

    # Calculate rates, handling division by zero
    stats['average'] = (stats['runs_conceded'] / stats['wickets']) if stats['wickets'] > 0 else np.inf
    stats['economy'] = (stats['runs_conceded'] / stats['balls_bowled'] * 6) if stats['balls_bowled'] > 0 else np.inf
    stats['strike_rate'] = (stats['balls_bowled'] / stats['wickets']) if stats['wickets'] > 0 else np.inf

    return stats

# --- NEW HELPER for Batsman Match Aggregation ---
def aggregate_runs_by_match_year(df):
    """Aggregates runs scored per match and adds the year."""
    if df is None or df.empty or not all(c in df.columns for c in ['match_id', 'start_date', 'runs_scored']):
        return pd.DataFrame() # Return empty if required cols missing

    # Ensure start_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['start_date']):
         df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')

    df_agg = df.dropna(subset=['start_date']) # Drop rows where date conversion failed
    if df_agg.empty:
        return pd.DataFrame()

    # Extract year
    df_agg['year'] = df_agg['start_date'].dt.year

    # Group by match and year, sum runs
    match_runs = df_agg.groupby(['year', 'match_id'])['runs_scored'].sum().reset_index()
    match_runs.rename(columns={'runs_scored': 'runs_in_match'}, inplace=True)

    # Filter out matches where player didn't score (or data is bad) - optional, depends on if you want to show 0s
    # match_runs = match_runs[match_runs['runs_in_match'] >= 0] # Keep 0s

    return match_runs


# --- Plotting Functions ---

# --- REMOVED: plot_batsman_timeline --- (Now using Ridge Plot)

# --- NEW: Ridge Plot for Runs Distribution by Year ---
def plot_runs_distribution_ridge(df):
    """
    Creates a ridge plot (using violin) showing the distribution of matchwise runs scored by year.
    """
    match_runs_df = aggregate_runs_by_match_year(df)

    if match_runs_df.empty or 'year' not in match_runs_df.columns or 'runs_in_match' not in match_runs_df.columns:
        return go.Figure().update_layout(title="Runs Distribution by Year (Ridge Plot) - Insufficient Data", xaxis_title="Runs Scored in Match", yaxis_title="Year")

    # Ensure year is treated as a categorical variable for plotting, sort it
    match_runs_df['year'] = match_runs_df['year'].astype(str)
    years_sorted = sorted(match_runs_df['year'].unique()) # Sort chronologically

    # Create the plot using px.violin, styled like a ridge plot
    fig = px.violin(
        match_runs_df,
        x='runs_in_match',
        y='year',
        orientation='h', # Horizontal orientation
        points=False,     # Don't show individual points
        color='year',     # Color by year
        category_orders={'year': years_sorted}, # Ensure years are ordered correctly
        box=False,         # Hide the inner box plot
        violinmode='overlay', # Overlay helps but doesn't separate, px handles separation by y-category
        title="Distribution of Runs Scored per Match by Year (Ridge Plot)",
        labels={'runs_in_match': 'Runs Scored in Match', 'year': 'Year'},
        color_discrete_sequence=px.colors.sequential.Viridis_r # Example color sequence
    )

    # Customize the appearance for a ridge plot feel
    fig.update_traces(
        side='positive', # Show only one side of the violin
        width=1.0,       # Adjust width if needed, relative to category spacing
        meanline_visible=False # Hide the mean line
    )

    fig.update_layout(
        xaxis=dict(range=[0, max(200, match_runs_df['runs_in_match'].max() * 1.05)]), # Set x-axis range, ensure at least 0-150
        yaxis_title="Year",
        xaxis_title="Runs Scored in Match",
        showlegend=False, # Legend is redundant if colors match y-axis labels
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    # Improve hover data (plotly express violin hover is sometimes tricky to customize perfectly)
    # fig.update_traces(hovertemplate='Year: %{y}<br>Runs: %{x}<extra></extra>') # Basic hover

    return fig

def plot_run_contribution(stats):
    """Plots a pie chart of run sources (1s/2s/3s, 4s, 6s)."""
    if not stats or stats.get('runs', 0) == 0:
        return go.Figure().update_layout(title="Run Contribution (No Runs Scored)")
    runs_4s = stats.get('fours', 0) * 4
    runs_6s = stats.get('sixes', 0) * 6
    total_runs = stats.get('runs', 0)
    other_runs = max(0, total_runs - runs_4s - runs_6s)
    data = {'Run Type': ['Other Runs (1s, 2s, 3s)', 'Runs from 4s', 'Runs from 6s'], 'Runs': [other_runs, runs_4s, runs_6s]}
    df_runs = pd.DataFrame(data)
    df_runs = df_runs[df_runs['Runs'] > 0] # Only show slices with runs
    if df_runs.empty:
        return go.Figure().update_layout(title="Run Contribution (No Runs Scored)")
    fig = px.pie(df_runs, names='Run Type', values='Runs', title="Run Scoring Contribution", hole=0.35, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05]*len(df_runs))
    fig.update_layout(showlegend=True, margin=dict(t=50, b=0, l=0, r=0))
    return fig

def plot_dismissal_pie(stats):
    """Plots a pie chart of batsman dismissal types."""
    if not stats or not stats.get('dismissals') or stats.get('outs', 0) == 0:
        return go.Figure().update_layout(title="Dismissal Analysis (No Dismissals Recorded)")

    dismissals_data = stats.get('dismissals', {})
    # Filter out 'not out' or zero counts
    labels = [k.capitalize() for k, v in dismissals_data.items() if v > 0 and k != 'not out']
    values = [v for k, v in dismissals_data.items() if v > 0 and k != 'not out']

    if not values:
        return go.Figure().update_layout(title="Dismissal Analysis (No Dismissals Recorded)")

    fig = px.pie(names=labels, values=values,
                 title="Dismissal Type Breakdown (%)", hole=0.35,
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05] * len(labels))
    fig.update_layout(showlegend=True, legend_title_text='Dismissal Type', margin=dict(t=50, b=0, l=0, r=0))
    return fig


# --- MODIFIED: Strike Rate vs Order Plot ---
def plot_sr_vs_order(df):
    """
    Plots Strike Rate vs Batting Order.
    Bar color intensity represents average runs scored per match at that order.
    """
    # --- MODIFIED: Added 'match_id' to required columns ---
    required_cols = ['order_seen', 'runs_scored', 'balls_faced', 'match_id']
    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        return go.Figure().update_layout(title="Strike Rate vs Batting Order (Insufficient Data)", xaxis_title="Batting Order", yaxis_title="Strike Rate")

    df_plot = df.copy()
    # --- MODIFIED: Include 'match_id' in processing if needed (it's often string, no numeric conversion needed) ---
    for col in ['order_seen', 'runs_scored', 'balls_faced']:
         df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
    df_plot = df_plot.dropna(subset=['order_seen', 'runs_scored', 'balls_faced', 'match_id']) # ensure match_id is not null
    df_plot = df_plot[df_plot['balls_faced'] > 0] # Only include innings where balls were faced
    if df_plot.empty:
        return go.Figure().update_layout(title="Strike Rate vs Batting Order (No Valid Data)", xaxis_title="Batting Order", yaxis_title="Strike Rate")

    df_plot['order_seen'] = df_plot['order_seen'].astype(int)

    # --- MODIFIED: Grouping now includes counting unique matches ---
    grouped = df_plot.groupby('order_seen').agg(
        total_runs=('runs_scored', 'sum'),
        total_balls=('balls_faced', 'sum'),
        match_count=('match_id', 'nunique') # Count unique matches played at this order
    ).reset_index()

    # Calculate Strike Rate
    grouped['strike_rate'] = (grouped['total_runs'] / grouped['total_balls']) * 100
    grouped['strike_rate'] = grouped['strike_rate'].replace([np.inf, -np.inf, np.nan], 0)

    # --- MODIFIED: Calculate Average Runs per Match at that order ---
    grouped['avg_runs_per_match'] = grouped.apply(
        lambda row: row['total_runs'] / row['match_count'] if row['match_count'] > 0 else 0,
        axis=1
    )
    grouped['avg_runs_per_match'] = grouped['avg_runs_per_match'].replace([np.inf, -np.inf, np.nan], 0)


    if grouped.empty:
        return go.Figure().update_layout(title="Strike Rate vs Batting Order (No Data After Grouping)", xaxis_title="Batting Order", yaxis_title="Strike Rate")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped['order_seen'],
        y=grouped['strike_rate'],
        marker=dict(
            # --- MODIFIED: Color by average runs per match ---
            color=grouped['avg_runs_per_match'],
            colorscale='Viridis', # Or choose another like 'Plasma', 'Blues', 'YlGnBu'
            # --- MODIFIED: Update color bar title ---
            colorbar=dict(title="Avg Runs / Match", x=1.02, y=0.5, len=0.75),
            showscale=True,
            line_width=0
        ),
        hovertemplate=(
            "<b>Order:</b> %{x}<br>"
            "<b>Strike Rate:</b> %{y:.2f}<br>"
            # --- MODIFIED: Update hover text for color ---
            "<b>Avg Runs per Match:</b> %{marker.color:.1f}<extra></extra>"
        )
    ))

    max_sr = grouped['strike_rate'].max() if not grouped.empty else 0
    y_axis_max = max(100, max_sr * 1.1) if pd.notna(max_sr) and max_sr > 0 else 100

    if not grouped.empty:
        min_order = grouped['order_seen'].min()
        max_order = grouped['order_seen'].max()
        min_order = max(1, min_order) if pd.notna(min_order) else 1
        max_order = max(min_order, max_order) if pd.notna(max_order) else min_order
        orders = list(range(min_order, max_order + 1))
        xaxis_config = dict(tickmode='array', tickvals=orders, ticktext=[str(o) for o in orders])
    else:
        xaxis_config = {}

    fig.update_layout(
        # --- MODIFIED: Update subtitle ---
        title='Strike Rate vs Batting Order<br>(Color intensity indicates Average Runs per Match at that order)',
        xaxis_title='Batting Order',
        yaxis_title='Strike Rate',
        showlegend=False,
        yaxis_range=[0, y_axis_max],
        xaxis=xaxis_config,
        margin=dict(t=80, r=120) # Keep margin for color bar
    )
    return fig


# --- Bowler Plotting Functions (Unchanged) ---

def plot_bowler_timeline(df, aggregation_period='1Y'): # Callers will pass 1Y
    """Plots Wickets Taken and Economy Rate over time."""
    # Ensure 'start_date' column exists and has valid dates
    if df is None or df.empty or 'start_date' not in df.columns or df['start_date'].isnull().all():
        return go.Figure().update_layout(title="Bowling Performance Over Time (No Data/Dates)", xaxis_title="Date", yaxis_title="Wickets / Economy")

    if not pd.api.types.is_datetime64_any_dtype(df['start_date']):
         df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
         if df['start_date'].isnull().all():
            return go.Figure().update_layout(title="Bowling Performance Over Time (Invalid Dates)", xaxis_title="Date", yaxis_title="Wickets / Economy")

    # Aggregate first to get per-match bowling figures before filtering
    match_bowling_stats = df.groupby(['match_id', 'start_date']).agg(
        balls_bowled=('balls_bowled', 'sum'),
        runs_conceded=('runs_conceded', 'sum'),
        wickets_taken=('wickets_taken', 'sum')
        # Add other columns if needed for hover or calculation
    ).reset_index()

    df_bowled = match_bowling_stats[match_bowling_stats['balls_bowled'] > 0].copy()
    if df_bowled.empty:
        return go.Figure().update_layout(title="Bowling Performance Over Time (No Bowling Innings)", xaxis_title="Date", yaxis_title="Wickets / Economy")

     # Handle potential NaN values in key numeric columns before aggregation
    numeric_cols = ['wickets_taken', 'runs_conceded', 'balls_bowled']
    for col in numeric_cols:
        if col in df_bowled.columns:
            df_bowled[col] = df_bowled[col].fillna(0)
        else:
             df_bowled[col] = 0

    df_time = df_bowled.set_index('start_date')
    # Now resample the *match-aggregated* data
    df_resampled = df_time.resample(aggregation_period).agg(
        wickets_taken=('wickets_taken', 'sum'),
        runs_conceded=('runs_conceded', 'sum'),
        balls_bowled=('balls_bowled', 'sum'),
        matches=('match_id', 'nunique') # Count distinct matches *with bowling* in period
    ).reset_index()

    df_resampled['economy_rate'] = df_resampled.apply(
        lambda row: (row['runs_conceded'] / row['balls_bowled'] * 6) if row['balls_bowled'] > 0 else np.inf,
        axis=1
    )
    df_resampled['economy_rate'] = df_resampled['economy_rate'].replace([np.inf, -np.inf], np.nan)
    df_resampled = df_resampled[(df_resampled['wickets_taken'] > 0) | (df_resampled['balls_bowled'] > 0)]

    if df_resampled.empty:
         return go.Figure().update_layout(title="Bowling Performance Over Time (No Data after Resample)", xaxis_title="Date", yaxis_title="Wickets / Economy")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_resampled['start_date'], y=df_resampled['wickets_taken'], mode='lines+markers', name='Wickets Taken', yaxis='y1',
        hovertemplate='<b>Period Start</b>: %{x|%Y-%m}<br><b>Wickets</b>: %{y}<br><b>Matches Bowled</b>: %{customdata[0]}<extra></extra>',
        customdata=df_resampled[['matches']]
    ))
    fig.add_trace(go.Scatter(
        x=df_resampled['start_date'], y=df_resampled['economy_rate'], mode='lines+markers', name='Economy Rate', yaxis='y2',
        line=dict(dash='dash'), connectgaps=False,
        hovertemplate='<b>Period Start</b>: %{x|%Y-%m}<br><b>Economy</b>: %{y:.2f}<extra></extra>'
    ))

    max_econ = df_resampled['economy_rate'].dropna().max()
    econ_upper_limit = max(15, max_econ * 1.2) if pd.notna(max_econ) and max_econ > 0 else 15 # Adjusted upper limit

    fig.update_layout(
        title=f"Wickets & Economy Over Time (Aggregated per {aggregation_period})",
        xaxis_title="Period Start",
        yaxis=dict(title=dict(text="Wickets Taken", font=dict(color="#1f77b4")), tickfont=dict(color="#1f77b4")),
        yaxis2=dict(title=dict(text="Economy Rate", font=dict(color="#ff7f0e")), tickfont=dict(color="#ff7f0e"), anchor="x", overlaying="y", side="right", range=[0, econ_upper_limit]),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_wicket_types_pie(stats):
    """Plots a pie chart of bowler wicket types (bowled, lbw, other)."""
    if not stats or stats.get('wickets', 0) == 0:
        return go.Figure().update_layout(title="Wicket Types (No Wickets Taken)")
    total_wickets = stats.get('wickets', 0)
    bowled = stats.get('bowled_wickets', 0)
    lbw = stats.get('lbw_wickets', 0)
    other = max(0, total_wickets - bowled - lbw)
    data = {'Wicket Type': ['Bowled', 'LBW', 'Other (Caught, Stumped, etc.)'], 'Count': [bowled, lbw, other]}
    df_wickets = pd.DataFrame(data)
    df_wickets = df_wickets[df_wickets['Count'] > 0] # Only show types with wickets
    if df_wickets.empty:
        return go.Figure().update_layout(title="Wicket Types (No Wickets Recorded)") # Changed message slightly
    fig = px.pie(df_wickets, names='Wicket Type', values='Count', title="Wicket Types Breakdown", hole=0.35, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition='inside', textinfo='percent+value', pull=[0.05]*len(df_wickets))
    fig.update_layout(showlegend=True, legend_title_text='Wicket Type', margin=dict(t=50, b=0, l=0, r=0))
    return fig

def plot_dot_ball_percentage_timeline(df, aggregation_period='1Y'): # Callers will pass 1Y
    """Plots Dot Ball Percentage over time."""
    dot_col = 'dot_balls_as_bowler'
    # Ensure 'start_date' column exists and has valid dates
    if df is None or df.empty or 'start_date' not in df.columns or df['start_date'].isnull().all() or dot_col not in df.columns:
        missing_info = "No Data/Dates" if dot_col in df.columns else f"Missing Column: {dot_col}"
        return go.Figure().update_layout(title=f"Dot Ball % Over Time ({missing_info})", xaxis_title="Date", yaxis_title="Dot Ball %")

    if not pd.api.types.is_datetime64_any_dtype(df['start_date']):
         df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
         if df['start_date'].isnull().all():
            return go.Figure().update_layout(title="Dot Ball % Over Time (Invalid Dates)", xaxis_title="Date", yaxis_title="Dot Ball %")

    # Aggregate first to get per-match bowling figures before filtering
    match_bowling_stats = df.groupby(['match_id', 'start_date']).agg(
        balls_bowled=('balls_bowled', 'sum'),
        dot_balls=(dot_col, 'sum') # Aggregate dot balls per match
    ).reset_index()

    # Ensure columns exist after aggregation, fill potential NaNs with 0
    if 'balls_bowled' not in match_bowling_stats.columns: match_bowling_stats['balls_bowled'] = 0
    else: match_bowling_stats['balls_bowled'] = match_bowling_stats['balls_bowled'].fillna(0)

    if 'dot_balls' not in match_bowling_stats.columns: match_bowling_stats['dot_balls'] = 0
    else: match_bowling_stats['dot_balls'] = match_bowling_stats['dot_balls'].fillna(0)


    df_bowled = match_bowling_stats[match_bowling_stats['balls_bowled'] > 0].copy()
    if df_bowled.empty:
        return go.Figure().update_layout(title="Dot Ball % Over Time (No Bowling Innings)", xaxis_title="Date", yaxis_title="Dot Ball %")

    df_time = df_bowled.set_index('start_date')
    # Now resample the *match-aggregated* data
    df_resampled = df_time.resample(aggregation_period).agg(
        dot_balls=('dot_balls', 'sum'),
        balls_bowled=('balls_bowled', 'sum'),
        matches=('match_id', 'nunique') # Count distinct matches *with bowling* in period
    ).reset_index()

    df_resampled['dot_ball_percentage'] = df_resampled.apply(
        lambda row: (row['dot_balls'] / row['balls_bowled'] * 100) if row['balls_bowled'] > 0 else 0,
        axis=1
    )
    df_resampled['dot_ball_percentage'] = df_resampled['dot_ball_percentage'].replace([np.inf, -np.inf, np.nan], 0)
    df_resampled = df_resampled[df_resampled['balls_bowled'] > 0]

    if df_resampled.empty:
         return go.Figure().update_layout(title="Dot Ball % Over Time (No Data after Resample)", xaxis_title="Date", yaxis_title="Dot Ball %")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_resampled['start_date'], y=df_resampled['dot_ball_percentage'], mode='lines+markers', name='Dot Ball %',
        hovertemplate='<b>Period Start</b>: %{x|%Y-%m}<br><b>Dot Ball %%</b>: %{y:.1f}%%<br><b>Matches Bowled</b>: %{customdata[0]}<extra></extra>',
        customdata=df_resampled[['matches']]
    ))
    fig.update_layout(
        title=f"Dot Ball Percentage Over Time (Aggregated per {aggregation_period})",
        xaxis_title="Period Start",
        yaxis=dict(title="Dot Ball Percentage (%)", range=[0, 100]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


# --- Page Layout Definition ---
def layout():
    # Define the basic structure, dropdowns populated by callback
    return dbc.Container([
        dbc.Row(dbc.Col(html.H1("🏏 Player Performance Analysis"), width=12), className="mb-4 mt-2"),

        # --- Control Row ---
        dbc.Row([
            # Player Dropdown
            dbc.Col([
                dbc.Label("Select Player:", html_for='pa-player-dropdown'),
                dcc.Dropdown(id='pa-player-dropdown', options=[], value=None, clearable=False)
            ], width=12, lg=4, className="mb-3"),

            # Role RadioItems
            dbc.Col([
                dbc.Label("Select Role:", html_for='pa-role-radio'),
                dcc.RadioItems(
                    id='pa-role-radio',
                    options=[{'label': 'Batsman', 'value': 'Batsman'}, {'label': 'Bowler', 'value': 'Bowler'}],
                    value='Batsman', inline=True, labelStyle={'margin-right': '15px'}, inputStyle={"margin-right": "5px"}
                )
            ], width=12, lg=2, className="mb-3 align-self-center"), # Align vertically

            # Date Range Picker
             dbc.Col([
                dbc.Label("Select Date Range:", html_for='pa-date-range-picker'),
                dcc.DatePickerRange(
                    id='pa-date-range-picker',
                    min_date_allowed=None, # Will be set based on data
                    max_date_allowed=None, # Will be set based on data
                    start_date=None,       # Initially empty
                    end_date=None,         # Initially empty
                    display_format='YYYY-MM-DD',
                    className="d-block" # Ensure it takes block display
                )
            ], width=12, lg=6, className="mb-3"),


        ], className="align-items-end"), # Align items to bottom for better look

        # --- Filter Row ---
         dbc.Row([
             # Match Type Dropdown
            dbc.Col([
                dbc.Label("Filter by Match Type:", html_for='pa-match-type-dropdown'),
                dcc.Dropdown(id='pa-match-type-dropdown', options=[], value=DEFAULT_ALL_VALUE, clearable=False)
            ], width=12, md=6, lg=4, className="mb-3"),

            # Venue Dropdown
            dbc.Col([
                dbc.Label("Filter by Venue:", html_for='pa-venue-dropdown'),
                dcc.Dropdown(id='pa-venue-dropdown', options=[], value=DEFAULT_ALL_VALUE, clearable=False)
            ], width=12, md=6, lg=4, className="mb-3"),

         ], className="mb-4"),


        html.Hr(),

        # Output Area
        dbc.Row(dbc.Col(
            dcc.Loading(id="pa-loading-output", children=[html.Div(id='pa-stats-output-area')]),
            width=12
        ))

    ], fluid=True)


# --- Main Callback for Player Analysis Page ---
@callback(
    Output('pa-stats-output-area', 'children'),
    Output('pa-player-dropdown', 'options'),
    Output('pa-player-dropdown', 'value'),
    Output('pa-match-type-dropdown', 'options'),
    Output('pa-venue-dropdown', 'options'),
    Output('pa-date-range-picker', 'min_date_allowed'), # Set min date for picker
    Output('pa-date-range-picker', 'max_date_allowed'), # Set max date for picker
    Output('pa-date-range-picker', 'start_date'),      # Potentially set default start
    Output('pa-date-range-picker', 'end_date'),        # Potentially set default end
    Input('player-analysis-data-store', 'data'),      # Input: Data from the central store
    Input('pa-player-dropdown', 'value'),
    Input('pa-role-radio', 'value'),
    Input('pa-match-type-dropdown', 'value'),
    Input('pa-venue-dropdown', 'value'),
    Input('pa-date-range-picker', 'start_date'),      # Input: Date range start
    Input('pa-date-range-picker', 'end_date'),        # Input: Date range end
    State('pa-player-dropdown', 'options'),           # State: Current options
    State('pa-date-range-picker', 'start_date'),      # State: Current start date in picker
    State('pa-date-range-picker', 'end_date'),        # State: Current end date in picker
)
def update_player_analysis(player_data_json, selected_player, selected_role,
                           selected_match_type, selected_venue,
                           selected_start_date_str, selected_end_date_str,
                           current_player_options,
                           state_start_date, state_end_date):
    """
    Loads data, populates dropdowns & date picker, filters data (including date range),
    and generates visualizations based on selections.
    """
    triggered_id = ctx.triggered_id
    print(f"Callback update_player_analysis triggered by: {triggered_id}")
    print(f"Inputs: Player='{selected_player}', Role='{selected_role}', Type='{selected_match_type}', Venue='{selected_venue}', Start='{selected_start_date_str}', End='{selected_end_date_str}'")

    # Default outputs for options/dates in case of early exit
    player_opts = dash.no_update
    player_val = dash.no_update
    match_type_opts = dash.no_update
    venue_opts = dash.no_update
    min_date = dash.no_update
    max_date = dash.no_update
    start_date_out = dash.no_update
    end_date_out = dash.no_update

    # --- Data Loading and Initial Validation ---
    if not player_data_json and triggered_id == 'player-analysis-data-store':
        print("No data found in player-analysis-data-store on initial load.")
        no_data_alert = dbc.Alert("Error: Player analysis data could not be loaded.", color="danger")
        return no_data_alert, [], None, [], [], None, None, None, None
    elif not player_data_json:
         print("Data store is empty, but trigger was not the store itself. Preventing update.")
         no_data_alert = dbc.Alert("Error: Player analysis data is missing.", color="danger")
         return (no_data_alert, player_opts, player_val, match_type_opts, venue_opts,
                 min_date, max_date, start_date_out, end_date_out)

    try:
        # >>> Make sure 'orient' matches how data was stored <<<
        df_page_data = pd.read_json(player_data_json, orient='split') # Assuming 'split' was used
        print(f"Data deserialized successfully. Shape: {df_page_data.shape}")
        if 'start_date' in df_page_data.columns:
            df_page_data['start_date'] = pd.to_datetime(df_page_data['start_date'], errors='coerce')
            original_rows = len(df_page_data)
            df_page_data.dropna(subset=['start_date'], inplace=True)
            rows_dropped = original_rows - len(df_page_data)
            if rows_dropped > 0:
                print(f"Warning: Dropped {rows_dropped} rows with invalid 'start_date' values.")
            if df_page_data.empty:
                 raise ValueError("No valid dates found after conversion.")
            print("start_date column converted and validated.")
        else:
            print("Warning: 'start_date' column missing. Date filtering and timeline plots disabled.")
            # If start_date is critical for core function, you might want to raise an error here instead
            # raise ValueError("'start_date' column is required but missing.")
    except Exception as e:
        print(f"ERROR deserializing or processing date column: {e}")
        error_alert = dbc.Alert(f"Error processing player data: {e}", color="danger")
        return error_alert, [], None, [], [], None, None, None, None

    if df_page_data.empty:
        print("DataFrame is empty after loading/date conversion.")
        empty_alert = dbc.Alert("No player data available.", color="warning")
        return empty_alert, [], None, [], [], None, None, None, None

    # --- Populate Dropdowns & Date Picker (only on data load or if empty) ---
    triggered_by_store = triggered_id == 'player-analysis-data-store'
    options_are_empty = not current_player_options

    min_data_date = df_page_data['start_date'].min() if 'start_date' in df_page_data.columns and not df_page_data.empty else None
    max_data_date = df_page_data['start_date'].max() if 'start_date' in df_page_data.columns and not df_page_data.empty else None
    min_date_str = min_data_date.strftime('%Y-%m-%d') if pd.notna(min_data_date) else None
    max_date_str = max_data_date.strftime('%Y-%m-%d') if pd.notna(max_data_date) else None

    if triggered_by_store or options_are_empty:
        print("Populating dropdown options and date picker range...")
        if 'name' in df_page_data.columns:
            # Filter out potential NaN or None player names before creating options
            valid_player_names = df_page_data["name"].dropna().unique()
            player_choices = sorted(valid_player_names)
            player_opts = [{'label': p, 'value': p} for p in player_choices]
            # Determine default/selected player robustly
            current_selection = selected_player # From input
            if not player_opts: # No valid players
                 player_val = None
            elif options_are_empty: # First load or options were somehow cleared
                player_val = player_opts[0]['value'] # Set to first valid player
            elif current_selection not in player_choices: # Previous selection is no longer valid
                player_val = player_opts[0]['value'] # Reset to first valid player
            else: # Keep current selection if valid
                player_val = current_selection
        else:
            player_opts = [{'label': "Data Error (Name missing)", 'value': "", 'disabled': True}]
            player_val = None

        default_mt_opts = [{'label': 'All Match Types', 'value': DEFAULT_ALL_VALUE}]
        if 'match_type' in df_page_data.columns:
            match_types = sorted([str(mt) for mt in df_page_data["match_type"].dropna().unique()])
            # Filter out common string representations of NaN/None
            match_type_opts = default_mt_opts + [{'label': mt, 'value': mt} for mt in match_types if mt.lower() not in ['nan', 'none', 'null', '']]
        else:
             match_type_opts = default_mt_opts

        default_v_opts = [{'label': 'All Venues', 'value': DEFAULT_ALL_VALUE}]
        if 'venue' in df_page_data.columns:
            venues = sorted([str(v) for v in df_page_data["venue"].dropna().unique()])
            # Filter out common string representations of NaN/None
            venue_opts = default_v_opts + [{'label': v, 'value': v} for v in venues if v.lower() not in ['nan', 'none', 'null', '']]
        else:
            venue_opts = default_v_opts

        # Set Date Picker Range and Default Dates
        min_date = min_date_str
        max_date = max_date_str
        # Only set default dates if they are currently None (don't override user selection)
        if state_start_date is None and min_date_str:
             start_date_out = min_date_str
        if state_end_date is None and max_date_str:
             end_date_out = max_date_str

        print(f"Options populated. Player default/current: {player_val}, Date Range: {min_date} to {max_date}")

        # --- Handling initial load state ---
        # If the trigger was the data store, `player_val` now holds the determined player (first in list or existing valid)
        # We need to use this value for filtering if no player was previously selected (which would be `None`)
        current_player_for_initial_run = player_val
        if triggered_by_store and selected_player is None:
             selected_player = current_player_for_initial_run # Use the determined player for the rest of the function run

        # If after all this, we still have no player selected (e.g., data error, empty player list)
        if selected_player is None:
            return dbc.Alert("Select a player to begin.", color="info"), player_opts, player_val, match_type_opts, venue_opts, min_date, max_date, start_date_out, end_date_out
        else:
             print(f"Proceeding with player: {selected_player}")

    # --- Use the selected or determined player value ---
    current_player = selected_player
    if not current_player:
        print("No player selected or determined.")
        # Return current options/dates but show message
        return (dbc.Alert("Please select a player.", color="info"),
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, # Keep existing options
                dash.no_update, dash.no_update, dash.no_update, dash.no_update) # Keep existing dates

    # --- Data Filtering ---
    print(f"Filtering data for player: {current_player}")
    # Filter based on 'name' column, ensure robustness against case issues if necessary
    # df_page_data['name'] = df_page_data['name'].astype(str) # Ensure string type if needed
    filtered_df = df_page_data[df_page_data['name'] == current_player].copy()

    if filtered_df.empty:
        msg = f"No data found for player: {current_player} in the dataset."
        print(msg)
        # Return updated options/dates from population step, but show message
        return (dbc.Alert(msg, color="warning"), player_opts, player_val, match_type_opts, venue_opts,
                 min_date, max_date, start_date_out, end_date_out)

    # --- Apply Date Filter ---
    date_filter_applied = False
    start_dt, end_dt = None, None
    # Use the dates from the callback Input arguments first, fall back to State if Input is None (e.g. initial load after store)
    current_start_date_str = selected_start_date_str if selected_start_date_str is not None else state_start_date
    current_end_date_str = selected_end_date_str if selected_end_date_str is not None else state_end_date

    if 'start_date' in filtered_df.columns and current_start_date_str and current_end_date_str:
        try:
            # Convert to datetime objects for comparison, normalize to ignore time part
            start_dt = pd.to_datetime(current_start_date_str).normalize()
            end_dt = pd.to_datetime(current_end_date_str).normalize()
            if pd.notna(start_dt) and pd.notna(end_dt):
                print(f"Filtering by Date Range: {start_dt.date()} to {end_dt.date()}")
                # Ensure the comparison is also done on normalized dates
                filtered_df = filtered_df[
                    (filtered_df['start_date'].dt.normalize() >= start_dt) &
                    (filtered_df['start_date'].dt.normalize() <= end_dt)
                ]
                date_filter_applied = True
                # Update output dates to reflect the applied filter (might be redundant if inputs triggered it)
                start_date_out = current_start_date_str
                end_date_out = current_end_date_str
            else:
                 print("Date range not applied, invalid date strings received.")
        except Exception as date_err:
            print(f"Error applying date filter: {date_err}")
            # Optionally: return an error message or just proceed without date filtering

    # --- Apply Categorical Filters ---
    current_match_type = selected_match_type or DEFAULT_ALL_VALUE
    if current_match_type != DEFAULT_ALL_VALUE and 'match_type' in filtered_df.columns:
        print(f"Filtering by Match Type: {current_match_type}")
        # Handle potential type mismatches (e.g., numeric vs string)
        filtered_df = filtered_df[filtered_df['match_type'].astype(str) == str(current_match_type)]

    current_venue = selected_venue or DEFAULT_ALL_VALUE
    if current_venue != DEFAULT_ALL_VALUE and 'venue' in filtered_df.columns:
        print(f"Filtering by Venue: {current_venue}")
        filtered_df = filtered_df[filtered_df['venue'].astype(str) == str(current_venue)]

    # Check if empty *after* filtering
    if filtered_df.empty:
        filter_parts = []
        if current_match_type != DEFAULT_ALL_VALUE: filter_parts.append(f"Match Type='{current_match_type}'")
        if current_venue != DEFAULT_ALL_VALUE: filter_parts.append(f"Venue='{current_venue}'")
        if date_filter_applied: filter_parts.append(f"Dates='{start_dt.date()} to {end_dt.date()}'")
        filter_desc_str = ", ".join(filter_parts) if filter_parts else "the selected criteria"
        msg = f"No data found for {current_player} with {filter_desc_str}."
        print(msg)
        # Return updated options/dates from population step, but show message
        return (dbc.Alert(msg, color="warning"), player_opts, player_val, match_type_opts, venue_opts,
                 min_date, max_date, start_date_out, end_date_out)

    # --- Generate Content Based on Role ---
    output_content = []
    date_range_desc_str = f" ({start_dt.date()} to {end_dt.date()})" if date_filter_applied and pd.notna(start_dt) else " (All Selected Dates)"
    mt_desc = "All" if current_match_type == DEFAULT_ALL_VALUE else current_match_type
    v_desc = "All" if current_venue == DEFAULT_ALL_VALUE else current_venue
    filter_desc = f"Filters: Match Type = {mt_desc}, Venue = {v_desc}{date_range_desc_str}"

    # --- Role: Batsman ---
    if selected_role == 'Batsman':
        print("Generating Batsman Stats...")
        batsman_stats = calculate_batsman_stats(filtered_df.copy()) # Pass copy

        if batsman_stats.get('innings', 0) == 0:
             no_batting_alert = dbc.Alert(f"No batting innings found for {current_player} with the selected filters.", color="info")
             output_content = [
                html.H3(f"{current_player} - Batsman Performance"),
                html.P(filter_desc, className="text-muted small"),
                no_batting_alert
             ]
        else:
            summary_cards = dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Matches", className="card-title"), html.P(f"{batsman_stats.get('matches', 0)}")])), width=6, sm=4, md=2, className="mb-2"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Innings", className="card-title"), html.P(f"{batsman_stats.get('innings', 0)}")])), width=6, sm=4, md=2, className="mb-2"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Runs", className="card-title"), html.P(f"{batsman_stats.get('runs', 0):,}")])), width=6, sm=4, md=2, className="mb-2"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Average", className="card-title"), html.P(f"{batsman_stats.get('average', 0):.2f}" if np.isfinite(batsman_stats.get('average', 0)) else 'N/A')])), width=6, sm=6, md=3, className="mb-2"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Strike Rate", className="card-title"), html.P(f"{batsman_stats.get('strike_rate', 0):.2f}")])), width=6, sm=6, md=3, className="mb-2"),
            ], className="mb-3 text-center")

            # Generate Plots using the filtered data
            fig_ridge = plot_runs_distribution_ridge(filtered_df) # Requires 'start_date'
            fig_sr_order = plot_sr_vs_order(filtered_df)        # Requires 'order_seen', 'match_id'
            fig_runs_pie = plot_run_contribution(batsman_stats)
            fig_dismissal_pie = plot_dismissal_pie(batsman_stats) # Requires 'out_kind'

            output_content = [
                html.H3(f"{current_player} - Batsman Performance"),
                html.P(filter_desc, className="text-muted small"),
                summary_cards,
                html.Hr(),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_ridge), width=12)], className="mb-4"),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_sr_order), width=12)], className="mb-4"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_runs_pie), width=12, md=6, className="mb-3"),
                    dbc.Col(dcc.Graph(figure=fig_dismissal_pie), width=12, md=6, className="mb-3"),
                ]),
            ]

    # --- Role: Bowler ---
    elif selected_role == 'Bowler':
        print("Generating Bowler Stats...")
        bowler_stats = calculate_bowler_stats(filtered_df.copy()) # Pass copy

        # Define aggregation period for bowler timeline plots
        BOWLER_AGG_PERIOD = '1Y' # Example: Yearly aggregation

        if bowler_stats.get('innings_bowled', 0) == 0:
             no_bowling_alert = dbc.Alert(f"No bowling innings found for {current_player} with the selected filters.", color="info")
             output_content = [
                 html.H3(f"{current_player} - Bowler Performance"),
                 html.P(filter_desc, className="text-muted small"),
                 no_bowling_alert
             ]
        else:
            summary_cards = dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Matches Plyd", className="card-title"), html.P(f"{bowler_stats.get('matches', 0)}")])), width=6, sm=4, lg=2, className="mb-2"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Inns Bowled", className="card-title"), html.P(f"{bowler_stats.get('innings_bowled', 0)}")])), width=6, sm=4, lg=2, className="mb-2"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Wickets", className="card-title"), html.P(f"{bowler_stats.get('wickets', 0)}")])), width=6, sm=4, lg=2, className="mb-2"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Average", className="card-title"), html.P(f"{bowler_stats.get('average', 0):.2f}" if np.isfinite(bowler_stats.get('average', 0)) else 'N/A')])), width=6, sm=4, lg=2, className="mb-2"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Economy", className="card-title"), html.P(f"{bowler_stats.get('economy', 0):.2f}" if np.isfinite(bowler_stats.get('economy', 0)) else 'N/A')])), width=6, sm=4, lg=2, className="mb-2"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Strike Rate", className="card-title"), html.P(f"{bowler_stats.get('strike_rate', 0):.2f}" if np.isfinite(bowler_stats.get('strike_rate', 0)) else 'N/A')])), width=6, sm=4, lg=2, className="mb-2"),
            ], className="mb-3 text-center")

            # Generate Plots using the filtered data
            fig_timeline_bowler = plot_bowler_timeline(filtered_df, aggregation_period=BOWLER_AGG_PERIOD) # Requires 'start_date'
            fig_dot_ball_timeline = plot_dot_ball_percentage_timeline(filtered_df, aggregation_period=BOWLER_AGG_PERIOD) # Requires 'start_date', 'dot_balls_as_bowler'
            fig_wicket_pie = plot_wicket_types_pie(bowler_stats) # Requires 'bowled_done', 'lbw_done' etc. from stats calculation

            output_content = [
                html.H3(f"{current_player} - Bowler Performance"),
                html.P(filter_desc, className="text-muted small"),
                summary_cards,
                html.Hr(),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_timeline_bowler), width=12)], className="mb-3"),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_dot_ball_timeline), width=12)], className="mb-3"),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_wicket_pie), width=12, md=6, className="mb-3")]) # Center pie maybe?
            ]

    # --- Invalid Role ---
    else:
        output_content = dbc.Alert("Invalid role selected.", color="danger")


    # Return the generated content AND the potentially updated dropdown/date picker states
    # Player options/value etc. might have been updated if triggered by store/options empty
    # Date range min/max/start/end might have been updated
    return (html.Div(output_content), player_opts, player_val, match_type_opts, venue_opts,
            min_date, max_date, start_date_out, end_date_out)

# --- END OF FILE player_analysis.py ---