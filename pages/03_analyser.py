# --- IMPORTS and HELPER FUNCTIONS remain the same ---
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import warnings
import pycountry
import json
import datetime  # Import datetime for date handling

# --- Helper Functions (get_country_code, get_all_country_iso3, etc...) ---
# Assume they are correctly defined here as in the previous version (pasted below for completeness)


def get_country_code(name, code_type='alpha_3'):
    # (Keep the function definition)
    overrides_alpha3 = {"United Arab Emirates": "ARE", "Scotland": "GBR", "United States of America": "USA", "Netherlands": "NLD", "England": "GBR", "Ireland": "IRL", "West Indies": "JAM",  # Use Jamaica as a representative ISO for map, actual data uses 'West Indies'
                        "Hong Kong": "HKG", "Papua New Guinea": "PNG", "Bermuda": "BMU", "Afghanistan": "AFG", "Bangladesh": "BGD", "India": "IND", "Pakistan": "PAK", "Sri Lanka": "LKA", "Australia": "AUS", "New Zealand": "NZL", "South Africa": "ZAF", "Zimbabwe": "ZWE", "Kenya": "KEN", "Canada": "CAN", "Namibia": "NAM", "Nepal": "NPL", "Oman": "OMN", "ICC World XI": None,  # No specific country
                        # Historically relevant but no single modern ISO
                        "Asia XI": None, "Africa XI": None, "East Africa": None,
                        }
    # Standardize common variations before lookup/override
    name_map = {
        'U.A.E.': 'United Arab Emirates', 'UAE': 'United Arab Emirates',
        'P.N.G.': 'Papua New Guinea', 'USA': 'United States of America',
        'West Indies Cricket Board': 'West Indies',
        # Exclude A teams explicitly
        'England Lions': None, 'Ireland A': None, 'South Africa A': None,
    }
    std_name = name.strip() if isinstance(name, str) else name
    std_name = name_map.get(std_name, std_name)  # Apply standardization

    if not std_name:
        return None
    if std_name in overrides_alpha3:
        return overrides_alpha3[std_name]
    try:
        country = pycountry.countries.lookup(std_name)
        return country.alpha_3 if code_type == 'alpha_3' else country.alpha_2
    except LookupError:
        try:
            # Handle common fuzzy matches if direct lookup fails
            if 'england' in std_name.lower():
                return "GBR"
            if 'scotland' in std_name.lower():
                return "GBR"  # Map Scotland to GBR for plotting simplicity if needed
            if 'ireland' in std_name.lower():
                return "IRL"  # Ensure Ireland maps correctly

            results = pycountry.countries.search_fuzzy(std_name)
            if results:
                return results[0].alpha_3 if code_type == 'alpha_3' else results[0].alpha_2
            else:
                print(
                    f"Warning: Could not find ISO code for '{name}' (standardized: '{std_name}')")
                return None  # Explicitly return None if fuzzy search fails
        except LookupError:
            print(f"Warning: Fuzzy search failed for '{std_name}'")
            return None
    except Exception as e:
        print(f"Error in get_country_code for '{name}': {e}")
        return None


def get_all_country_iso3():
    # (Keep the function definition)
    codes = set()
    for country in pycountry.countries:
        try:
            codes.add(country.alpha_3)
        except AttributeError:
            # print(f"Warning: Skipping country without alpha_3: {getattr(country, 'name', 'Unknown')}")
            continue
    # Add codes from overrides that might not be in pycountry
    override_codes = {"ARE", "GBR", "USA", "NLD", "IRL", "JAM", "HKG", "PNG", "BMU", "AFG",
                      "BGD", "IND", "PAK", "LKA", "AUS", "NZL", "ZAF", "ZWE", "KEN", "CAN", "NAM", "NPL", "OMN"}
    codes.update(override_codes)
    return codes


def export_pdf(stats_text, filename="player_stats.pdf"):
    # (Keep the function definition)
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text = c.beginText(inch, height - inch)
    text.setFont("Helvetica", 10)
    for line in stats_text.split("\n"):
        text.textLine(line)
    c.drawText(text)
    c.save()
    buffer.seek(0)
    return buffer


def plot_dismissal_pie(df, title="Dismissal Analysis"):
    # (Keep the function definition - check columns exist etc.)
    if df is None or df.empty or not all(col in df.columns for col in ['Count', 'Dismissal Type']):
        return go.Figure(layout=go.Layout(title=go.layout.Title(text=f"{title}<br>(Data missing)")))
    if df['Count'].sum() == 0:
        return go.Figure(layout=go.Layout(title=go.layout.Title(text=f"{title}<br>(No dismissals)")))
    try:
        fig = px.pie(df, names='Dismissal Type', values='Count', title=title,
                     hole=0.35, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textposition='inside',
                          textinfo='percent+label', pull=[0.05]*len(df))
        fig.update_layout(showlegend=False, margin=dict(t=50, b=0, l=0, r=0))
        return fig
    except Exception as e:
        print(f"Error creating dismissal pie: {e}")
        return go.Figure(layout=go.Layout(title=go.layout.Title(text=f"{title}<br>(Plotting error)")))


def plot_run_contribution_pie(runs_4s, runs_6s, other_runs, title="Run Scoring Breakdown"):
    # (Keep the function definition)
    data = {'Run Type': ['Runs from 4s', 'Runs from 6s', 'Other Runs (1s, 2s, 3s)'], 'Runs': [
        runs_4s, runs_6s, other_runs]}
    df_runs = pd.DataFrame(data)
    df_runs = df_runs[df_runs['Runs'] > 0]
    if df_runs.empty:
        return go.Figure(layout=go.Layout(title=go.layout.Title(text=f"{title}<br>(No runs scored)")))
    try:
        fig = px.pie(df_runs, names='Run Type', values='Runs', title=title,
                     hole=0.35, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition='outside',
                          textinfo='percent+value', pull=[0.05]*len(df_runs))
        fig.update_layout(showlegend=False, margin=dict(t=50, b=0, l=0, r=0))
        return fig
    except Exception as e:
        print(f"Error creating run contribution pie: {e}")
        return go.Figure(layout=go.Layout(title=go.layout.Title(text=f"{title}<br>(Plotting error)")))


def create_comparison_gauge(value, reference, title, lower_bound=0, upper_bound=None, higher_is_better=True):
    # (Keep the function definition - check for NaN etc.)
    value = float(value) if value is not None and not np.isnan(
        value) else np.nan  # Handle NaN explicitly
    reference = float(reference) if reference is not None and not np.isnan(
        reference) else np.nan  # Handle NaN explicitly

    # Ensure value and reference are finite for calculations if not NaN
    value_calc = value if pd.notna(value) and np.isfinite(value) else np.nan
    ref_calc = reference if pd.notna(
        reference) and np.isfinite(reference) else np.nan

    if np.isnan(value_calc):  # If primary value is NaN, can't plot gauge meaningfully
        return go.Figure(layout=go.Layout(title=go.layout.Title(text=f"{title}<br>(N/A vs Opp.)"), height=250, margin=dict(t=60, b=10, l=30, r=30)))

    # Determine max value for scaling, ignoring NaN/inf
    valid_values = [v for v in [value_calc, ref_calc]
                    if pd.notna(v) and np.isfinite(v)]
    # Default to 0 if no valid values
    max_val = max(valid_values) if valid_values else 0

    if upper_bound is None:
        # Set upper bound slightly above the max of value or reference
        # Add buffer, handle zero case
        upper_bound = max_val * 1.5 if max_val > 0 else (lower_bound + 10)
    else:
        # Ensure provided upper_bound is at least the max observed value
        upper_bound = max(upper_bound, max_val)

    # Ensure upper bound is strictly greater than lower bound
    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 1  # Minimal range if bounds collapse

    # Handle case where reference might be NaN for delta display
    delta_config = {'reference': ref_calc, 'relative': False,
                    'valueformat': '.2f'} if pd.notna(ref_calc) else None

    try:
        fig = go.Figure(go.Indicator(
            # Conditionally add delta
            mode="gauge+number" + ("+delta" if delta_config else ""),
            value=value_calc,  # Use the validated value
            title={'text': f"{title}<br>(vs Career Period)"},
            delta=delta_config,  # Pass config or None
            gauge={
                'axis': {'range': [lower_bound, upper_bound]},
                'bar': {'color': '#1f77b4'},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [  # Optional: Add color steps if desired
                    # {'range': [lower_bound, ref_calc * 0.8], 'color': 'lightcoral'} if pd.notna(ref_calc) else {},
                    # {'range': [ref_calc * 1.2, upper_bound], 'color': 'lightgreen'} if pd.notna(ref_calc) else {},
                ],
                'threshold': {  # Show reference line if valid
                    'line': {'color': '#ff7f0e', 'width': 4},
                    'thickness': 0.75,
                    'value': ref_calc} if pd.notna(ref_calc) else None
            }
        ))
        fig.update_layout(height=250, margin=dict(t=60, b=10, l=30, r=30))
        return fig
    except Exception as e:
        print(f"Error creating gauge '{title}': {e}")
        return go.Figure(layout=go.Layout(title=go.layout.Title(text=f"{title}<br>(Plotting error)"), height=250))


def create_comparison_bar(value_opp, value_career, label_opp, label_career, title):
    # (Keep the function definition - format value etc.)
    # Handle potential NaN or infinite values before converting to float
    value_opp = float(value_opp) if pd.notna(
        value_opp) and np.isfinite(value_opp) else 0.0
    value_career = float(value_career) if pd.notna(
        value_career) and np.isfinite(value_career) else 0.0

    def format_value(v):
        if isinstance(v, (int, float)):
            # Format floats with precision, integers without decimals
            if isinstance(v, float):
                return f'{v:,.2f}'
            else:
                return f'{v:,}'  # Integer formatting
        return str(v)  # Fallback for other types

    labels = [label_opp, label_career]
    values = [value_opp, value_career]
    colors = ['#1f77b4', '#ff7f0e']
    try:
        fig = go.Figure(data=[go.Bar(x=labels, y=values, text=[format_value(
            v) for v in values], textposition='auto', marker_color=colors)])
        fig.update_layout(title=title, xaxis_title="Context", yaxis_title="Value", height=300, margin=dict(
            t=50, b=30, l=30, r=30), yaxis=dict(rangemode='tozero'))
        return fig
    except Exception as e:
        print(f"Error creating bar chart '{title}': {e}")
        return go.Figure(layout=go.Layout(title=go.layout.Title(text=f"{title}<br>(Plotting error)"), height=300))


# --- Constants ---
ALLOWED_FORMATS = ["ODI", "T20", "T20I", "Test"]  # Match formats used in app
COLOR_PLAYED = "#1f77b4"  # Blue for played
COLOR_NOT_PLAYED = "#D3D3D3"  # Lighter grey for not played
# Light Salmon for played but outside selected date range (optional, can simplify)
COLOR_PLAYED_OUTSIDE_RANGE = "#FFA07A"

# --- Register Page ---
dash.register_page(__name__, name='Player Analyzer',
                   path='/analyzer', title='Player Analyzer')  # Add title

# --- Default Placeholder Message ---
default_placeholder = dbc.Alert(
    "Select a player, format, and date range, then click a played (blue) country on the map to view detailed statistics.",
    color="info",
    className="text-center mt-4",  # Add margin top
    id="analyzer-placeholder-message"  # Give it an ID if we want to update its text
)

# --- Layout Function ---


def layout():
    print("Generating Analyzer page layout...")
    initial_player_opts = [
        {'label': "Loading players...", 'value': "", 'disabled': True}]
    today = datetime.date.today()
    initial_start_date = today - \
        datetime.timedelta(days=5*365)  # Default to 5 years ago

    return dbc.Container([
        dcc.Store(id='analyzer-iso-map-store', storage_type='memory', data={}),
        dcc.Store(id='analyzer-selected-opposition-store',
                  storage_type='memory', data=None),
        # Store for filtered data used by downloads (optional optimization)
        # dcc.Store(id='analyzer-filtered-data-store', storage_type='memory'),
        dcc.Download(id="analyzer-download-csv"),
        dcc.Download(id="analyzer-download-pdf"),

        dbc.Row(dbc.Col(html.H2("Player Performance Analyzer"), width=12)),

        dbc.Row([
            # Control Column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Selection Criteria"),
                    dbc.CardBody([
                         html.Div([
                             html.Label("1. Select Player:",
                                        className="fw-bold"),
                             dcc.Dropdown(id='analyzer-player-dropdown', options=initial_player_opts,
                                          value=None, placeholder="Select player...", clearable=False),
                         ], className="mb-3"),
                         html.Div([
                             html.Label("2. Select Format:",
                                        className="fw-bold"),
                             dbc.RadioItems(id='analyzer-format-radio', options=[{'label': fmt, 'value': fmt} for fmt in ALLOWED_FORMATS],
                                            value=ALLOWED_FORMATS[0], inline=True, labelStyle={'margin-right': '15px'}, inputClassName="me-1"),
                         ], className="mb-3"),
                         # --- ADDED DATE PICKER ---
                         html.Div([
                             html.Label("3. Select Date Range:",
                                        className="fw-bold"),
                             dcc.Loading(  # Add loading indicator around date picker
                                 id="loading-date-picker",
                                 type="circle",
                                 children=[
                                     dcc.DatePickerRange(
                                         id='analyzer-date-picker-range',
                                         min_date_allowed=initial_start_date -
                                         # Default wide range
                                         datetime.timedelta(days=20*365),
                                         max_date_allowed=today,
                                         start_date=None,  # Set by callback
                                         end_date=None,   # Set by callback
                                         display_format='YYYY-MM-DD',
                                         className="d-block",  # Make it block level for better spacing
                                         disabled=True  # Initially disabled until player/format selected
                                     )
                                 ]
                             )
                         ], className="mb-3"),
                         dbc.Button("Submit", id='analyzer-submit-button', n_clicks=0,
                                    color='primary', className="mt-2", disabled=True),
                         # --- END OF DATE PICKER ---
                         html.Hr(),
                         html.P(["4. Click a country on the map to see stats against that opponent within the selected period."],
                                className="text-muted fst-italic"),
                         html.P([
                             html.Span("■", style={
                                       'color': COLOR_PLAYED, 'fontSize': '20px'}), " Opponent Played (in period)", html.Br(),
                             html.Span("■", style={
                                       'color': COLOR_NOT_PLAYED, 'fontSize': '20px'}), " Opponent Not Played (in period)"
                             # Optional: Add legend for COLOR_PLAYED_OUTSIDE_RANGE if used
                         ], className="small")
                         ])
                ])
            ], width=12, lg=4, className="mb-3 mb-lg-0"),

            # Map Column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Opposition from Map"),
                    dbc.CardBody(
                         dcc.Loading(
                             id="loading-map", type="circle",
                             children=[dcc.Graph(
                                 id='analyzer-map-graph', config={'displayModeBar': False}, style={'height': '450px'})]
                         )
                         )
                ])
            ], width=12, lg=8),
        ], className="mb-4"),

        # Stats Output Area - Will contain either placeholder or stats
        html.Div(id='analyzer-stats-output-area',
                 children=default_placeholder)  # Start with placeholder

    ], fluid=True)

# --- Utility to Deserialize Data from Store ---


def deserialize_data(stored_data):
    # (Keep the function definition - enhanced error checking)
    if stored_data is None:
        print("Warning: No data found in store.")
        return None
    try:
        df = pd.read_json(stored_data, orient='split')
        if df.empty:
            print("Warning: Deserialized DataFrame is empty.")
            return df  # Return empty df, let downstream handle it
        # Ensure start_date is datetime AFTER deserialization
        if 'start_date' in df.columns:
            try:
                # Check if it's already datetime (less likely after JSON)
                if not pd.api.types.is_datetime64_any_dtype(df['start_date']):
                    # Attempt conversion from string/timestamp
                    df['start_date'] = pd.to_datetime(
                        df['start_date'], errors='coerce')
                    if df['start_date'].isnull().any():
                        print(
                            "Warning: Some 'start_date' values failed conversion during deserialization.")
                # Optional: Convert to date if time part is irrelevant and consistently midnight
                # df['start_date'] = df['start_date'].dt.date
            except Exception as e:
                print(
                    f"Warning: Could not convert 'start_date' back to datetime after deserialization: {e}")
                # Decide if failure is critical - maybe drop rows or return None?
                df.drop(columns=['start_date'], inplace=True,
                        errors='ignore')  # Or handle differently
        else:
            print("Warning: 'start_date' column not found after deserialization.")

        print(f"Data deserialized successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR: Failed to deserialize data from store: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Callbacks ---

# Callback 0: Populate Player Dropdown (Remains the same)
@callback(
    Output('analyzer-player-dropdown', 'options'),
    Output('analyzer-player-dropdown', 'value'),
    Input('main-data-store', 'data'),
    prevent_initial_call=False  # Run on app load
)
def update_analyzer_player_options(stored_data):
    print("Callback triggered: update_analyzer_player_options")
    df = deserialize_data(stored_data)
    if df is None or df.empty:
        print("  - Data unavailable for player options.")
        return ([{'label': "Data loading failed", 'value': "", 'disabled': True}], None)
    if 'name' not in df.columns:
        print("  - 'name' column missing for player options.")
        return ([{'label': "Player name column missing", 'value': "", 'disabled': True}], None)
    try:
        # Use dropna() and unique() robustly
        player_choices = sorted(df["name"].dropna().unique())
        if not player_choices:
            print("  - No unique player names found.")
            return ([{'label': "No players found", 'value': "", 'disabled': True}], None)

        options = [{'label': p, 'value': p} for p in player_choices]
        print(f"  - Generated {len(options)} player options.")
        # Select first player as default, ensure options list is not empty
        default_value = options[0]['value'] if options else None
        return options, default_value
    except Exception as e:
        print(f"  - Error generating player options: {e}")
        return ([{'label': "Error processing players", 'value': "", 'disabled': True}], None)


# **NEW** Callback 0.5: Update Date Picker Range based on Player/Format
@callback(
    Output('analyzer-date-picker-range', 'min_date_allowed'),
    Output('analyzer-date-picker-range', 'max_date_allowed'),
    Output('analyzer-date-picker-range', 'start_date'),
    Output('analyzer-date-picker-range', 'end_date'),
    Output('analyzer-date-picker-range', 'disabled'),
    Input('analyzer-player-dropdown', 'value'),
    Input('analyzer-format-radio', 'value'),
    Input('main-data-store', 'data'),  # React to data load as well
    prevent_initial_call=True  # Don't run on initial load before player is selected
)
def update_analyzer_date_picker_range(selected_player, selected_format, stored_data):
    print(
        f"Callback triggered: update_analyzer_date_picker_range (Player: {selected_player}, Format: {selected_format})")
    if not selected_player or not selected_format:
        print("  - Player or format not selected. Disabling date picker.")
        today = datetime.date.today()
        # Default past range
        min_allowed = today - datetime.timedelta(days=30*365)
        return min_allowed, today, None, None, True  # Disable picker

    df = deserialize_data(stored_data)
    if df is None or df.empty or 'start_date' not in df.columns or df['start_date'].isnull().all():
        print(
            "  - Data unavailable or 'start_date' missing/invalid. Disabling date picker.")
        today = datetime.date.today()
        min_allowed = today - datetime.timedelta(days=30*365)
        return min_allowed, today, None, None, True  # Disable picker

    # Filter data for the selected player and format
    player_format_df = df[
        (df["name"] == selected_player) &
        (df["match_type"] == selected_format) &
        df['start_date'].notna()  # Ensure we only consider rows with valid dates
    ]

    if player_format_df.empty:
        print(
            f"  - No data found for {selected_player} in {selected_format}. Disabling date picker.")
        today = datetime.date.today()
        min_allowed = today - datetime.timedelta(days=30*365)
        return min_allowed, today, None, None, True  # Disable picker

    try:
        # Find the actual min and max dates from the filtered data
        min_date = player_format_df['start_date'].min().date()
        max_date = player_format_df['start_date'].max().date()
        print(f"  - Date range for selection: {min_date} to {max_date}")

        # Set the picker's allowed range and initial selection to the full range
        return min_date, max_date, min_date, max_date, False  # Enable picker

    except Exception as e:
        print(f"  - Error calculating date range: {e}")
        today = datetime.date.today()
        min_allowed = today - datetime.timedelta(days=30*365)
        return min_allowed, today, None, None, True  # Disable on error


# Callback 1: Update Map (Takes Date Range Input)
@callback(
    Output('analyzer-map-graph', 'figure'),
    Output('analyzer-iso-map-store', 'data'),
    Output('analyzer-selected-opposition-store', 'data',
           allow_duplicate=True),  # Reset opposition
    Input('analyzer-submit-button', 'n_clicks'),
    State('analyzer-player-dropdown', 'value'),
    State('analyzer-format-radio', 'value'),
    State('analyzer-date-picker-range', 'start_date'),  # *** ADDED INPUT ***
    State('analyzer-date-picker-range', 'end_date'),   # *** ADDED INPUT ***
    State('main-data-store', 'data'),
    prevent_initial_call=True
)
def analyzer_update_map(n_clicks, selected_player, selected_format, start_date, end_date, stored_data):
    if not n_clicks:
        raise PreventUpdate
    """Generates the map based on player, format, AND date range."""
    print(
        f"Callback triggered: analyzer_update_map (Player: {selected_player}, Format: {selected_format}, Range: {start_date} to {end_date})")

    # Basic validation
    if not selected_player or not selected_format:
        print("  - Player or format missing.")
        return go.Figure(layout=go.Layout(title="Select Player and Format")), {}, None
    if not start_date or not end_date:
        print("  - Date range missing.")
        # Potentially return map with all opponents but grayed out? Or just a message.
        return go.Figure(layout=go.Layout(title="Select Date Range")), {}, None

    df_full = deserialize_data(stored_data)
    if df_full is None or df_full.empty:
        print("  - Main data not loaded.")
        return go.Figure(layout=go.Layout(title="Data not loaded")), {}, None
    if 'start_date' not in df_full.columns:
        print("  - 'start_date' column missing.")
        return go.Figure(layout=go.Layout(title="Date information missing in data")), {}, None

    # --- Filter Data including Date Range ---
    try:
        # Ensure start_date is datetime if not already done by deserialize_data
        if not pd.api.types.is_datetime64_any_dtype(df_full['start_date']):
            df_full['start_date'] = pd.to_datetime(
                df_full['start_date'], errors='coerce')

        # Normalize to midnight for date comparison
        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()

        mask = (
            (df_full["name"] == selected_player) &
            (df_full["match_type"] == selected_format) &
            (df_full['start_date'].notna()) &
            (df_full['start_date'] >= start_dt) &
            (df_full['start_date'] <= end_dt)
        )
        player_df = df_full[mask].copy()

    except Exception as e:
        print(f"  - Error during date filtering: {e}")
        return go.Figure(layout=go.Layout(title="Error processing date filter")), {}, None

    if player_df.empty:
        print(
            f"  - No {selected_format} data found for {selected_player} within {start_date} to {end_date}.")
        # Create a map showing no played opponents
        all_iso3_codes = get_all_country_iso3()
        world_df = pd.DataFrame(list(all_iso3_codes), columns=['iso_code'])
        world_df['PlayedAgainst'] = False
        world_df['HoverName'] = world_df['iso_code'].apply(
            lambda x: f"Unknown ({x})")  # Basic hover
        world_df['TotalRuns'] = 0
        world_df['Innings'] = 0
        fig_map_empty = px.choropleth(world_df, locations="iso_code", locationmode="ISO-3", color_discrete_map={True: COLOR_PLAYED, False: COLOR_NOT_PLAYED}, color="PlayedAgainst", hover_name="HoverName", hover_data={
                                      "iso_code": False, "PlayedAgainst": False, "TotalRuns": False, "Innings": False}, custom_data=['iso_code'])
        fig_map_empty.update_layout(title=f"No opponents played by {selected_player} ({selected_format}) between {start_date} and {end_date}", title_x=0.5, showlegend=False, geo=dict(
            showframe=False, showcoastlines=False, projection_type='natural earth', bgcolor='rgba(0,0,0,0)', landcolor=COLOR_NOT_PLAYED), margin={"r": 0, "t": 40, "l": 0, "b": 0})
        return fig_map_empty, {}, None  # Return empty map, empty iso map, reset opposition

    # --- Continue with Map Generation (using the filtered player_df) ---
    if 'opposition_team' not in player_df.columns:
        print("  - Opposition data missing.")
        return go.Figure(layout=go.Layout(title="Opposition data missing")), {}, None

    # Get ISO codes for opponents *within the date range*
    player_df["iso_code"] = player_df["opposition_team"].apply(
        lambda x: get_country_code(x, 'alpha_3'))
    valid_opps = player_df.dropna(subset=['iso_code'])

    # Create mapping from ISO to Opposition Name *only for played opponents*
    iso_to_opposition_map = {}
    if not valid_opps.empty:
        iso_to_opposition_map = valid_opps.groupby(
            'iso_code')['opposition_team'].first().to_dict()
        print(
            f"  - Opponents played in period (ISO map): {iso_to_opposition_map}")

    # Aggregate data for hover info (based on filtered data)
    agg_data = player_df.groupby('iso_code', as_index=False).agg(
        Innings=('match_id', 'nunique'),  # Count unique matches as innings
        TotalRuns=('runs_scored', 'sum')
    ).dropna(subset=['iso_code'])

    # Create the world map base
    all_iso3_codes = get_all_country_iso3()
    world_df = pd.DataFrame(list(all_iso3_codes), columns=['iso_code'])

    # Merge aggregated stats for played opponents
    if not agg_data.empty:
        world_df = world_df.merge(agg_data, on='iso_code', how='left')
    else:
        # Ensure columns exist even if no matches played in period
        world_df['Innings'] = 0
        world_df['TotalRuns'] = 0.0

    # Determine played status and fill missing values
    world_df['PlayedAgainst'] = world_df['iso_code'].isin(
        iso_to_opposition_map.keys())  # Played if in our map
    world_df['HoverName'] = world_df['iso_code'].apply(
        lambda code: iso_to_opposition_map.get(
            code, f"Not Played ({code})") if pd.notna(code) else "N/A"
    )
    world_df['TotalRuns'] = world_df['TotalRuns'].fillna(0).astype(int)
    world_df['Innings'] = world_df['Innings'].fillna(0).astype(int)

    # --- Generate Choropleth ---
    try:
        fig_map = px.choropleth(
            world_df,
            locations="iso_code",
            locationmode="ISO-3",
            color="PlayedAgainst",  # Color by played status
            color_discrete_map={True: COLOR_PLAYED, False: COLOR_NOT_PLAYED},
            hover_name="HoverName",
            hover_data={
                "iso_code": False,  # Don't show iso_code in main hover
                "PlayedAgainst": False,  # Already indicated by color/name
                "TotalRuns": ':,.0f',  # Show runs if played
                "Innings": True      # Show innings if played
            },
            custom_data=['iso_code']  # Pass ISO code for click events
        )
        fig_map.update_layout(
            title=f"Opponents Played by {selected_player} ({selected_format}) between {start_date} and {end_date}",
            title_x=0.5,
            showlegend=False,
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='natural earth',
                bgcolor='rgba(0,0,0,0)',
                # Land color determined by 'PlayedAgainst' status
                # Light borders between subunits
                subunitcolor='rgba(255,255,255,0.5)'
            ),
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            coloraxis_showscale=False  # Hide color scale for discrete colors
        )
        # Ensure land is colored correctly even if not explicitly in data using the base map color
        fig_map.update_geos(landcolor=COLOR_NOT_PLAYED)  # Base color for land
        fig_map.update_traces(
            marker_line_width=0.5,
            marker_line_color='white',
            selector=dict(type='choropleth')
        )

        print("  - Map generated successfully.")
        # Return the map figure, the mapping data, and None to reset the selected opposition store
        return fig_map, iso_to_opposition_map, None
    except Exception as e:
        print(f"  - Error creating choropleth map: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure(layout=go.Layout(title="Error generating map")), {}, None


# Callback 2: Update Selected Opposition based on Map Click (Remains the same logic, but relies on map data)
@callback(
    Output('analyzer-selected-opposition-store', 'data'),
    Input('analyzer-map-graph', 'clickData'),
    # Use the map generated by the previous callback
    State('analyzer-iso-map-store', 'data'),
    prevent_initial_call=True
)
def analyzer_update_selected_opposition(clickData, iso_map_data):
    if clickData is None or not iso_map_data:
        # No click or no valid map data (e.g., player hasn't played anyone)
        # print("Preventing update for selected opposition - no click or map data.")
        raise PreventUpdate

    try:
        clicked_iso = None
        # Extract ISO code robustly from clickData
        point = clickData['points'][0]
        if 'customdata' in point and point['customdata']:
            clicked_iso = point['customdata'][0]
        # Fallback if customdata isn't there (should be)
        elif 'location' in point:
            clicked_iso = point['location']

        if clicked_iso is None:
            print("  - Could not extract ISO code from click data.")
            raise PreventUpdate

        print(
            f"Callback triggered: analyzer_update_selected_opposition (Clicked ISO: {clicked_iso})")

        # Check if the clicked ISO corresponds to an opponent actually played *in the selected period*
        if clicked_iso in iso_map_data:
            opposition_name = iso_map_data[clicked_iso]
            print(
                f"  - Resolved opposition (played in period): {opposition_name}")
            return opposition_name  # Set the selected opposition
        else:
            # Clicked on a country not played in the selected period (or an invalid area)
            print(
                f"  - Clicked ISO '{clicked_iso}' not found in playable opponents for the period.")
            # Decide behaviour: either prevent update or explicitly clear selection
            return None  # Clear selection if a non-played country is clicked

    except (KeyError, IndexError, TypeError) as e:
        print(f"  - Error processing map click data: {e}")
        return None  # Clear selection on error


# Callback 3: Update Stats Display Area (Uses Date Range State)
@callback(
    Output('analyzer-stats-output-area', 'children'),
    # Triggered by map click (or reset)
    Input('analyzer-selected-opposition-store', 'data'),
    State('analyzer-player-dropdown', 'value'),
    State('analyzer-format-radio', 'value'),
    State('analyzer-date-picker-range', 'start_date'),  # *** ADDED STATE ***
    State('analyzer-date-picker-range', 'end_date'),   # *** ADDED STATE ***
    State('main-data-store', 'data'),
    prevent_initial_call=True
)
def analyzer_update_stats_area(selected_opposition, selected_player, selected_format, start_date, end_date, stored_data):
    """Generates and displays the stats based on player, format, opposition AND date range."""
    print(
        f"Callback triggered: analyzer_update_stats_area (Opposition: {selected_opposition}, Period: {start_date} to {end_date})")

    # If opposition is cleared (None), show the placeholder message
    if not selected_opposition:
        print("  - No opposition selected. Showing placeholder.")
        return default_placeholder  # Defined outside layout

    # Validate other inputs
    if not selected_player or not selected_format:
        print("  - Player or format missing. Cannot generate stats.")
        return dbc.Alert("Player or Format not selected.", color="warning")
    if not start_date or not end_date:
        print("  - Date range missing. Cannot generate stats.")
        return dbc.Alert("Date range not selected.", color="warning")

    df_full = deserialize_data(stored_data)
    if df_full is None or df_full.empty:
        print("  - Stats update failed: Main data not available.")
        return dbc.Alert("Error: Could not load data for statistics.", color="danger")
    if 'start_date' not in df_full.columns:
        print("  - 'start_date' column missing.")
        return dbc.Alert("Error: Date information missing in data.", color="danger")

    # --- Filter Data including Date Range ---
    try:
        # Ensure start_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_full['start_date']):
            df_full['start_date'] = pd.to_datetime(
                df_full['start_date'], errors='coerce')

        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()

        # Create the date mask
        date_mask = (
            df_full['start_date'].notna() &
            (df_full['start_date'] >= start_dt) &
            (df_full['start_date'] <= end_dt)
        )

        # Filter for stats vs opponent within the date range
        country_mask = (
            (df_full["name"] == selected_player) &
            (df_full["match_type"] == selected_format) &
            (df_full["opposition_team"] == selected_opposition) &
            date_mask  # Apply date mask
        )
        country_stats_df = df_full[country_mask].copy()

        # Filter for career stats within the date range (for comparison)
        career_mask = (
            (df_full["name"] == selected_player) &
            (df_full["match_type"] == selected_format) &
            date_mask  # Apply date mask
        )
        # Use this for comparison gauges/bars
        career_stats_df_period = df_full[career_mask].copy()

    except KeyError as e:
        print(f"  - Error filtering data: Missing column {e}")
        return dbc.Alert(f"Error: Data is missing required column ({e}).", color="danger")
    except Exception as e:
        print(f"  - Error filtering data: {e}")
        return dbc.Alert(f"Error filtering data: {e}", color="danger")

    # --- Validate Filtered Data ---
    if country_stats_df.empty:
        msg = f"No match data found for {selected_player} vs {selected_opposition} in {selected_format} between {start_date} and {end_date}."
        print(f"  - {msg}")
        return dbc.Alert(msg, color="warning")
    if career_stats_df_period.empty:
        # This shouldn't happen if country_stats_df is not empty, but check anyway
        msg = f"Data inconsistency: No overall {selected_format} career data found for {selected_player} in the period {start_date} to {end_date}."
        print(f"  - {msg}")
        return dbc.Alert(msg, color="danger")

    # --- Calculate Statistics (based on date-filtered data) ---
    print("  - Calculating statistics for the period...")
    try:
        # Stats vs Opponent (in period)
        # Use match_id for unique matches
        opp_matches = country_stats_df['match_id'].nunique()
        opp_total_runs = int(country_stats_df["runs_scored"].sum())
        opp_balls_faced = int(country_stats_df["balls_faced"].sum())
        opp_outs = int(country_stats_df["player_out"].sum())
        opp_bat_avg = opp_total_runs / opp_outs if opp_outs > 0 else np.nan
        opp_bat_sr = (opp_total_runs / opp_balls_faced) * \
            100 if opp_balls_faced > 0 else 0.0
        opp_fours = int(country_stats_df["fours_scored"].sum())
        opp_sixes = int(country_stats_df["sixes_scored"].sum())
        opp_wickets = int(country_stats_df["wickets_taken"].sum())
        opp_balls_bowled = int(country_stats_df["balls_bowled"].sum())
        opp_runs_conceded = int(country_stats_df["runs_conceded"].sum())
        opp_bowl_avg = opp_runs_conceded / opp_wickets if opp_wickets > 0 else np.nan
        opp_bowl_econ = (opp_runs_conceded / opp_balls_bowled) * \
            6 if opp_balls_bowled > 0 else np.nan  # Handle division by zero for econ

        # Career Stats (in period - for comparison)
        car_period_matches = career_stats_df_period['match_id'].nunique()
        car_period_total_runs = int(
            career_stats_df_period["runs_scored"].sum())
        car_period_balls_faced = int(
            career_stats_df_period["balls_faced"].sum())
        car_period_outs = int(career_stats_df_period["player_out"].sum())
        car_period_bat_avg = car_period_total_runs / \
            car_period_outs if car_period_outs > 0 else np.nan
        car_period_bat_sr = (car_period_total_runs / car_period_balls_faced) * \
            100 if car_period_balls_faced > 0 else 0.0
        car_period_wickets = int(career_stats_df_period["wickets_taken"].sum())
        car_period_balls_bowled = int(
            career_stats_df_period["balls_bowled"].sum())
        car_period_runs_conceded = int(
            career_stats_df_period["runs_conceded"].sum())
        car_period_bowl_avg = car_period_runs_conceded / \
            car_period_wickets if car_period_wickets > 0 else np.nan
        car_period_bowl_econ = (car_period_runs_conceded / car_period_balls_bowled) * \
            6 if car_period_balls_bowled > 0 else np.nan  # Handle division by zero

        print("  - Statistics calculated successfully.")
    except Exception as e:
        print(f"  - Error calculating statistics: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Error calculating statistics: {e}", color="danger")

    # --- Create Visualizations (using calculated stats) ---
    # Dismissal analysis vs opponent (in period)
    dismissal_counts = pd.DataFrame()
    if 'out_kind' in country_stats_df.columns:
        # Filter for actual dismissals ('player_out' == 1) and count 'out_kind'
        dismissals_df = country_stats_df[country_stats_df["player_out"] == 1].copy(
        )
        if not dismissals_df.empty:
            # Handle potential 'not out' or NaN in 'out_kind' even if 'player_out' is 1 (data quality issue)
            dismissals_df['out_kind'] = dismissals_df['out_kind'].fillna(
                'Unknown')
            dismissal_counts = dismissals_df["out_kind"].value_counts(
            ).reset_index()
            dismissal_counts.columns = ["Dismissal Type", "Count"]
            # Exclude 'not out' if it sneaks in
            dismissal_counts = dismissal_counts[dismissal_counts["Dismissal Type"].str.lower(
            ) != 'not out']
            fig_dismissals = plot_dismissal_pie(
                dismissal_counts, title=f"Dismissals vs {selected_opposition}")
        else:
            fig_dismissals = go.Figure(layout=go.Layout(
                title=f"Dismissals vs {selected_opposition}<br>(No dismissals recorded in period)"))
    else:
        fig_dismissals = go.Figure(layout=go.Layout(
            title=f"Dismissals vs {selected_opposition}<br>(Dismissal data unavailable)"))

    # Run contribution vs opponent (in period)
    runs_from_4s = opp_fours * 4
    runs_from_6s = opp_sixes * 6
    # Calculate other runs carefully, ensure non-negative
    other_runs = max(0, opp_total_runs - runs_from_4s - runs_from_6s)
    fig_runs = plot_run_contribution_pie(
        runs_from_4s, runs_from_6s, other_runs, title=f"Run Scoring Breakdown vs {selected_opposition}")

    # Comparison Visuals (Opponent Period vs Career Period)
    fig_gauge_bat_avg = create_comparison_gauge(
        opp_bat_avg, car_period_bat_avg, "Batting Average", higher_is_better=True)
    # Economy: lower is better (implicit in gauge threshold)
    fig_gauge_bowl_econ = create_comparison_gauge(
        opp_bowl_econ, car_period_bowl_econ, "Bowling Economy", upper_bound=15, higher_is_better=False)
    fig_bar_bat_sr = create_comparison_bar(
        opp_bat_sr, car_period_bat_sr, f"vs {selected_opposition}", "Career (Period)", "Batting Strike Rate")
    # fig_bar_runs = create_comparison_bar(opp_total_runs, car_period_total_runs, f"vs {selected_opposition}", "Career (Period)", "Total Runs Scored") # Less insightful maybe
    fig_bar_wkts = create_comparison_bar(
        opp_wickets, car_period_wickets, f"vs {selected_opposition}", "Career (Period)", "Total Wickets Taken")

    # --- Assemble Layout ---
    player_team = career_stats_df_period["player_team"].iloc[0] if not career_stats_df_period.empty and "player_team" in career_stats_df_period.columns and not career_stats_df_period["player_team"].empty else "N/A"
    header_text = f"{selected_player} ({player_team}) vs {selected_opposition} ({selected_format})"
    subheader_text = f"Period: {start_date} to {end_date}"

    stats_layout = html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H4(header_text, className="text-center mb-0"),
                html.P(subheader_text,
                       className="text-center text-muted small mb-0")
            ]),
            dbc.CardBody([
                # Summary Row
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Overall vs Opponent (Period)",
                                className="card-title text-center"),
                        # Use matches calculated earlier
                        html.P(f"Matches: {opp_matches}"),
                        html.P(f"Runs Scored: {opp_total_runs:,}"),
                        html.P(f"Wickets Taken: {opp_wickets}")
                    ]), color="light", className="h-100"), width=12, md=4, className="mb-3"),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Batting vs Opponent (Period)",
                                className="card-title text-center"),
                        html.P(f"Average: {f'{opp_bat_avg:.2f}' if pd.notna(
                            opp_bat_avg) else 'N/A'}"),
                        html.P(f"Strike Rate: {f'{opp_bat_sr:.2f}' if pd.notna(
                            opp_bat_sr) else 'N/A'}"),
                        html.P(f"Outs: {opp_outs}")
                    ]), color="light", className="h-100"), width=6, md=4, className="mb-3"),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Bowling vs Opponent (Period)",
                                className="card-title text-center"),
                        html.P(f"Average: {f'{opp_bowl_avg:.2f}' if pd.notna(
                            opp_bowl_avg) else 'N/A'}"),
                        html.P(f"Economy: {f'{opp_bowl_econ:.2f}' if pd.notna(
                            opp_bowl_econ) else 'N/A'}"),
                        html.P(f"Wickets: {opp_wickets}")
                    ]), color="light", className="h-100"), width=6, md=4, className="mb-3"),
                ], className="mb-4"),
                html.Hr(),
                # Comparison Row
                html.H5("Comparison: Performance vs Opponent vs Career Average (within Period)",
                        className="text-center mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_gauge_bat_avg, config={
                            'displayModeBar': False}), width=12, sm=6, lg=3, className="mb-3"),
                    dbc.Col(dcc.Graph(figure=fig_bar_bat_sr, config={
                            'displayModeBar': False}), width=12, sm=6, lg=3, className="mb-3"),
                    dbc.Col(dcc.Graph(figure=fig_gauge_bowl_econ, config={
                            'displayModeBar': False}), width=12, sm=6, lg=3, className="mb-3"),
                    dbc.Col(dcc.Graph(figure=fig_bar_wkts, config={
                            'displayModeBar': False}), width=12, sm=6, lg=3, className="mb-3"),
                    # Use align-items-stretch if heights vary
                ], className="mb-4 align-items-stretch"),
                html.Hr(),
                # Detailed Analysis Row
                html.H5(
                    f"Detailed Analysis vs {selected_opposition} (Period)", className="text-center mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_runs, config={
                        'displayModeBar': False}), width=12, md=6, className="mb-3"),
                    dbc.Col(dcc.Graph(figure=fig_dismissals, config={
                        'displayModeBar': False}), width=12, md=6, className="mb-3"),
                ]),
                # Dismissal Table (conditional display)
                dbc.Row([
                    dbc.Col([
                        html.H6("Dismissal Details (vs Opponent, Period)",
                                className="mt-3 text-center"),
                        dash_table.DataTable(
                            data=dismissal_counts.to_dict('records'),
                            columns=[{'name': i, 'id': i}
                                     for i in dismissal_counts.columns],
                            style_cell={
                                'textAlign': 'left', 'padding': '5px', 'fontFamily': 'sans-serif'},
                            style_header={'fontWeight': 'bold',
                                          'backgroundColor': 'rgb(230, 230, 230)'},
                            page_size=5,  # Show fewer rows or adjust as needed
                            style_table={'overflowX': 'auto'}
                        )
                    ], width=12, md=8, lg=6)  # Center the table slightly
                ],
                    # Center the column containing the table
                    className="mt-3 justify-content-center",
                    # Hide row if no dismissal data
                    style={
                        'display': 'flex' if not dismissal_counts.empty else 'none'}
                ),

                # Download Buttons
                html.Hr(className="my-4"),
                html.Div([
                    html.H5("Download Filtered Data (Period)",
                            className="text-center mb-3"),
                    dbc.Row([
                         dbc.Col(dbc.Button([html.I(className="fas fa-download me-2"), "Download CSV (vs Opponent, Period)"], id="analyzer-csv-button", n_clicks=0,
                                 # Add margin bottom for small screens
                                            color="primary"), width="auto", className="d-flex justify-content-center mb-2 mb-md-0"),
                         dbc.Col(dbc.Button([html.I(className="fas fa-file-pdf me-2"), "Download PDF Summary (vs Opp., Period)"],
                                 id="analyzer-pdf-button", n_clicks=0, color="danger"), width="auto", className="d-flex justify-content-center"),
                         ], justify="center", className="g-2")  # Use g-2 for spacing between buttons
                ], className="mt-4")
            ])  # End CardBody
        ], className="mt-4 shadow-sm")  # Add margin top and shadow to card
    ])  # End of main Div

    print("  - Stats layout generated successfully.")
    return stats_layout


# Callback 4: Download CSV (Uses Date Range State)
@callback(
    Output("analyzer-download-csv", "data"),
    Input("analyzer-csv-button", "n_clicks"),
    State('analyzer-player-dropdown', 'value'),
    State('analyzer-format-radio', 'value'),
    State('analyzer-selected-opposition-store', 'data'),
    State('analyzer-date-picker-range', 'start_date'),  # *** ADDED STATE ***
    State('analyzer-date-picker-range', 'end_date'),   # *** ADDED STATE ***
    State('main-data-store', 'data'),
    prevent_initial_call=True,
)
def analyzer_download_csv(n_clicks, player, format, opposition, start_date, end_date, stored_data):
    # (Logic is similar to stats callback, but prepares CSV)
    if n_clicks is None or n_clicks == 0 or not opposition or not player or not format or not start_date or not end_date:
        raise PreventUpdate

    print(
        f"Callback triggered: analyzer_download_csv (Clicks: {n_clicks}, Period: {start_date}-{end_date})")
    df_full = deserialize_data(stored_data)
    if df_full is None or df_full.empty:
        print("  - CSV download failed: Main data not available.")
        raise PreventUpdate
    if 'start_date' not in df_full.columns:
        print("  - 'start_date' column missing for CSV filtering.")
        raise PreventUpdate

    try:
        # Ensure start_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_full['start_date']):
            df_full['start_date'] = pd.to_datetime(
                df_full['start_date'], errors='coerce')

        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()

        # Filter for data vs opponent within the date range
        mask = (
            (df_full["name"] == player) &
            (df_full["match_type"] == format) &
            (df_full["opposition_team"] == opposition) &
            (df_full['start_date'].notna()) &
            (df_full['start_date'] >= start_dt) &
            (df_full['start_date'] <= end_dt)
        )
        country_stats_df_period = df_full[mask].copy()

        if country_stats_df_period.empty:
            print("  - No data to download for CSV in the selected period.")
            # Optionally: return a notification to the user instead of just preventing update
            raise PreventUpdate

        # Sanitize filename components
        safe_player = "".join(c if c.isalnum() else "_" for c in player)
        safe_opposition = "".join(
            c if c.isalnum() else "_" for c in opposition)
        safe_format = "".join(c if c.isalnum() else "_" for c in format)
        safe_start = start_date.replace("-", "")
        safe_end = end_date.replace("-", "")
        fn = f"{safe_player}_vs_{safe_opposition}_{safe_format}_{safe_start}_{safe_end}_matches.csv"

        print(f"  - Preparing CSV download: {fn}")
        # Convert DataFrame to CSV string for download
        csv_string = country_stats_df_period.to_csv(
            index=False, encoding='utf-8')
        # Use dict format for dcc.send_data_frame alternative
        return dict(content=csv_string, filename=fn)

    except Exception as e:
        print(f"  - Error during CSV generation: {e}")
        import traceback
        traceback.print_exc()
        # Potentially return an error message to the user
        raise PreventUpdate

# Callback 5: Download PDF Summary (Uses Date Range State)


@callback(
    Output("analyzer-download-pdf", "data"),
    Input("analyzer-pdf-button", "n_clicks"),
    State('analyzer-player-dropdown', 'value'),
    State('analyzer-format-radio', 'value'),
    State('analyzer-selected-opposition-store', 'data'),
    State('analyzer-date-picker-range', 'start_date'),  # *** ADDED STATE ***
    State('analyzer-date-picker-range', 'end_date'),   # *** ADDED STATE ***
    State('main-data-store', 'data'),
    prevent_initial_call=True,
)
def analyzer_download_pdf(n_clicks, player, format, opposition, start_date, end_date, stored_data):
    # (Logic is similar to stats callback, but calculates summary for PDF)
    if n_clicks is None or n_clicks == 0 or not opposition or not player or not format or not start_date or not end_date:
        raise PreventUpdate

    print(
        f"Callback triggered: analyzer_download_pdf (Clicks: {n_clicks}, Period: {start_date}-{end_date})")
    df_full = deserialize_data(stored_data)
    if df_full is None or df_full.empty:
        print("  - PDF download failed: Main data not available.")
        raise PreventUpdate
    if 'start_date' not in df_full.columns:
        print("  - 'start_date' column missing for PDF filtering.")
        raise PreventUpdate

    try:
        # Ensure start_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_full['start_date']):
            df_full['start_date'] = pd.to_datetime(
                df_full['start_date'], errors='coerce')

        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()

        # Filter data vs opponent within the date range
        mask = (
            (df_full["name"] == player) &
            (df_full["match_type"] == format) &
            (df_full["opposition_team"] == opposition) &
            (df_full['start_date'].notna()) &
            (df_full['start_date'] >= start_dt) &
            (df_full['start_date'] <= end_dt)
        )
        country_stats_df_period = df_full[mask].copy()

        if country_stats_df_period.empty:
            print("  - No data to generate PDF summary for the selected period.")
            raise PreventUpdate

        # --- Calculate Summary Stats for PDF ---
        total_runs = int(country_stats_df_period["runs_scored"].sum())
        balls_faced = int(country_stats_df_period["balls_faced"].sum())
        outs = int(country_stats_df_period["player_out"].sum())
        batting_avg = total_runs / outs if outs > 0 else np.nan
        batting_strike_rate = (total_runs / balls_faced) * \
            100 if balls_faced > 0 else 0.0
        fours = int(country_stats_df_period["fours_scored"].sum())
        sixes = int(country_stats_df_period["sixes_scored"].sum())

        total_wickets = int(country_stats_df_period["wickets_taken"].sum())
        balls_bowled = int(country_stats_df_period["balls_bowled"].sum())
        total_runs_conceded = int(
            country_stats_df_period["runs_conceded"].sum())
        bowling_avg = total_runs_conceded / total_wickets if total_wickets > 0 else np.nan
        bowling_economy = (total_runs_conceded / balls_bowled) * \
            6 if balls_bowled > 0 else np.nan  # Handle div by zero
        matches = country_stats_df_period['match_id'].nunique()

        def format_stat(value, decimals=2):
            # Format NaN as 'N/A', floats to specified decimals, integers as is
            if pd.isna(value):
                return 'N/A'
            if isinstance(value, float):
                return f'{value:.{decimals}f}'
            return f'{value:,}'  # Integer formatting

        # --- Create PDF Text ---
        pdf_text = (
            f"Performance Summary\n"
            f"--------------------\n"
            f"Player: {player}\n"
            f"Opposition: {opposition}\n"
            f"Format: {format}\n"
            f"Period: {start_date} to {end_date}\n"
            f"--------------------\n"
            f"Matches: {matches}\n\n"
            f"Batting Summary (Period):\n"
            f"  Runs Scored: {format_stat(total_runs)}\n"
            f"  Balls Faced: {format_stat(balls_faced)}\n"
            f"  Dismissals: {format_stat(outs)}\n"
            f"  Average: {format_stat(batting_avg)}\n"
            f"  Strike Rate: {format_stat(batting_strike_rate)}\n"
            f"  Fours: {format_stat(fours)}\n"
            f"  Sixes: {format_stat(sixes)}\n\n"
            f"Bowling Summary (Period):\n"
            f"  Wickets Taken: {format_stat(total_wickets)}\n"
            f"  Runs Conceded: {format_stat(total_runs_conceded)}\n"
            f"  Balls Bowled: {format_stat(balls_bowled)}\n"
            f"  Average: {format_stat(bowling_avg)}\n"
            f"  Economy Rate: {format_stat(bowling_economy)}\n"
        )

        # Assumes export_pdf function is defined
        pdf_buffer = export_pdf(pdf_text)

        # Sanitize filename components
        safe_player = "".join(c if c.isalnum() else "_" for c in player)
        safe_opposition = "".join(
            c if c.isalnum() else "_" for c in opposition)
        safe_format = "".join(c if c.isalnum() else "_" for c in format)
        safe_start = start_date.replace("-", "")
        safe_end = end_date.replace("-", "")
        fn = f"{safe_player}_vs_{safe_opposition}_{safe_format}_{safe_start}_{safe_end}_summary.pdf"

        print(f"  - Preparing PDF download: {fn}")
        return dcc.send_bytes(pdf_buffer.getvalue(), filename=fn)

    except Exception as e:
        print(f"  - Error during PDF generation: {e}")
        import traceback
        traceback.print_exc()
        raise PreventUpdate
