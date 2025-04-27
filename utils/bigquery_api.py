"""
bigquery_api.py
Centralised, reusable helpers for fetching cricket-match data
from Google BigQuery.

Every function returns a **pandas DataFrame** that the Dash page
(or anything else) can consume directly.
"""
from google.cloud import bigquery
import pandas as pd
from functools import lru_cache
from datetime import datetime, date
from google.oauth2 import service_account

# -------------------------------------------------------------------------
#  CONFIG – EDIT THESE ----------------------------------------------------
# -------------------------------------------------------------------------
PROJECT_ID = "primordial-veld-456613-n6"
DATASET = "cs661_gr2_discovered_001"
TABLE = "total_data"
# -------------------------------------------------------------------------
KEY_PATH = "../primordial-veld-456613-n6-c5dd57e4037a.json"
CREDENTIALS = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

_client = bigquery.Client(project=PROJECT_ID, credentials=CREDENTIALS)


def _run(query: str, params: list[bigquery.ScalarQueryParameter]) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    return _client.query(query, job_config=job_config).result().to_dataframe()


# -------------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------------
@lru_cache(maxsize=256)
def list_players() -> list[str]:
    """Unique player names – used to populate the dropdown."""
    q = f"""
        SELECT DISTINCT name
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE name IS NOT NULL
        ORDER BY name
    """
    return _run(q, []).name.tolist()


@lru_cache(maxsize=1024)
def date_span(player: str, match_format: str) -> tuple[date, date]:
    """Earliest & latest match dates for a player/format combo."""
    q = f"""
        SELECT MIN(start_date) AS min_d, MAX(start_date) AS max_d
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE name=@p AND match_type=@f
    """
    df = _run(q, [
        bigquery.ScalarQueryParameter("p", "STRING", player),
        bigquery.ScalarQueryParameter("f", "STRING", match_format),
    ])
    return df.min_d.iloc[0].date(), df.max_d.iloc[0].date()


def fetch_player_data(
    player: str,
    match_format: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Full granular rows for one player / format / date-range.
    This is the ONLY heavy query the Dash page ever fires.
    """
    q = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE name=@p
          AND match_type=@f
          AND start_date BETWEEN @s AND @e
    """
    return _run(q, [
        bigquery.ScalarQueryParameter("p", "STRING",  player),
        bigquery.ScalarQueryParameter("f", "STRING",  match_format),
        bigquery.ScalarQueryParameter("s", "DATE",    start),
        bigquery.ScalarQueryParameter("e", "DATE",    end),
    ])
