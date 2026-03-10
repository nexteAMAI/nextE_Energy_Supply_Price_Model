"""
Joint Allocation Office (JAO) API Connector.

Cross-border transmission capacity allocation data:
  - Net Transfer Capacities (NTC) on RO-HU, RO-BG borders
  - CORE FBMC parameters (PTDF, RAM)
  - Explicit auction results (long-term, yearly, monthly, daily)
  - Congestion rents

Reference: Addendum Section A.4.
User guide: https://www.jao.eu/sites/default/files/2021-11/API_User_Guide_v1.0.pdf
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Union

import pandas as pd
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jao.eu"

# Romanian borders in the CORE FBMC region
RO_BORDERS = ["RO-HU", "RO-BG"]


class JAOClient:
    """REST API client for JAO cross-border capacity data."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.jao_api_key
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Execute GET request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params or {})
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # NTC / ATC capacity data
    # ------------------------------------------------------------------

    def get_ntc(
        self,
        border: str,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """
        Fetch Net Transfer Capacity for a specific border.

        Parameters
        ----------
        border : str
            Border code (e.g., 'RO-HU', 'RO-BG')
        """
        params = {
            "border": border,
            "fromDate": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "toDate": pd.Timestamp(end).strftime("%Y-%m-%d"),
        }

        try:
            data = self._get("api/data/ntc", params)
            df = pd.DataFrame(data)
            if not df.empty and "dateTime" in df.columns:
                df["dateTime"] = pd.to_datetime(df["dateTime"])
                df = df.set_index("dateTime")
            logger.info("JAO NTC %s: %d rows", border, len(df))
            return df
        except Exception as e:
            logger.error("JAO NTC query failed for %s: %s", border, e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Auction results
    # ------------------------------------------------------------------

    def get_auction_results(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
        auction_type: str = "daily",
    ) -> pd.DataFrame:
        """Fetch explicit auction results (daily/monthly/yearly)."""
        params = {
            "fromDate": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "toDate": pd.Timestamp(end).strftime("%Y-%m-%d"),
            "type": auction_type,
        }
        try:
            data = self._get("api/data/auction-results", params)
            df = pd.DataFrame(data)
            logger.info("JAO auction results (%s): %d records", auction_type, len(df))
            return df
        except Exception as e:
            logger.error("JAO auction results query failed: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # CORE FBMC data (flow-based market coupling)
    # ------------------------------------------------------------------

    def get_fbmc_data(
        self,
        target_date: Union[date, datetime, str],
    ) -> Dict[str, Any]:
        """
        Fetch CORE FBMC parameters for a given delivery date.
        Returns raw JSON response with PTDF matrices, RAM values.
        """
        d = pd.Timestamp(target_date).strftime("%Y-%m-%d")
        try:
            data = self._get("api/data/coreFlowBased", {"date": d})
            logger.info("JAO FBMC data for %s: retrieved", d)
            return data
        except Exception as e:
            logger.error("JAO FBMC query failed for %s: %s", d, e)
            return {}

    # ------------------------------------------------------------------
    # Convenience: All Romanian border NTCs
    # ------------------------------------------------------------------

    def get_all_ro_ntc(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> Dict[str, pd.DataFrame]:
        """Fetch NTC for all Romanian borders."""
        results = {}
        for border in RO_BORDERS:
            results[border] = self.get_ntc(border, start, end)
        return results
