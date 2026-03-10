"""
Balancing Services API Connector.

Specialized source for European balancing market data at 15-min granularity:
  - Imbalance settlement prices (positive/negative direction)
  - Imbalance total volumes (surplus/deficit)
  - Balancing energy activated volumes and prices (aFRR, mFRR)
  - Balancing capacity procurement

Reference: Addendum Section A.3.
Swagger docs: https://api.balancing.services/v1/documentation
Coverage: July 2024 onward for Romania (area=RO).
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.balancing.services/v1"
RESERVE_TYPES = ["FCR", "aFRR_UP", "aFRR_DOWN", "mFRR_UP", "mFRR_DOWN", "RR"]


class BalancingServicesClient:
    """REST API client for the Balancing Services platform."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.balancing_services_api_key
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    def _get(self, endpoint: str, params: Dict[str, Any]) -> List[Dict]:
        """Execute GET request with pagination handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        all_data = []

        while url:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                all_data.extend(data)
                url = None  # No pagination for list responses
            elif isinstance(data, dict):
                all_data.extend(data.get("data", data.get("items", [data])))
                # Handle pagination via 'next' link if present
                url = data.get("next")
                params = {}  # Params are included in the next URL
            else:
                break

        return all_data

    def _date_params(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> Dict[str, str]:
        """Build date range query parameters."""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        return {
            "area": "RO",
            "period-start-at": start.isoformat(),
            "period-end-at": end.isoformat(),
        }

    # ------------------------------------------------------------------
    # Imbalance Prices (Module 5 — Core)
    # ------------------------------------------------------------------

    def get_imbalance_prices(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """
        Fetch imbalance settlement prices for Romania.
        Returns 15-min resolution with positive/negative direction.

        Columns: timestamp_eet, direction, price (RON/MWh), currency
        """
        params = self._date_params(start, end)
        data = self._get("imbalance/prices", params)

        if not data:
            logger.warning("No imbalance price data returned for %s — %s", start, end)
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["period_startAt"] = pd.to_datetime(df["period_startAt"], utc=True)
        df = df.set_index("period_startAt")
        df.index = df.index.tz_convert(settings.timezone)
        df.index.name = "timestamp_eet"

        # Pivot positive/negative into separate columns
        positive = df[df["direction"] == "positive"]["price"].rename("price_positive_ron")
        negative = df[df["direction"] == "negative"]["price"].rename("price_negative_ron")
        result = pd.concat([positive, negative], axis=1).sort_index()

        logger.info("Balancing Services imbalance prices: %d rows, %s — %s",
                     len(result), result.index.min(), result.index.max())
        return result

    # ------------------------------------------------------------------
    # Imbalance Total Volumes (Module 5 — Core)
    # ------------------------------------------------------------------

    def get_imbalance_volumes(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """
        Fetch total imbalance volumes for Romania.
        Returns 15-min resolution with surplus/deficit split.

        Columns: timestamp_eet, volume_surplus_mw, volume_deficit_mw
        """
        params = self._date_params(start, end)
        data = self._get("imbalance/total-volumes", params)

        if not data:
            logger.warning("No imbalance volume data returned for %s — %s", start, end)
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["period_startAt"] = pd.to_datetime(df["period_startAt"], utc=True)
        df = df.set_index("period_startAt")
        df.index = df.index.tz_convert(settings.timezone)
        df.index.name = "timestamp_eet"

        surplus = df[df["direction"] == "surplus"]["averagePowerMW"].rename("volume_surplus_mw")
        deficit = df[df["direction"] == "deficit"]["averagePowerMW"].rename("volume_deficit_mw")
        result = pd.concat([surplus, deficit], axis=1).sort_index()

        logger.info("Balancing Services imbalance volumes: %d rows", len(result))
        return result

    # ------------------------------------------------------------------
    # Balancing Energy — Activated Volumes & Prices (Module 5)
    # ------------------------------------------------------------------

    def get_balancing_energy_activations(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """Fetch activated balancing energy volumes (aFRR, mFRR)."""
        params = self._date_params(start, end)
        data = self._get("balancing-energy/activated-volumes", params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["period_startAt"] = pd.to_datetime(df["period_startAt"], utc=True)
        df = df.set_index("period_startAt")
        df.index = df.index.tz_convert(settings.timezone)
        df.index.name = "timestamp_eet"
        logger.info("Balancing energy activations: %d rows", len(df))
        return df

    def get_balancing_energy_prices(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """Fetch balancing energy prices per reserve type."""
        params = self._date_params(start, end)
        data = self._get("balancing-energy/prices", params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["period_startAt"] = pd.to_datetime(df["period_startAt"], utc=True)
        df = df.set_index("period_startAt")
        df.index = df.index.tz_convert(settings.timezone)
        df.index.name = "timestamp_eet"
        logger.info("Balancing energy prices: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Balancing Capacity — Procured Volumes (Module 5)
    # ------------------------------------------------------------------

    def get_balancing_capacity(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """Fetch procured balancing capacity volumes (FCR, aFRR, mFRR)."""
        params = self._date_params(start, end)
        data = self._get("balancing-capacity/procured-volumes", params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        logger.info("Balancing capacity procured: %d records", len(df))
        return df

    # ------------------------------------------------------------------
    # Daily refresh bundle
    # ------------------------------------------------------------------

    def daily_refresh(
        self,
        target_date: Optional[date] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Execute standard daily refresh for D-1 balancing data."""
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        start = datetime.combine(target_date, datetime.min.time())
        end = start + timedelta(days=1)

        return {
            "imbalance_prices": self.get_imbalance_prices(start, end),
            "imbalance_volumes": self.get_imbalance_volumes(start, end),
            "balancing_energy_activations": self.get_balancing_energy_activations(start, end),
            "balancing_energy_prices": self.get_balancing_energy_prices(start, end),
        }
