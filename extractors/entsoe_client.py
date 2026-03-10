"""
ENTSO-E Transparency Platform API Connector.

Authoritative source for:
  - Day-ahead prices (hourly/15-min from Jan 2025)
  - Actual generation by fuel type
  - Load actual + forecast
  - Wind/solar forecasts
  - Cross-border physical flows
  - Imbalance volumes and prices

Reference: Addendum Section A.2.
Python wrapper: https://github.com/EnergieID/entsoe-py
Romania bidding zone EIC: 10YRO-TEL------P
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Union

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# Neighbouring bidding zones for cross-border analysis
NEIGHBOUR_ZONES = {
    "HU": "10YHU-MAVIR----U",
    "BG": "10YCA-BULGARIA-R",
    "RS": "10YCS-SERBIATSOV",
    "UA": "10Y1001C--00003F",
}


class ENTSOEClient:
    """Wrapper around the entsoe-py client for Romanian grid data."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.entsoe_api_key
        self.country_code = "RO"
        self._client = None

    @property
    def client(self):
        """Lazy-initialize the ENTSO-E pandas client."""
        if self._client is None:
            try:
                from entsoe import EntsoePandasClient
                self._client = EntsoePandasClient(api_key=self.api_key)
                logger.info("ENTSO-E client initialized successfully.")
            except ImportError:
                raise ImportError(
                    "entsoe package not installed. "
                    "Install with: pip install entsoe-py"
                )
        return self._client

    def _make_timestamps(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> tuple:
        """Convert to timezone-aware Timestamps for ENTSO-E API calls."""
        tz = settings.timezone
        start = pd.Timestamp(start, tz=tz)
        end = pd.Timestamp(end, tz=tz)
        return start, end

    def _to_eet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DatetimeIndex is in EET/EEST."""
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(settings.timezone)
        else:
            df.index = df.index.tz_convert(settings.timezone)
        df.index.name = "timestamp_eet"
        return df

    # ------------------------------------------------------------------
    # Day-Ahead Prices (Module 1)
    # ------------------------------------------------------------------

    def query_day_ahead_prices(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """
        Fetch day-ahead market clearing prices for Romania.
        Returns Series as DataFrame with column 'dam_price_eur_mwh'.
        """
        start, end = self._make_timestamps(start, end)
        series = self.client.query_day_ahead_prices(self.country_code, start=start, end=end)
        df = series.to_frame(name="dam_price_eur_mwh")
        df = self._to_eet(df)
        logger.info("ENTSO-E DAM prices: %d rows, %s — %s",
                     len(df), df.index.min(), df.index.max())
        return df

    # ------------------------------------------------------------------
    # Generation by Fuel Type (Module 3)
    # ------------------------------------------------------------------

    def query_generation(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """
        Fetch actual generation per production type (10+ fuel categories).
        Returns DataFrame with multi-level or flat columns by fuel type.
        """
        start, end = self._make_timestamps(start, end)
        df = self.client.query_generation(self.country_code, start=start, end=end, psr_type=None)
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{c[0]}_{c[1]}".strip("_") for c in df.columns]
        df = self._to_eet(df)
        logger.info("ENTSO-E generation: %d rows × %d fuel types",
                     len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Load — Actual & Forecast (Module 3)
    # ------------------------------------------------------------------

    def query_load(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """Fetch actual total load (MW)."""
        start, end = self._make_timestamps(start, end)
        series = self.client.query_load(self.country_code, start=start, end=end)
        df = series.to_frame(name="actual_load_mw")
        df = self._to_eet(df)
        logger.info("ENTSO-E load actual: %d rows", len(df))
        return df

    def query_load_forecast(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """Fetch day-ahead load forecast (MW)."""
        start, end = self._make_timestamps(start, end)
        series = self.client.query_load_forecast(self.country_code, start=start, end=end)
        df = series.to_frame(name="forecast_load_mw")
        df = self._to_eet(df)
        logger.info("ENTSO-E load forecast: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Wind & Solar Forecasts (Module 3)
    # ------------------------------------------------------------------

    def query_wind_solar_forecast(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """Fetch day-ahead wind + solar generation forecast."""
        start, end = self._make_timestamps(start, end)
        df = self.client.query_wind_and_solar_forecast(self.country_code, start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{c[0]}_{c[1]}".strip("_") for c in df.columns]
        df = self._to_eet(df)
        logger.info("ENTSO-E wind+solar forecast: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Cross-Border Physical Flows (Module 3)
    # ------------------------------------------------------------------

    def query_crossborder_flows(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
        neighbours: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Fetch net physical flows for all Romanian borders.
        Net flow > 0 means import into Romania.

        Parameters
        ----------
        neighbours : list, optional
            Subset of ['HU', 'BG', 'RS', 'UA']. Default: all.

        Returns
        -------
        DataFrame with columns per border (net MW) + 'total_net_import'.
        """
        start, end = self._make_timestamps(start, end)
        if neighbours is None:
            neighbours = list(NEIGHBOUR_ZONES.keys())

        flows = {}
        for nb in neighbours:
            try:
                # RO → NB (export)
                export = self.client.query_crossborder_flows(
                    "RO", nb, start=start, end=end
                )
                # NB → RO (import)
                imp = self.client.query_crossborder_flows(
                    nb, "RO", start=start, end=end
                )
                net = imp - export  # positive = net import into RO
                flows[f"net_{nb}_mw"] = net
            except Exception as e:
                logger.warning("Failed to fetch flows for RO-%s: %s", nb, e)

        if not flows:
            return pd.DataFrame()

        df = pd.DataFrame(flows)
        df["total_net_import_mw"] = df.sum(axis=1)
        df = self._to_eet(df)
        logger.info("ENTSO-E cross-border flows: %d rows, borders: %s",
                     len(df), list(flows.keys()))
        return df

    # ------------------------------------------------------------------
    # Imbalance Prices & Volumes (Module 5)
    # ------------------------------------------------------------------

    def query_imbalance_prices(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """Fetch imbalance settlement prices (long/short)."""
        start, end = self._make_timestamps(start, end)
        df = self.client.query_imbalance_prices(self.country_code, start=start, end=end)
        if isinstance(df, pd.Series):
            df = df.to_frame(name="imbalance_price")
        df = self._to_eet(df)
        logger.info("ENTSO-E imbalance prices: %d rows", len(df))
        return df

    def query_imbalance_volumes(
        self,
        start: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> pd.DataFrame:
        """Fetch imbalance volumes (net, MWh)."""
        start, end = self._make_timestamps(start, end)
        series = self.client.query_imbalance_volumes(self.country_code, start=start, end=end)
        df = series.to_frame(name="imbalance_volume_mwh") if isinstance(series, pd.Series) else series
        df = self._to_eet(df)
        logger.info("ENTSO-E imbalance volumes: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Full daily refresh bundle
    # ------------------------------------------------------------------

    def daily_refresh(
        self,
        target_date: Optional[date] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute the standard daily data refresh.
        Fetches D-1 data for all Module 1, 3, and 5 datasets.

        Returns dict of DataFrames keyed by dataset name.
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        start = target_date
        end = target_date + timedelta(days=1)

        results = {}
        extractors = {
            "dam_prices": self.query_day_ahead_prices,
            "generation": self.query_generation,
            "load_actual": self.query_load,
            "load_forecast": self.query_load_forecast,
            "wind_solar_forecast": self.query_wind_solar_forecast,
            "crossborder_flows": self.query_crossborder_flows,
            "imbalance_prices": self.query_imbalance_prices,
            "imbalance_volumes": self.query_imbalance_volumes,
        }

        for name, func in extractors.items():
            try:
                results[name] = func(start, end)
            except Exception as e:
                logger.error("Daily refresh failed for %s: %s", name, e)
                results[name] = pd.DataFrame()

        logger.info("ENTSO-E daily refresh complete: %d datasets", len(results))
        return results
