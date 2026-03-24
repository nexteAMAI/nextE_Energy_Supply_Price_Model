"""
Energy Quantified (Montel) API Connector.

Handles extraction of:
  - TIMESERIES: Spot prices, generation actuals, consumption, backcasts
  - INSTANCE: Wind/solar/load forecasts
  - OHLC: Commodity futures (TTF, coal, EUA, Brent), power forwards (EEX)
  - Forward curve snapshots

Reference: Addendum Section A.1, EQ Excel Integrator v1.1 templates.
Python client docs: https://energyquantified-python.readthedocs.io/en/latest/
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class EQClient:
    """Wrapper around the Energy Quantified Python client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.eq_api_key
        self._client = None

    @property
    def client(self):
        """Lazy-initialize the EQ client."""
        if self._client is None:
            try:
                from energyquantified import EnergyQuantified
                self._client = EnergyQuantified(api_key=self.api_key)
                logger.info("EQ client initialized successfully.")
            except ImportError:
                raise ImportError(
                    "energyquantified package not installed. "
                    "Install with: pip install energyquantified"
                )
            except Exception as e:
                raise ConnectionError(f"Failed to initialize EQ client: {e}")
        return self._client

    # ------------------------------------------------------------------
    # Curve discovery
    # ------------------------------------------------------------------

    def search_curves(self, query: str, area: str = "RO") -> list:
        """Search for available EQ curves matching a query string."""
        results = self.client.metadata.curves(q=query, area=area)
        logger.info("Curve search '%s' (area=%s): %d results", query, area, len(results))
        return results

    def resolve_curve(self, curve_name: str) -> Any:
        """Resolve a curve name to an EQ Curve object."""
        results = self.client.metadata.curves(q=curve_name)
        if not results:
            raise ValueError(f"No EQ curve found for: '{curve_name}'")
        # Return exact match if available, otherwise first result
        for r in results:
            if str(r) == curve_name or curve_name in str(r):
                return r
        return results[0]

    # ------------------------------------------------------------------
    # TIMESERIES extraction
    # ------------------------------------------------------------------

    def get_timeseries(
        self,
        curve_name: str,
        begin: Union[date, datetime, str],
        end: Union[date, datetime, str],
        frequency: Optional[str] = None,
        aggregation: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract a timeseries from EQ API.

        Parameters
        ----------
        curve_name : str
            EQ curve identifier (e.g., 'RO Price Spot EUR/MWh OPCOM H Actual')
        begin, end : date-like
            Date range for extraction
        frequency : str, optional
            Resampling frequency (e.g., 'PT15M', 'PT1H', 'P1D', 'P1M')
        aggregation : str, optional
            Aggregation method (e.g., 'AVERAGE', 'SUM')

        Returns
        -------
        pd.DataFrame with DatetimeIndex (Europe/Bucharest) and value column.
        """
        curve = self.resolve_curve(curve_name)
        begin = pd.Timestamp(begin)
        end = pd.Timestamp(end)

        kwargs = {"begin": begin, "end": end}
        if frequency:
            from energyquantified.metadata import Frequency
            kwargs["frequency"] = getattr(Frequency, frequency, frequency)
        if aggregation:
            from energyquantified.metadata import Aggregation
            kwargs["aggregation"] = getattr(Aggregation, aggregation, aggregation)

        ts = self.client.timeseries.load(curve, **kwargs)

        # Convert to pandas DataFrame
        records = [(v.date, v.value) for v in ts.data if v.value is not None]
        df = pd.DataFrame(records, columns=["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Bucharest")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Bucharest")
        df = df.set_index("timestamp").sort_index()
        df.columns = [curve_name]

        logger.info("EQ timeseries '%s': %d data points, %s — %s",
                     curve_name, len(df), df.index.min(), df.index.max())
        return df

    # ------------------------------------------------------------------
    # INSTANCE (forecast) extraction
    # ------------------------------------------------------------------

    def get_instance(
        self,
        curve_name: str,
        begin: Union[date, datetime, str],
        end: Union[date, datetime, str],
        tag: str = "EC",
        issued: str = "latest",
        limit: int = 2,
    ) -> pd.DataFrame:
        """
        Extract forecast instance data from EQ API.

        Parameters
        ----------
        curve_name : str
            EQ forecast curve name
        tag : str
            Forecast tag (e.g., 'EC' for ECMWF)
        issued : str
            'latest' or 'earliest'
        limit : int
            Number of most recent instances to fetch

        Returns
        -------
        pd.DataFrame with forecast values.
        """
        curve = self.resolve_curve(curve_name)
        begin = pd.Timestamp(begin)
        end = pd.Timestamp(end)

        instances = self.client.instances.list(curve, tags=[tag], limit=limit)
        if not instances:
            logger.warning("No instances found for '%s' with tag '%s'", curve_name, tag)
            return pd.DataFrame()

        # Load the latest/earliest instance
        target = instances[0] if issued == "latest" else instances[-1]
        ts = self.client.instances.load(curve, begin=begin, end=end, tag=tag)

        records = [(v.date, v.value) for v in ts.data if v.value is not None]
        df = pd.DataFrame(records, columns=["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Bucharest")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Bucharest")
        df = df.set_index("timestamp").sort_index()
        df.columns = [curve_name]

        logger.info("EQ instance '%s' (tag=%s): %d points", curve_name, tag, len(df))
        return df

    # ------------------------------------------------------------------
    # OHLC (futures) extraction
    # ------------------------------------------------------------------

    def get_ohlc(
        self,
        curve_name: str,
        begin: Union[date, datetime, str],
        end: Union[date, datetime, str],
        field: str = "settlement",
        period: str = "month",
        front: int = 1,
    ) -> pd.DataFrame:
        """
        Extract OHLC (futures settlement) data.

        Parameters
        ----------
        curve_name : str
            EQ OHLC curve name (e.g., 'NL Futures Natural Gas EUR/MWh ICE-TTF OHLC')
        field : str
            Price field — 'settlement', 'open', 'high', 'low', 'close'
        period : str
            Contract period — 'month', 'quarter', 'year', 'mdec'
        front : int
            Front contract number (1 = front-month/year)

        Returns
        -------
        pd.DataFrame with trading date index and OHLC values.
        """
        curve = self.resolve_curve(curve_name)
        begin = pd.Timestamp(begin)
        end = pd.Timestamp(end)

        ohlc_data = self.client.ohlc.load(
            curve, begin=begin, end=end,
        )

        records = []
        for product in ohlc_data.data:
            for ohlc in product.data:
                records.append({
                    "trading_date": ohlc.traded,
                    "delivery_start": product.period.begin if hasattr(product.period, 'begin') else None,
                    "open": ohlc.open,
                    "high": ohlc.high,
                    "low": ohlc.low,
                    "close": ohlc.close,
                    "settlement": ohlc.settlement,
                    "volume": ohlc.volume,
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df["trading_date"] = pd.to_datetime(df["trading_date"])
            df = df.set_index("trading_date").sort_index()

        logger.info("EQ OHLC '%s' (%s/%s): %d records", curve_name, field, period, len(df))
        return df

    # ------------------------------------------------------------------
    # Forward curve snapshot (latest trading day)
    # ------------------------------------------------------------------

    def get_forward_curve(
        self,
        curve_name: str = "RO Futures Power Base EUR/MWh EEX OHLC",
        field: str = "settlement",
    ) -> pd.DataFrame:
        """
        Get the latest forward curve — all tenors for the most recent trading day.

        Returns
        -------
        pd.DataFrame with delivery period index and settlement price.
        """
        curve = self.resolve_curve(curve_name)

        fwd = self.client.ohlc.latest_as_periods(curve, field=field)
        records = []
        for item in fwd.data:
            records.append({
                "delivery_start": item.begin,
                "delivery_end": item.end,
                "settlement": item.value,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["delivery_start"] = pd.to_datetime(df["delivery_start"])
            df = df.set_index("delivery_start").sort_index()

        logger.info("EQ forward curve '%s': %d tenors, latest trading day",
                     curve_name, len(df))
        return df

    # ------------------------------------------------------------------
    # SRMC convenience (uses OHLC data)
    # ------------------------------------------------------------------

    def get_commodity_settlements(
        self,
        begin: Union[date, datetime, str],
        end: Union[date, datetime, str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all commodity futures settlements needed for SRMC calculation.
        Returns dict of DataFrames keyed by commodity name.
        """
        commodities = {}
        curve_map = {
            "ttf_gas": "NL Futures Natural Gas EUR/MWh ICE-TTF OHLC",
            "eua_carbon": "Futures EUA EUR/t ICE OHLC",
            "coal_api2": "Futures Coal API-2 USD/t ICE OHLC",
            "brent_oil": "Futures Crude Oil Brent USD/bbl ICE OHLC",
            "ro_power_base": "RO Futures Power Base EUR/MWh EEX OHLC",
        }

        for name, curve_name in curve_map.items():
            try:
                commodities[name] = self.get_ohlc(
                    curve_name, begin=begin, end=end,
                    field="settlement", period="month", front=1,
                )
            except Exception as e:
                logger.error("Failed to fetch %s: %s", name, e)
                commodities[name] = pd.DataFrame()

        return commodities
