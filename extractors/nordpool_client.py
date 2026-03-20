"""
Nord Pool Data Client — nextE Energy Supply
============================================
Real-time and historical data retrieval from Nord Pool (BRM, EPEX SPOT).

Fetches:
  - Day-Ahead Market (DAM) prices for TEL delivery area
  - Intraday Market (IDM) hourly statistics
  - Forward market (BRM) price indices
  - Cross-border flows and interconnector capacity

API: data.nordpoolgroup.com REST API (public, no authentication required)
Rate limit: 60 requests/minute
Timezone: EET (Eastern European Time)

Author: nextE AI Workstation
Version: 1.0.0
Date: 2026-03-19
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# CONSTANTS
# ============================================================================

NORDPOOL_API_BASE = "https://dataportal-api.nordpoolgroup.com/api/v1"
TEL_PRICE_AREA = "TEL"  # Romania (Transelectrica) price area
EEX_PRICE_AREA = "DE"   # Germany (EEX reference for basis calculation)

EXCHANGE_RATE_EUR_RON = 4.96  # Approximate (updated periodically)

# Request timeouts and retries
REQUEST_TIMEOUT_SEC = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2

# API rate limiting
REQUESTS_PER_MINUTE = 60
MIN_REQUEST_INTERVAL_SEC = 60 / REQUESTS_PER_MINUTE


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class NordPoolDAMPrice:
    """Single DAM price point."""
    timestamp: datetime
    price_area: str
    hour: int
    price_eur_per_mwh: float
    price_eur_per_kwh: float


@dataclass
class NordPoolIDMStatistics:
    """IDM hourly statistics."""
    timestamp: datetime
    hour: int
    num_trades: int
    min_price_eur_per_mwh: float
    max_price_eur_per_mwh: float
    weighted_avg_price_eur_per_mwh: float
    volume_mwh: float


# ============================================================================
# CLIENT IMPLEMENTATION
# ============================================================================

class NordPoolClient:
    """
    Nord Pool REST API client for electricity market data.

    Handles connection pooling, rate limiting, error handling, and timezone conversion.
    """

    def __init__(self, rate_limit_requests_per_min: int = 60):
        """
        Initialize Nord Pool API client.

        Args:
            rate_limit_requests_per_min: API rate limit (default 60 req/min)
        """
        self.base_url = NORDPOOL_API_BASE
        self.session = self._create_session()
        self.rate_limit_requests_per_min = rate_limit_requests_per_min
        self.min_request_interval = 60 / rate_limit_requests_per_min
        self.last_request_time = 0

    def _create_session(self) -> requests.Session:
        """Create requests session with connection pooling and retry strategy."""
        session = requests.Session()

        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF_SEC,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting to avoid API throttling."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Make HTTP request to Nord Pool API with error handling.

        Args:
            endpoint: API endpoint (relative to base URL)
            params: Query parameters

        Returns:
            Response JSON dict

        Raises:
            requests.RequestException: If request fails after retries
        """
        self._enforce_rate_limit()

        url = f"{self.base_url}/{endpoint}"
        headers = {
            "User-Agent": "nextE-EnergyPricingEngine/1.0",
            "Accept": "application/json",
        }

        try:
            logger.debug(f"GET {endpoint} with params {params}")
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Nord Pool API request failed: {e}")
            raise

    def get_dam_prices(
        self,
        price_area: str = TEL_PRICE_AREA,
        date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch day-ahead market (DAM) prices for specified date and price area.

        Args:
            price_area: Price area code (default "TEL" for Romania)
            date: Date to fetch (default: today). Must be date object (no time).

        Returns:
            DataFrame with columns:
                - timestamp (datetime in EET)
                - price_area
                - hour (0-23)
                - price_eur_per_mwh
                - price_eur_per_kwh

        Example:
            >>> client = NordPoolClient()
            >>> dam_prices = client.get_dam_prices(
            ...     price_area="TEL",
            ...     date=datetime(2026, 3, 19).date()
            ... )
            >>> print(dam_prices.describe())
        """
        if date is None:
            date = datetime.now().date()

        if not isinstance(date, (datetime, type(pd.Timestamp))):
            raise TypeError("date must be datetime or pd.Timestamp")

        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        try:
            data = self._make_request(
                "marketprices/PXO/results",
                params={
                    "priceArea": price_area,
                    "date": date_str,
                },
            )
        except requests.RequestException as e:
            logger.error(f"Failed to fetch DAM prices for {price_area} on {date_str}: {e}")
            raise

        rows = []
        for entry in data.get("results", []):
            for price_point in entry.get("prices", []):
                timestamp = pd.to_datetime(price_point["timestamp"]).tz_localize("UTC").tz_convert("EET")

                rows.append({
                    "timestamp": timestamp,
                    "price_area": price_area,
                    "hour": price_point.get("hour", -1),
                    "price_eur_per_mwh": float(price_point.get("price", 0)),
                    "price_eur_per_kwh": float(price_point.get("price", 0)) / 1000,
                })

        df = pd.DataFrame(rows)

        if df.empty:
            logger.warning(f"No DAM prices returned for {price_area} on {date_str}")
            return pd.DataFrame(columns=["timestamp", "price_area", "hour", "price_eur_per_mwh", "price_eur_per_kwh"])

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Fetched {len(df)} DAM prices for {price_area} on {date_str} | "
            f"Mean: {df['price_eur_per_mwh'].mean():.2f} EUR/MWh | "
            f"Range: {df['price_eur_per_mwh'].min():.2f}-{df['price_eur_per_mwh'].max():.2f}"
        )

        return df

    def get_dam_prices_range(
        self,
        price_area: str = TEL_PRICE_AREA,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Fetch DAM prices for date range with pagination.

        Args:
            price_area: Price area code
            start_date: Range start (default: 30 days ago)
            end_date: Range end (default: today)
            progress_callback: Optional callable(current_date, num_days) for progress tracking

        Returns:
            Concatenated DataFrame for entire date range

        Example:
            >>> client = NordPoolClient()
            >>> prices_30d = client.get_dam_prices_range(
            ...     price_area="TEL",
            ...     start_date=datetime(2026, 2, 17),
            ...     end_date=datetime(2026, 3, 19)
            ... )
            >>> print(f"Fetched {len(prices_30d)} hourly prices")
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        start_date = pd.Timestamp(start_date).normalize()
        end_date = pd.Timestamp(end_date).normalize()

        if start_date > end_date:
            raise ValueError("start_date must be before end_date")

        num_days = (end_date - start_date).days + 1
        all_prices = []

        for i, date in enumerate(pd.date_range(start_date, end_date, freq="D")):
            if progress_callback:
                progress_callback(date, num_days)

            try:
                df = self.get_dam_prices(price_area=price_area, date=date)
                if not df.empty:
                    all_prices.append(df)
            except Exception as e:
                logger.warning(f"Skipped {date.date()}: {e}")
                continue

        if not all_prices:
            logger.warning(f"No prices fetched for {price_area} in range {start_date} to {end_date}")
            return pd.DataFrame(columns=["timestamp", "price_area", "hour", "price_eur_per_mwh", "price_eur_per_kwh"])

        result = pd.concat(all_prices, ignore_index=True)
        result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Fetched {len(result)} total prices for {price_area} over {num_days} days | "
            f"Mean: {result['price_eur_per_mwh'].mean():.2f} EUR/MWh"
        )

        return result

    def get_idm_statistics(
        self,
        price_area: str = TEL_PRICE_AREA,
        date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch intraday market (IDM) hourly statistics.

        Args:
            price_area: Price area code
            date: Date to fetch (default: today)

        Returns:
            DataFrame with IDM statistics

        Example:
            >>> client = NordPoolClient()
            >>> idm_stats = client.get_idm_statistics(
            ...     price_area="TEL",
            ...     date=datetime(2026, 3, 19).date()
            ... )
        """
        if date is None:
            date = datetime.now().date()

        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        try:
            data = self._make_request(
                "marketprices/PI/results",
                params={
                    "priceArea": price_area,
                    "date": date_str,
                },
            )
        except requests.RequestException as e:
            logger.error(f"Failed to fetch IDM statistics for {price_area} on {date_str}: {e}")
            raise

        rows = []
        for entry in data.get("results", []):
            for hour_data in entry.get("hours", []):
                timestamp = pd.to_datetime(entry.get("date")).replace(
                    hour=hour_data.get("hour", 0)
                ).tz_localize("UTC").tz_convert("EET")

                rows.append({
                    "timestamp": timestamp,
                    "hour": hour_data.get("hour", -1),
                    "num_trades": hour_data.get("numberOfTrades", 0),
                    "min_price_eur_per_mwh": float(hour_data.get("minPrice", 0)),
                    "max_price_eur_per_mwh": float(hour_data.get("maxPrice", 0)),
                    "weighted_avg_price_eur_per_mwh": float(hour_data.get("vWAP", 0)),
                    "volume_mwh": float(hour_data.get("volume", 0)),
                })

        df = pd.DataFrame(rows)

        if df.empty:
            logger.warning(f"No IDM statistics returned for {price_area} on {date_str}")
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} IDM hourly statistics for {price_area} on {date_str}")

        return df

    def get_brm_forward_prices(
        self,
        contract_type: str = "monthly",
        lookback_months: int = 12,
    ) -> pd.DataFrame:
        """
        Fetch BRM (Bilaterals-Romania-Market) forward price indices.

        Args:
            contract_type: "daily" | "weekly" | "monthly" | "quarterly" | "seasonal" | "annual"
            lookback_months: Historical depth (default 12 months)

        Returns:
            DataFrame with forward prices

        Example:
            >>> client = NordPoolClient()
            >>> brm_monthly = client.get_brm_forward_prices(
            ...     contract_type="monthly",
            ...     lookback_months=12
            ... )
        """
        try:
            data = self._make_request(
                "marketprices/Forwards/results",
                params={
                    "commodityCode": "ENOBRM",  # Romania BRM
                    "contractType": contract_type,
                    "deliveryMonth": (datetime.now() - timedelta(days=30*lookback_months)).strftime("%Y%m"),
                },
            )
        except requests.RequestException as e:
            logger.error(f"Failed to fetch BRM forward prices: {e}")
            raise

        rows = []
        for entry in data.get("results", []):
            rows.append({
                "contract": entry.get("contract"),
                "delivery_period": entry.get("delivery"),
                "price_eur_per_mwh": float(entry.get("price", 0)),
                "change_pct": float(entry.get("percentChange", 0)),
                "timestamp": pd.to_datetime(entry.get("time")).tz_localize("UTC").tz_convert("EET"),
            })

        df = pd.DataFrame(rows)

        if df.empty:
            logger.warning(f"No BRM forward prices returned for {contract_type}")
            return pd.DataFrame()

        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} BRM forward prices ({contract_type})")

        return df

    def calculate_basis_spread(
        self,
        tel_prices: pd.Series,
        eex_prices: Optional[pd.Series] = None,
    ) -> float:
        """
        Calculate basis spread (TEL DAM - EEX DAM).

        Args:
            tel_prices: Romania (TEL) DAM prices
            eex_prices: Germany (EEX) DAM prices (fetched if not provided)

        Returns:
            Mean basis spread (EUR/MWh), positive if TEL > EEX

        Example:
            >>> client = NordPoolClient()
            >>> tel_prices = client.get_dam_prices("TEL")["price_eur_per_mwh"]
            >>> basis = client.calculate_basis_spread(tel_prices)
        """
        if eex_prices is None:
            try:
                eex_df = self.get_dam_prices("DE")
                eex_prices = eex_df["price_eur_per_mwh"]
            except Exception as e:
                logger.warning(f"Failed to fetch EEX prices for basis calculation: {e}")
                return 0.0

        if len(tel_prices) != len(eex_prices):
            logger.warning("Price series have different lengths; cannot calculate basis")
            return 0.0

        basis_spread = (tel_prices - eex_prices).mean()

        logger.info(f"Basis spread (TEL-EEX): {basis_spread:.2f} EUR/MWh")

        return basis_spread

    def close(self) -> None:
        """Close client session."""
        self.session.close()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def fetch_dam_prices_and_basis(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Convenience function: fetch DAM prices for TEL and EEX, calculate basis.

    Args:
        start_date: Range start
        end_date: Range end

    Returns:
        Tuple of (tel_prices_df, eex_prices_df, basis_spread_eur_per_mwh)

    Example:
        >>> tel_prices, eex_prices, basis = fetch_dam_prices_and_basis(
        ...     start_date=datetime(2026, 2, 17),
        ...     end_date=datetime(2026, 3, 19)
        ... )
        >>> print(f"Basis spread: {basis:.2f} EUR/MWh")
    """
    client = NordPoolClient()

    try:
        tel_prices = client.get_dam_prices_range(
            price_area="TEL",
            start_date=start_date,
            end_date=end_date,
        )

        eex_prices = client.get_dam_prices_range(
            price_area="DE",
            start_date=start_date,
            end_date=end_date,
        )

        # Calculate mean basis
        tel_mean = tel_prices["price_eur_per_mwh"].mean()
        eex_mean = eex_prices["price_eur_per_mwh"].mean()
        basis = tel_mean - eex_mean

        return tel_prices, eex_prices, basis

    finally:
        client.close()


# ============================================================================
# END OF NORD POOL CLIENT
# ============================================================================
