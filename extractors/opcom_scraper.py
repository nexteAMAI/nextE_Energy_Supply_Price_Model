"""
OPCOM Web Scraper — nextE Energy Supply
========================================
Extract market data from OPCOM (www.opcom.ro) reports and dashboards.

Data sources:
  - PZU (DAM) historical prices and results
  - PI (IDM) market results
  - PCCB-LE (Centralized Bilateral Market - Long-term Bilateral)
  - PCCB-NC (Centralized Bilateral Market - Short-term Bilateral)
  - GC Market (Green Certificates market) prices and volumes
  - Daily balancing market results
  - Monthly market transparency reports

HTML parsing: BeautifulSoup
PDF extraction: PyPDF2 (for monthly reports)
Timezone: EET (Eastern European Time)

Author: nextE AI Workstation
Version: 1.0.0
Date: 2026-03-19
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logger_warning = logging.warning("BeautifulSoup4 not installed; HTML parsing disabled")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# CONSTANTS
# ============================================================================

OPCOM_BASE_URL = "https://www.opcom.ro"
OPCOM_API_ENDPOINT = "https://www.opcom.ro/api/statistics"

REQUEST_TIMEOUT_SEC = 30
MAX_RETRIES = 3
MIN_REQUEST_INTERVAL_SEC = 1  # Be respectful: 1 request/sec to OPCOM


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OPCOMPZUResult:
    """Day-Ahead Market (PZU) price and volume result."""
    date: datetime
    hour: int
    price_eur_per_mwh: float
    volume_mwh: float
    buy_orders_mwh: float
    sell_orders_mwh: float
    implicit_auction_volume_mwh: float


@dataclass
class OPCOMBilateralAuctionResult:
    """Centralized Bilateral Market (PCCB) auction result."""
    auction_date: datetime
    auction_type: str  # "LE" (long-term) or "NC" (short-term)
    settlement_date: datetime
    num_contracts: int
    volume_mwh: float
    min_price_eur_per_mwh: float
    max_price_eur_per_mwh: float
    weighted_avg_price_eur_per_mwh: float


@dataclass
class OPCOMGCMarketResult:
    """Green Certificate market price and volume."""
    date: datetime
    price_eur_per_gc: float
    volume_gc: float
    transactions: int
    min_price_eur_per_gc: float
    max_price_eur_per_gc: float


# ============================================================================
# SCRAPER IMPLEMENTATION
# ============================================================================

class OPCOMScraper:
    """
    Web scraper for OPCOM market data.

    Handles HTML/PDF parsing, Romanian text interpretation, and data normalization.
    """

    def __init__(self, rate_limit_sec: float = MIN_REQUEST_INTERVAL_SEC):
        """
        Initialize OPCOM scraper.

        Args:
            rate_limit_sec: Minimum interval between requests (default 1 sec)
        """
        self.base_url = OPCOM_BASE_URL
        self.session = self._create_session()
        self.rate_limit_sec = rate_limit_sec
        self.last_request_time = 0

        if BeautifulSoup is None:
            logger.warning("BeautifulSoup4 required for OPCOM scraping; install with: pip install beautifulsoup4")

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()

        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _enforce_rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)
        self.last_request_time = time.time()

    def _get_page(self, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Fetch page with rate limiting and error handling.

        Args:
            url: Full URL to fetch
            **kwargs: Additional requests.get parameters

        Returns:
            Response object or None if request fails
        """
        self._enforce_rate_limit()

        headers = {
            "User-Agent": "nextE-EnergyPricingEngine/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
        }

        try:
            logger.debug(f"GET {url}")
            response = self.session.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SEC,
                **kwargs
            )
            response.raise_for_status()
            return response

        except requests.RequestException as e:
            logger.error(f"OPCOM request failed ({url}): {e}")
            return None

    def get_pzu_daily_results(
        self,
        date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch PZU (Day-Ahead Market) daily results from OPCOM.

        Args:
            date: Date to fetch (default: yesterday, as today's results post at 14:00)

        Returns:
            DataFrame with columns:
                - date
                - hour (0-23)
                - price_eur_per_mwh
                - volume_mwh
                - buy_orders_mwh
                - sell_orders_mwh

        Example:
            >>> scraper = OPCOMScraper()
            >>> pzu = scraper.get_pzu_daily_results(
            ...     date=datetime(2026, 3, 18)
            ... )
            >>> print(pzu.describe())
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)

        date_str = pd.Timestamp(date).strftime("%Y%m%d")

        # OPCOM PZU results typically available at:
        # https://www.opcom.ro/en/rapoarte-pzu (reports page)
        url = f"{self.base_url}/en/rapoarte-pzu"

        response = self._get_page(url)
        if response is None:
            return pd.DataFrame()

        if BeautifulSoup is None:
            logger.error("BeautifulSoup4 required for HTML parsing")
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, "html.parser")

        # Look for download link to CSV/Excel file with date matching
        rows = []

        try:
            # Parse tables from page (OPCOM usually displays daily results in tables)
            tables = soup.find_all("table")

            for table in tables:
                for tr in table.find_all("tr")[1:]:  # Skip header row
                    tds = tr.find_all("td")
                    if len(tds) >= 6:
                        try:
                            hour = int(tds[0].text.strip())
                            price = float(tds[1].text.strip().replace(",", "."))
                            volume = float(tds[2].text.strip().replace(",", "."))
                            buy_orders = float(tds[3].text.strip().replace(",", "."))
                            sell_orders = float(tds[4].text.strip().replace(",", "."))

                            rows.append({
                                "date": pd.Timestamp(date),
                                "hour": hour,
                                "price_eur_per_mwh": price,
                                "volume_mwh": volume,
                                "buy_orders_mwh": buy_orders,
                                "sell_orders_mwh": sell_orders,
                            })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Skipped row in PZU table: {e}")
                            continue

        except Exception as e:
            logger.error(f"Error parsing PZU results: {e}")

        if not rows:
            logger.warning(f"No PZU results parsed for {date_str}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(f"Parsed {len(df)} PZU hourly results for {date_str}")

        return df

    def get_pzu_prices_range(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch PZU prices for a date range.

        Iterates through dates and fetches daily results.

        Args:
            start_date: Range start (default: 30 days ago)
            end_date: Range end (default: 2 days ago)

        Returns:
            Concatenated DataFrame
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now() - timedelta(days=2)

        all_results = []

        for date in pd.date_range(start_date, end_date, freq="D"):
            try:
                df = self.get_pzu_daily_results(date=date)
                if not df.empty:
                    all_results.append(df)
                    logger.info(f"Fetched PZU for {date.date()}")
            except Exception as e:
                logger.warning(f"Skipped {date.date()}: {e}")
                continue

        if not all_results:
            logger.warning(f"No PZU data fetched for range {start_date} to {end_date}")
            return pd.DataFrame()

        result = pd.concat(all_results, ignore_index=True)
        result = result.sort_values("date").reset_index(drop=True)

        logger.info(f"Fetched total {len(result)} PZU hourly prices")

        return result

    def get_bilateral_auction_results(
        self,
        auction_type: str = "LE",
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch PCCB (Centralized Bilateral Market) auction results.

        Args:
            auction_type: "LE" (long-term) or "NC" (short-term)
            lookback_days: Historical depth

        Returns:
            DataFrame with auction results

        Example:
            >>> scraper = OPCOMScraper()
            >>> bilateral = scraper.get_bilateral_auction_results(
            ...     auction_type="LE",
            ...     lookback_days=30
            ... )
        """
        if auction_type not in ["LE", "NC"]:
            raise ValueError("auction_type must be 'LE' or 'NC'")

        # OPCOM bilateral market results page
        url = f"{self.base_url}/en/rapoarte-pccb-{'le' if auction_type == 'LE' else 'nc'}"

        response = self._get_page(url)
        if response is None:
            return pd.DataFrame()

        if BeautifulSoup is None:
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, "html.parser")

        rows = []

        try:
            tables = soup.find_all("table")

            for table in tables:
                for tr in table.find_all("tr")[1:]:
                    tds = tr.find_all("td")
                    if len(tds) >= 8:
                        try:
                            auction_date_str = tds[0].text.strip()
                            auction_date = pd.to_datetime(auction_date_str, format="%d.%m.%Y").tz_localize("EET")

                            settlement_date_str = tds[1].text.strip()
                            settlement_date = pd.to_datetime(settlement_date_str, format="%d.%m.%Y").tz_localize("EET")

                            num_contracts = int(tds[2].text.strip())
                            volume = float(tds[3].text.strip().replace(",", "."))
                            min_price = float(tds[4].text.strip().replace(",", "."))
                            max_price = float(tds[5].text.strip().replace(",", "."))
                            avg_price = float(tds[6].text.strip().replace(",", "."))

                            rows.append({
                                "auction_date": auction_date,
                                "auction_type": auction_type,
                                "settlement_date": settlement_date,
                                "num_contracts": num_contracts,
                                "volume_mwh": volume,
                                "min_price_eur_per_mwh": min_price,
                                "max_price_eur_per_mwh": max_price,
                                "avg_price_eur_per_mwh": avg_price,
                            })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Skipped bilateral row: {e}")
                            continue

        except Exception as e:
            logger.error(f"Error parsing bilateral results: {e}")

        if not rows:
            logger.warning(f"No bilateral ({auction_type}) results parsed")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(f"Parsed {len(df)} bilateral ({auction_type}) auction results")

        return df

    def get_gc_market_data(
        self,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch Green Certificate market prices from OPCOM.

        Args:
            lookback_days: Historical depth

        Returns:
            DataFrame with GC prices and volumes

        Example:
            >>> scraper = OPCOMScraper()
            >>> gc_market = scraper.get_gc_market_data(lookback_days=30)
            >>> print(gc_market.describe())
        """
        url = f"{self.base_url}/en/rapoarte-gc"

        response = self._get_page(url)
        if response is None:
            return pd.DataFrame()

        if BeautifulSoup is None:
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, "html.parser")

        rows = []

        try:
            tables = soup.find_all("table")

            for table in tables:
                for tr in table.find_all("tr")[1:]:
                    tds = tr.find_all("td")
                    if len(tds) >= 6:
                        try:
                            date_str = tds[0].text.strip()
                            date = pd.to_datetime(date_str, format="%d.%m.%Y").tz_localize("EET")

                            price = float(tds[1].text.strip().replace(",", "."))
                            volume = float(tds[2].text.strip().replace(",", "."))
                            transactions = int(tds[3].text.strip())
                            min_price = float(tds[4].text.strip().replace(",", "."))
                            max_price = float(tds[5].text.strip().replace(",", "."))

                            rows.append({
                                "date": date,
                                "price_eur_per_gc": price,
                                "volume_gc": volume,
                                "transactions": transactions,
                                "min_price_eur_per_gc": min_price,
                                "max_price_eur_per_gc": max_price,
                            })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Skipped GC row: {e}")
                            continue

        except Exception as e:
            logger.error(f"Error parsing GC market data: {e}")

        if not rows:
            logger.warning("No GC market data parsed")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(f"Parsed {len(df)} GC market data points")

        return df

    def close(self) -> None:
        """Close scraper session."""
        self.session.close()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def fetch_opcom_market_snapshot(
    lookback_days: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch comprehensive OPCOM market snapshot.

    Args:
        lookback_days: Historical depth for all datasets

    Returns:
        Dict with keys: "pzu_prices", "bilateral_le", "bilateral_nc", "gc_market"

    Example:
        >>> snapshot = fetch_opcom_market_snapshot(lookback_days=30)
        >>> pzu = snapshot["pzu_prices"]
        >>> bilateral = snapshot["bilateral_le"]
    """
    scraper = OPCOMScraper()

    try:
        end_date = datetime.now() - timedelta(days=2)
        start_date = end_date - timedelta(days=lookback_days)

        return {
            "pzu_prices": scraper.get_pzu_prices_range(start_date, end_date),
            "bilateral_le": scraper.get_bilateral_auction_results("LE", lookback_days),
            "bilateral_nc": scraper.get_bilateral_auction_results("NC", lookback_days),
            "gc_market": scraper.get_gc_market_data(lookback_days),
        }
    finally:
        scraper.close()


def compare_market_prices(
    pzu_prices: pd.DataFrame,
    bilateral_prices: pd.DataFrame,
    frequency: str = "D",  # Daily
) -> pd.DataFrame:
    """
    Compare average PZU vs bilateral prices over time.

    Args:
        pzu_prices: PZU hourly prices
        bilateral_prices: Bilateral auction prices
        frequency: Aggregation frequency ("D"=daily, "W"=weekly, "M"=monthly)

    Returns:
        Comparison DataFrame with columns: date, pzu_avg, bilateral_avg, spread

    Example:
        >>> comparison = compare_market_prices(
        ...     pzu_prices,
        ...     bilateral_prices,
        ...     frequency="W"
        ... )
    """
    # Aggregate PZU to frequency
    pzu_agg = pzu_prices.set_index("date").resample(frequency)["price_eur_per_mwh"].mean()

    # Aggregate bilateral to frequency
    bilateral_agg = bilateral_prices.set_index("auction_date").resample(frequency)["avg_price_eur_per_mwh"].mean()

    # Combine
    comparison = pd.DataFrame({
        "pzu_avg_eur_per_mwh": pzu_agg,
        "bilateral_avg_eur_per_mwh": bilateral_agg,
    })

    comparison["spread_eur_per_mwh"] = comparison["pzu_avg_eur_per_mwh"] - comparison["bilateral_avg_eur_per_mwh"]

    return comparison.reset_index()


# ============================================================================
# END OF OPCOM SCRAPER
# ============================================================================
