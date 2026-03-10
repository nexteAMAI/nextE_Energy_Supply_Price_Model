"""
FX Rate Extractor.

Sources:
  - EUR/RON: BNR (National Bank of Romania) daily reference rate
  - USD/EUR: ECB (European Central Bank) reference rate
  - Pre-loaded: EURO_RON_Conversion_rate_01_01_2009_26_02_2026.csv

For the pre-loaded file, use extractors.data_loader.load_fx_eur_ron().
This module provides live BNR/ECB extraction for the daily refresh.
"""

import logging
from datetime import date, datetime
from typing import Optional, Union
from xml.etree import ElementTree

import pandas as pd
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

# BNR XML feed for EUR/RON
BNR_URL = "https://www.bnr.ro/nbrfxrates.xml"
# ECB daily exchange rates (CSV)
ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"


class FXClient:
    """Fetches live FX rates from BNR and ECB."""

    def get_eur_ron_latest(self) -> Optional[float]:
        """Fetch the latest EUR/RON rate from BNR XML feed."""
        try:
            resp = requests.get(BNR_URL, timeout=10)
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)

            # BNR namespace
            ns = {"bnr": "http://www.bnr.ro/xsd"}
            body = root.find(".//bnr:Body", ns)
            if body is None:
                # Try without namespace
                for cube in root.iter():
                    if cube.attrib.get("currency") == "EUR":
                        rate = float(cube.text)
                        logger.info("BNR EUR/RON latest: %.4f", rate)
                        return rate

            for cube in body.findall(".//bnr:Cube", ns):
                for rate_elem in cube.findall("bnr:Rate", ns):
                    if rate_elem.attrib.get("currency") == "EUR":
                        rate = float(rate_elem.text)
                        logger.info("BNR EUR/RON latest: %.4f", rate)
                        return rate

            logger.warning("EUR rate not found in BNR XML response")
            return None
        except Exception as e:
            logger.error("Failed to fetch BNR EUR/RON rate: %s", e)
            return None

    def get_usd_eur_latest(self) -> Optional[float]:
        """Fetch the latest USD/EUR rate from ECB."""
        try:
            resp = requests.get(ECB_URL, timeout=10)
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)

            ns = {"gesmes": "http://www.gesmes.org/xml/2002-08-01",
                  "ecb": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}

            for cube in root.iter():
                if cube.attrib.get("currency") == "USD":
                    usd_per_eur = float(cube.attrib["rate"])
                    rate = 1.0 / usd_per_eur  # USD/EUR = 1 / EUR/USD
                    logger.info("ECB USD/EUR latest: %.4f (EUR/USD: %.4f)", rate, usd_per_eur)
                    return rate

            logger.warning("USD rate not found in ECB XML response")
            return None
        except Exception as e:
            logger.error("Failed to fetch ECB USD/EUR rate: %s", e)
            return None

    def get_latest_rates(self) -> dict:
        """Fetch all FX rates needed by the model."""
        return {
            "eur_ron": self.get_eur_ron_latest(),
            "usd_eur": self.get_usd_eur_latest(),
            "timestamp": datetime.now().isoformat(),
        }


def convert_eur_to_ron(
    eur_values: pd.Series,
    fx_rates: pd.DataFrame,
) -> pd.Series:
    """
    Convert EUR series to RON using the closest available FX rate.

    Parameters
    ----------
    eur_values : pd.Series
        Values in EUR with DatetimeIndex
    fx_rates : pd.DataFrame
        FX rates with DatetimeIndex and 'eur_ron' column

    Returns
    -------
    pd.Series with RON values.
    """
    # Align dates (use date-level matching, forward-fill for weekends)
    eur_dates = eur_values.index.normalize()
    fx_daily = fx_rates["eur_ron"].reindex(eur_dates, method="ffill")
    fx_daily.index = eur_values.index  # Restore original index
    return eur_values * fx_daily
