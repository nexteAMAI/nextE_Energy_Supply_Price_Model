"""
Data extraction connectors for EQ, ENTSO-E, Balancing Services, JAO, FX,
Nord Pool, and OPCOM APIs.

  --- Supply Extension (v1.0.0, 2026-03-19) ---
  nordpool_client:  BRM/Nord Pool DAM+IDM+Forward data (TEL area)
  opcom_scraper:    OPCOM bilateral + GC market web scraper
"""

# --- Supply Extension Exports ---
try:
    from .nordpool_client import NordPoolClient
    from .opcom_scraper import OPCOMScraper
    _SUPPLY_EXTRACTORS_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        f"Supply extension extractors not fully loaded: {e}. "
        "Existing extractors are unaffected."
    )
    _SUPPLY_EXTRACTORS_AVAILABLE = False
