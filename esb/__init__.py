"""nextE Energy Supply Bid Management Tool - engine package.

Module build order (Phase 3): grid -> importer -> imbalance -> scenarios -> sources
-> merit_order -> pnl -> cashflow -> guarantees -> pricing -> reporting.
Each module replicates a named region of the Reference Case workbook; the mapping is
documented in docs/METHODOLOGY.md.
"""

__version__ = "0.4.0"
