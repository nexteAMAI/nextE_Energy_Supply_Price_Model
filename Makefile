# ==============================================================================
# RO Energy Pricing Engine — Makefile
# ==============================================================================

.PHONY: install test lint format extract process export backtest daily weekly clean

# --- Setup ---
install:
	pip install -r requirements.txt

# --- Pipeline ---
backtest:
	python pipeline.py --mode backtest

daily:
	python pipeline.py --mode daily

weekly:
	python pipeline.py --mode weekly

export:
	python pipeline.py --mode export-only

# --- Testing ---
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=processors --cov=extractors --cov=outputs --cov-report=html

# --- Code Quality ---
lint:
	ruff check .

format:
	black .

# --- Streamlit ---
dashboard:
	streamlit run streamlit_app/app.py --server.port 8501

# --- Cleanup ---
clean:
	rm -rf data/processed/*.csv data/processed/*.parquet data/processed/*.json
	rm -rf data/raw/*
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache htmlcov .coverage
