# Analysis Intelligence

Cricket Performance & Pressure Intelligence. Streamlit + Pandas + Plotly.

## Run locally
```
cd analysis_intelligence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. The six CSV files are in `data/`.

## Regenerate the analytics (about one minute)
```
python -m engine.run
```
Writes `analytics/` (rankings, per-innings traces, ball-level pressure values, meta.json, validation.json, TOP10.md). The website only reads these files.

## Layout
- `app.py` entry point and navigation
- `views/` one module per page (named `views`, not `pages`, so Streamlit's auto page discovery stays off)
- `utils/data_loader.py` the only place CSVs are read; verified cleaning rules and descriptive aggregates
- `utils/ui.py`, `utils/visualizations.py` shared HTML components and Plotly theme
- `engine/` pressure engine: `prep.py` (state + exclusions), `core.py` (A, WP, leverage, PI, out-of-fold), `scoring.py` (shrinkage, rankings), `validate.py`, `run.py`
- `analytics/` engine outputs read by the website
- `assets/style.css` all styling; colours are CSS variables at the top of the file

## Status
Live: data layer, player profiles, pressure index, reliability rankings (IPL primary, Emerging prospects, T20I separate), validation, methodology.
