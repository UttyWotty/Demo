### RCA Demo (5 Whys + Pareto)

Portfolio-safe demo of the RCA pipeline using synthetic data. No Snowflake or models; outputs JSON and a simple HTML report under `data/`.

#### Quick start (CLI)
```bash
pip install -r requirements.txt
python run_demo.py
```

This will:
- Generate synthetic shot-level data with CT, APPROVED_CT, equipment, parts, and time fields
- Identify top targets (day-of-week, equipment, part) by CT issues
- Apply a simplified 5 Whys analysis for each target
- Save `data/five_whys_results.json` and `data/rca_report.html`

#### Streamlit app
```bash
streamlit run app.py
```

#### Screenshot
Add a screenshot after running the app:

![Dashboard](assets/screenshot.png)

Tip: Save a full-width screenshot as `assets/screenshot.png`.

#### Files
- `pipeline/generate_demo.py`: creates synthetic RCA dataset
- `analyzer.py`: simple Pareto-like targeting and 5 Whys engine
- `run_demo.py`: entrypoint that runs the full demo

#### Notes
- All synthetic; safe to publish.
- Add screenshots or extend with Streamlit later if desired.


