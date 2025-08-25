### Product Analysis Demo (Session Analytics)

Portfolio-safe demo of the production session analytics using synthetic data only (no Snowflake).

#### What it does
- Generates synthetic shot-level data with `SUPPLIER_NAME`, `EQUIPMENT_CODE`, `LOCAL_SHOT_TIME`, `ACTUAL_CT`, `APPROVED_CT`
- Segments shots into sessions based on 8-hour gaps
- Detects stops using a mode-based threshold on inter-shot time
- Computes per-session KPIs: uptime, downtime, duration, stops, avg run duration, etc.

#### Quick start (CLI)
```bash
pip install -r requirements.txt
python run_demo.py
```

Outputs are written under `data/` (auto-created):
- `session_summary.csv`
- `product_session_report.html`

#### Streamlit app
```bash
streamlit run app.py
```

#### Screenshot
Add a screenshot after running the app:

![Dashboard](assets/screenshot.png)

Tip: Save a full-width screenshot as `assets/screenshot.png`.


