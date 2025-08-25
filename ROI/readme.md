### ROI Demo App (Streamlit)

Self-contained demo using synthetic data (no external services or secrets). Safe for portfolio.

#### Features
- Synthetic ROI-like dataset (suppliers, tools, daily metrics)
- CT trend, efficiency distribution, daily net ROI
- Recommended actions based on KPIs

#### Screenshot
Add a screenshot after running the app:

![Dashboard](assets/screenshot.png)

Tip: Run the app, then save a full-width screenshot as `assets/screenshot.png`.

#### Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional:
```bash
cp .env.example .env  # adjust MACHINE/LABOR rates if desired
```

#### Configuration
- Sidebar controls for days, suppliers, tools, and random seed
- Machine + labor hour rate for ROI math

#### Tech
- Python, Streamlit, Plotly, pandas, numpy
- No Snowflake or credentials required

#### License
MIT — see `LICENSE` for details.