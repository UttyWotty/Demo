### CT Efficiency Benchmarking Demo

Portfolio-safe demo that reproduces the supplier benchmarking workflow using synthetic data (no Snowflake or real data required).

#### What it does
- Generates a synthetic `MASTER_SHOT_TABLE`-like dataset
- Computes cycle-time efficiency metrics per tool and supplier
- Normalizes within tooling families and ranks suppliers
- Outputs interactive HTML charts and an HTML report under `data/`

#### Quick start
```bash
pip install -r requirements.txt
python run_demo.py
```

This will create HTML visualizations and a report in `data/` (auto-created).

#### Files
- `pipeline/generate_demo.py`: synthetic data generator
- `analyzer.py`: benchmarking logic without external connections
- `run_demo.py`: entrypoint to run the full demo

#### Notes
- No credentials needed; all data is generated locally.
- Safe to publish in a portfolio.


