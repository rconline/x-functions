# Notebooks

The acceptance notebooks (§15) are checked in as Python scripts with Jupytext
"percent"-format cell markers. Convert to `.ipynb` via:

```bash
pip install jupytext
jupytext --to ipynb 01_databricks_parity_demo.py
jupytext --to ipynb 02_governance_demo.py
```

Sources:
- [`01_databricks_parity_demo.py`](01_databricks_parity_demo.py)
- [`02_governance_demo.py`](02_governance_demo.py)
