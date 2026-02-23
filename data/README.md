# data/

This folder holds all CSV data files used by the project.

## Files

| File | Description | How to get it |
|---|---|---|
| `TCS_stock_history.csv` | Historical TCS stock data (main dataset) | Run `python src/download_latest_tcs_data.py` and rename, or provide your own |
| `tcs_stock_latest.csv` | Auto-generated latest data | Auto-created by `src/download_latest_tcs_data.py` |

## Quick Download

From the project root, run:

```bash
python src/download_latest_tcs_data.py
```

This will download the last 5 years of TCS stock data from Yahoo Finance
and save it as `data/tcs_stock_latest.csv`.

To use it as the main dataset, either:
- Copy / rename it to `TCS_stock_history.csv`, OR
- Update the path in `src/data_loader.py`

## Expected CSV Structure

```
Date,Open,High,Low,Close,Volume,Dividends,Stock Splits
2019-04-01,2013.5,2025.0,2000.0,2018.75,3245678,0.0,0.0
...
```
