# System Capacity & Care Load Analytics for Unaccompanied Children

End-to-end project: **Jupyter analysis → Streamlit dashboard**.

## What is in this folder

| File | Purpose |
|---|---|
| `HHS_Unaccompanied_Alien_Children_Program.csv` | Raw data (keep the file name exactly) |
| `UAC_Care_Load_Analysis.ipynb` | Full analysis: ingestion, validation, metrics, trends, stress, KPIs, recommendations |
| `app.py` | Streamlit dashboard (self-contained; same logic as the notebook) |
| `requirements.txt` | Packages for the dashboard / cloud deployment |
| `requirements-notebook.txt` | Packages for running the notebook locally |
| `figures/` | 12 PNG charts produced by the notebook  |
| `uac_daily_processed.csv`, `uac_kpis_by_period.csv` | Processed outputs (for appendix / Excel / Tableau) |

---

## Step 1 — Set up Python (once)

```bash
cd uac_project
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r requirements-notebook.txt
```

## Step 2 — Run the Jupyter analysis

```bash
jupyter notebook UAC_Care_Load_Analysis.ipynb
```
Then **Kernel → Restart & Run All**. It re-creates `figures/` and the two processed CSVs.
(The notebook shipped here is already executed, so you can read the results without running anything.)

## Step 3 — Run the Streamlit dashboard locally

```bash
streamlit run app.py
```
Opens at http://localhost:8501. Controls (left sidebar): date range, Daily/Weekly/Monthly granularity, load and flow metric toggles, 7/14-day rolling averages, strain-threshold percentile, optional CSV upload for newer data.
Tabs: System load overview · CBP vs HHS · Net intake & backlog · Stress & volatility · Data quality · Data explorer (with CSV download).

## Step 4 — Deploy (free, Streamlit Community Cloud)

1. Create a **public GitHub repo** and upload: `app.py`, `requirements.txt`, and `HHS_Unaccompanied_Alien_Children_Program.csv` (all in the repo **root**). Optionally add the notebook and README.
   ```bash
   git init && git add . && git commit -m "UAC care load analytics"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose the repo, branch `main`, main file path `app.py`.
4. (Optional) *Advanced settings* → pick Python 3.11 or 3.12.
5. Click **Deploy**. After a minute or two you get a public URL like `https://<name>.streamlit.app`.

Common deployment problems
* *FileNotFoundError for the CSV*: the CSV must be in the repo root with the exact file name.
* *ModuleNotFoundError*: the package is missing from `requirements.txt`.
* *App was changed but looks the same*: push to GitHub; the cloud app redeploys automatically (or use **Reboot app**).

## Metric definitions (identical in notebook and dashboard)

| Metric | Definition |
|---|---|
| Total System Load | CBP custody + HHS care |
| Net Daily Intake | Transfers into HHS − discharges from HHS |
| Care Load Growth Rate | Day-over-day % change in total load |
| Backlog indicator | 7-day average net intake > 0 for ≥ 7 consecutive days |
| Census build-up (supplementary) | 7-day average change in HHS census > 0 for ≥ 7 consecutive days |
| Strain window | 14-day average total load ≥ 75th percentile for ≥ 14 consecutive days (adjustable in the app) |
| Relief window | 7-day average total load falling week-over-week for ≥ 7 consecutive days |
| Care Load Volatility Index | Std. deviation of the daily growth rate (percentage points) |
| Backlog Accumulation Rate | Average net intake (children/day) on sustained-backlog days |
| Discharge Offset Ratio | Σ discharged ÷ Σ transferred (> 1 = discharges outpace transfers) |

## Data issues found (and how they are handled)

1. **450 blank rows** at the end of the file → dropped (720 real daily reports remain).
2. **`Children in HHS Care` is text** with commas (`"2,484"`) → converted to numbers.
3. **Dates are `Month DD, YYYY`, newest first** → parsed and sorted oldest → newest.
4. **355 of 1,075 calendar days have no report (67% coverage; almost all are Fridays and Saturdays)** → complete daily index built; census counts interpolated across gaps; daily flows are **not** invented (weekly/monthly views use the average per reported day).
5. **86 days where transfers > end-of-day CBP custody** → flagged, not deleted (custody is an end-of-day snapshot, so this can be legitimate). No days where discharges > HHS care.
6. **Net intake does not reconcile with the census** (correlation 0.22 between census change and transfers − discharges; in 2023 the census *rose* by 4,394 while transfers − discharges summed to −52,707). HHS evidently has inflows beyond CBP transfers. That is why the project adds a census-based build-up indicator next to the brief's backlog indicator, and why net intake should be read as a *pressure proxy*.
