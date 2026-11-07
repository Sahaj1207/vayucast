# 📘 LEARNINGS — everything this project teaches, from zero

This guide explains every idea used in VayuCast, **in the order the data flows through the project**. Each section has:

- **The idea** — in plain words
- **In this project** — the exact file and what it does
- **Try it** — a small experiment to make it stick
- **Go deeper** — a free resource

Don't read it all in one go. One section a day, with the code open next to it, is ideal.

---

## Part 0 — The tools

### 0.1 Virtual environments
**The idea:** every project gets its own private set of Python libraries, so projects don't break each other.
**In this project:** `.venv/`. You "enter" it with `.venv\Scripts\activate`.
**Try it:** run `pip list` inside and outside the environment. Compare.

### 0.2 Python packages and `python -m`
**The idea:** a folder with `__init__.py` is a *package*, and its files can import each other (`from src.config import CITIES`). `python -m src.train` runs a file *as part of the package*, so those imports work.
**In this project:** `src/` is the package. The API and dashboard import from it, so the logic exists in one place only.

### 0.3 Git and GitHub
**The idea:** Git saves snapshots ("commits") of your code. GitHub stores them online and runs automation.
**In this project:** `.gitignore` decides what is *not* saved (the big raw data, the virtual environment).
**Go deeper:** GitHub Skills → *Introduction to GitHub* (free, interactive).

---

## Part 1 — Getting data

### 1.1 APIs
**The idea:** an API is a URL that returns data instead of a web page. You send parameters, you get JSON back.
**In this project:** `src/fetch_data.py` calls three Open-Meteo APIs: air quality, historical weather and weather forecast.
**Try it:** paste this into your browser:
`https://air-quality-api.open-meteo.com/v1/air-quality?latitude=15.85&longitude=74.5&hourly=pm2_5&past_days=1`

### 1.2 Robust code: retries and timeouts
**The idea:** networks fail. Good code waits and tries again, instead of crashing.
**In this project:** `get_json()` retries 3 times with a growing wait.

### 1.3 Hourly → daily aggregation
**The idea:** turn many small rows into one summary row. Averages for pollution, a **sum** for rain (10 mm + 5 mm = 15 mm of rain, not 7.5).
**In this project:** `to_daily()`. Days with fewer than 18 of 24 hours are dropped, because an average of 3 hours isn't a daily average.

### 1.4 Domain knowledge: the Indian AQI
**The idea:** the AQI turns pollutant concentrations into one 0–500 number using lookup bands and straight-line interpolation. The worst pollutant decides.
**In this project:** `src/aqi.py`, with self-tests in `tests/test_aqi.py`.
**Lesson:** understanding the domain matters as much as ML. A data scientist who knows *why* PM2.5 matters builds better features.

---

## Part 2 — Understanding and cleaning data (EDA)

### 2.1 Exploratory Data Analysis
**The idea:** before modelling, *look*: shapes, missing values, distributions, patterns.
**In this project:** `notebooks/01_first_look.ipynb` teaches the 10 pandas commands you'll use forever (`head`, `info`, `describe`, filtering, `groupby`, `crosstab`, …).

### 2.2 Seasonality
**The idea:** many things repeat on a calendar. Here the monsoon (June–September) washes the air clean, and winter (November–January) traps pollution near the ground.
**In this project:** the Data insights tab in the dashboard. Delhi averages about 4× Belagavi's AQI.

### 2.3 Data-quality detective work
**The idea:** never trust data blindly. Ask: "is this value *physically possible*?"
**In this project:** on 14–15 December 2022 all cities showed near-zero pollution at once, including Delhi in winter. One city being clean after rain is real. Five or more cities across India at once is a glitch. `find_glitch_dates()` in `src/features.py` detects it with that exact logic.

### 2.4 Missing values: fill, drop or flag?
**The idea:** short gaps can be filled by interpolation (drawing a straight line between neighbours). Long gaps should stay empty — filling a whole week is inventing data.
**In this project:** `clean()` fills gaps of up to 3 days only.

---

## Part 3 — Thinking like a forecaster

### 3.1 Turning a time series into a table
**The idea:** ML models need rows of (features → answer). For forecasting, each row is *"the situation at the end of day t"*, and the answer is *"AQI on day t + h"*.
**In this project:** `build_features()`. Column `date` is day *t* and `target` is the AQI *h* days later.

### 3.2 Feature engineering
**The idea:** create inputs that make the pattern easy to learn. Often this matters more than the choice of algorithm.
**In this project (54 features):**

| Kind | Examples | Why it helps |
|---|---|---|
| Lags | `aqi_today`, `aqi_1d_ago`, `aqi_6d_ago` | Pollution today is the best clue for tomorrow |
| Rolling stats | `aqi_mean_7d`, `aqi_std_7d`, `aqi_max_7d` | Recent level and how jumpy it's been |
| Pollutants | `pm2_5_today`, `ozone_today`, `pm_ratio_today` | What *kind* of pollution |
| Weather today | `wind_speed_today`, `rain_sum_7d` | Is the air stagnant? Has it rained? |
| **Forecast weather** | `rain_target_day`, `wind_speed_target_day` | Rain and wind **tomorrow** clean the air **tomorrow** |
| Calendar | `target_month`, `target_doy_sin/cos`, `days_from_diwali` | Season and festivals |
| City | `city_delhi`, … (one-hot) | Each city has its own baseline |

**Cyclical encoding:** month 12 and month 1 are neighbours, but as plain numbers they're 11 apart. Using `sin` and `cos` of the day of year puts the calendar on a circle. The same trick handles wind direction (359° is next to 1°).

### 3.3 ⚠️ Data leakage — the most important lesson
**The idea:** leakage means the model accidentally sees information it wouldn't have in real life, usually from the future. It looks amazing in testing, then fails in production.
**In this project:**
- Every feature uses only data up to day *t*. Forecast weather and the calendar are allowed because they **are** known in advance.
- `tests/test_features.py::test_no_leakage_from_the_future` scrambles every value after day *t* and checks that the features for day *t* don't change. I tested the test: a sneaky "centered 7-day average" that peeks 3 days ahead **was caught**.
- **Honest caveat:** for history we used *observed* weather as a stand-in for the *forecast* weather. Real forecasts have errors, so live accuracy will be slightly worse than the test numbers. This is written in the README, because being upfront about limitations builds trust.

---

## Part 4 — Training and evaluating models

### 4.1 Baselines first
**The idea:** before any ML, ask "what would a dumb rule score?". If your model can't beat it, the model is useless.
**In this project:** `baseline_persistence` ("tomorrow = today") and `baseline_7day_mean`. Persistence is *hard* to beat for 1 day ahead.

### 4.2 Three model families
| Model | How it thinks | Strengths | Weaknesses |
|---|---|---|---|
| **Ridge regression** | A weighted sum of features (a straight line in many dimensions) | Fast, explainable, hard to overfit | Can't learn "if wind is high AND it's winter…" |
| **Random Forest** | 300 decision trees, each trained on a random sample, then averaged | Learns interactions, robust | Big files, can't extrapolate |
| **XGBoost** | Trees built one after another; each new tree fixes the previous trees' mistakes (*gradient boosting*) | Usually best on tables of data | More settings to tune |

**In this project:** `src/models.py`.

### 4.3 Predicting the *change* (a trick that helped a lot)
**The idea:** tree models can't predict values outside what they saw in training. Predicting **"how much will AQI change from today?"** is the same kind of question in Delhi (300) and Belagavi (40).
**In this project:** `DeltaModel` trains on `target − aqi_today` and adds today back at the end.

### 4.4 Train/test split for time series
**The idea:** never shuffle time-series data. Train on the past, test on the future, exactly like real life.
**In this project:** the last 365 days (Nov 2025 → Oct 2026) are the **test set**, locked away until the very end.

### 4.5 Cross-validation (time-based)
**The idea:** to *choose* a model fairly, you need several validation scores — not the test set, because looking at the test set to choose is also cheating.
**In this project:** `time_folds()` makes 4 expanding windows. Train on older data, validate on the next 90 days, slide forward, repeat. The model with the best **average cross-validation** error wins. The test set is used once, only to report.

### 4.6 Metrics
- **MAE** (mean absolute error): the average miss in AQI points. Easy to explain ("off by 16 on average").
- **RMSE:** like MAE, but punishes big misses more. RMSE ≫ MAE means occasional large misses (Delhi spikes).
- **R²:** the share of the variation explained. 1 is perfect, 0 is no better than guessing the average.
- **Category accuracy:** did we get "Good / Moderate / Poor…" right? That's what a user actually cares about.

### 4.7 Our real results (test year, never seen in training)

| Days ahead | Average error (MAE) | vs "same as today" | Right category | R² |
|---|---|---|---|---|
| 1 | **±15.8 AQI** | **16% better** | 78% | 0.88 |
| 2 | ±21.3 | 22% better | 72% | 0.81 |
| 3 | ±23.1 | 28% better | 68% | 0.79 |

**Story of how we got here** (worth telling in interviews):
- *Version 1* (past data only) beat persistence by only 1–12%, and cross-validation picked a different winner than the test set.
- *Version 2* added **forecast weather** and the **change-based target**. XGBoost won *both* cross-validation and the test, and the improvement grew to 16–28%.
- Lesson: the biggest gain came from **better features and framing**, not a fancier algorithm.

**Adding cities (6 → 9: Hyderabad, Kurnool, Lucknow):**
- Overall error improved slightly (16.1 → 15.8). More cities means more examples of how air behaves, and **one shared model learns from all of them**. That's why we train one model with a city column instead of one model per city.
- Error depends mostly on **how variable a city's air is**, not on the model: Belagavi ±7, Kurnool ±8, Hyderabad ±10, Mumbai ±14, Lucknow ±28, Delhi ±49. Always report errors **per group**, because an overall average can hide where a model struggles.
- Adding a city changes the input columns (`city_lucknow`…), so the old saved model can't be used and you **must retrain**.

### 4.8 The leakage trap, measured
Same XGBoost model, two ways of testing it:
- Shuffled K-fold: **MAE 12.1** ← looks great, but it's a lie
- Time-based: **MAE 15.8** ← the truth

Shuffling lets the model train on days right next to (and after) the days it's tested on. **Always validate the way you'll actually use the model.**

---

## Part 5 — Trust: uncertainty and explainability

### 5.1 Prediction intervals (quantile regression)
**The idea:** one number isn't enough. "AQI 290" vs "AQI 250–330" are very different messages. A *quantile* model predicts a percentile: the 10th and 90th percentile models together give an 80% range.
**In this project:** `make_quantile_model()` in `src/models.py`.
**Honest result:** the "80%" ranges actually contained the true value on **73–76%** of days, so they're a bit too narrow (over-confident).
**Next step to learn:** *conformal prediction* — widen the intervals using validation errors so the coverage really is 80%.

### 5.2 SHAP (explainable AI)
**The idea:** for each prediction, SHAP splits it into contributions: "today's AQI pushed it −23, forecast wind pushed it −8…". Based on Shapley values from game theory (fair credit-sharing between players).
**In this project:** `reports/figures/shap_summary.png`, plus live "why this forecast?" bars on the dashboard.
**What it learned (and it makes physical sense):**
- **High AQI today → forecast comes down** (spikes don't last: *mean reversion*)
- **Strong wind tomorrow → lower AQI** (pollution blown away)
- **Rain tomorrow → lower AQI** (washed out)
- **High air pressure → higher AQI** (high pressure means still air that traps pollution)

When the explanations match real physics, you can trust the model more.

---

## Part 6 — Putting it into production (MLOps)

### 6.1 Saving models
**The idea:** train once, save to a file, load it anywhere.
**In this project:** `joblib.dump()` saves `models/vayucast_models.joblib` (9 models: point + low + high, for each of 3 horizons). The final models are retrained on **all** data, since the test year was only for measuring.
**Gotcha:** a saved model may not load with different library versions, so `scikit-learn` and `xgboost` are **pinned** in `requirements.txt`.

### 6.2 Training/serving consistency
**The idea:** the live system must build features exactly the same way as training did, otherwise predictions are silently wrong.
**In this project:** training and live forecasting both call the same `clean()` and `build_features()`. I also checked that live data and historical data are **identical** on overlapping days (they were, to 0.00).

### 6.3 REST API (FastAPI)
**The idea:** wrap the model in a web service, so any app (website, mobile, other code) can ask for forecasts.
**In this project:** `api/main.py` serves `/health`, `/cities`, `/forecast/{city}` and `/metrics`. FastAPI generates interactive docs at `/docs` automatically. Results are cached for 30 minutes so the weather APIs aren't called on every request.
**Docker:** the `Dockerfile` packs the API and its dependencies into a container that runs the same everywhere.

### 6.4 Dashboard (Streamlit)
**The idea:** a web app written only in Python. Good for showing ML to non-programmers.
**In this project:** `app/dashboard.py` has 5 tabs: forecast cards in official CPCB colours with health advice, all cities, model performance, data insights and monitoring.

### 6.5 Testing
**The idea:** code that checks code. It catches bugs before users do.
**In this project:** 33 tests in `tests/`, covering AQI maths, **no leakage**, cleaning, model wrapper, drift maths and the API (with a fake forecast, so tests don't need the internet).

### 6.6 Automation (CI/CD with GitHub Actions)
**The idea:** robots run your scripts on a schedule or on every push.
**In this project (`.github/workflows/`):**
- `ci.yml` — runs the tests on every push
- `daily-forecast.yml` — every morning at 07:00 IST, forecasts all cities, scores old forecasts, checks drift, and commits the results
- `weekly-retrain.yml` — every Sunday, downloads fresh data, retrains, runs the tests, and only commits the new model if the tests pass

### 6.7 Monitoring and data drift — the hardest lesson in the project
**The idea:** a model that worked last year can quietly break if the world changes (*drift*). You watch two things:
1. **Live accuracy:** every forecast is logged (`data/predictions/forecast_log.csv`) and scored once its day has passed. This is the only number that truly matters.
2. **Input drift:** are today's inputs unusual compared with what the model was trained on?

**What actually happened building this (a real debugging story):**
1. First try: the Population Stability Index (PSI), the textbook drift score, flagged **every** feature as "big drift". Suspicious.
2. I checked for a pipeline bug by comparing live and historical data on the same days: identical. Not a bug.
3. Cause 1, **seasonality:** comparing November with the whole year always looks like drift. Fix: compare each city with **its own same season** in other years.
4. Cause 2, **autocorrelation:** 30 days in a row aren't 30 independent samples. Even *normal* past years produced PSI of 1–3 on these windows, far above the textbook 0.25 alert line.
5. Final design: the main signal is **"how many standard deviations is this month from normal for this city and season"**. PSI is kept for reference, alongside the range seen in normal years.
6. The result makes sense: *"October 2026 was unusually warm (+1.9 SD); everything else normal."* Retraining is only recommended for extreme drift or when live error gets 50% worse than the test error. **Drift alone isn't a reason to retrain — worse accuracy is.**

The lesson: textbook thresholds assume independent samples. Always check whether a method's assumptions fit your data.

---

## Part 7 — How to talk about this project

**30-second pitch:**
> "I built VayuCast, an end-to-end ML system that forecasts air quality 1–3 days ahead for nine Indian cities. It pulls four years of pollution and weather data, engineers 54 time-series features, and uses XGBoost, which beats the 'same as today' baseline by 16–28% on a held-out year. It gives uncertainty ranges and SHAP explanations, runs behind a FastAPI service and a Streamlit dashboard, retrains weekly with GitHub Actions, and monitors live accuracy and seasonal drift."

**Questions you should now be able to answer:**
1. Why can't you shuffle time-series data for validation? *(Part 4.8: the 12.1 vs 15.8 example)*
2. What is data leakage, and how did you prevent it? *(3.3, plus the scramble test)*
3. Why compare against a baseline? *(4.1)*
4. Why XGBoost over Random Forest? *(4.2, chosen by cross-validation, not by the test set)*
5. Why predict the change rather than the value? *(4.3)*
6. How do you know when to retrain? *(6.7)*
7. What would you improve? *Calibrate the intervals (conformal prediction); compare with CPCB ground stations; use real archived weather forecasts instead of observed weather; add more cities; try LightGBM or a temporal neural network.*

---

## Part 8 — Study path (free resources, in order)

| Week | Topic | Resource |
|---|---|---|
| 1 | Python basics | Kaggle Learn → *Python*; or CS50P (Harvard, free) |
| 1 | Git and GitHub | GitHub Skills → *Introduction to GitHub* |
| 2 | pandas and charts | Kaggle Learn → *Pandas*, *Data Visualization*; this repo's Lesson 1 notebook |
| 3 | Statistics intuition | StatQuest (YouTube): mean/SD, distributions, correlation |
| 4 | ML fundamentals | Kaggle Learn → *Intro to ML*, *Intermediate ML* |
| 5 | Time series | Kaggle Learn → *Time Series* |
| 5–6 | Deeper ML | Andrew Ng's *Machine Learning Specialization* (Coursera, free to audit); *Hands-On Machine Learning* by Aurélien Géron, chapters 2, 4, 6, 7 |
| 7 | Explainability | SHAP docs → *An introduction to explainable AI with Shapley values* |
| 8 | APIs and apps | FastAPI official tutorial; Streamlit *Get started* |
| 9 | MLOps | DataTalksClub *MLOps Zoomcamp* (free); MLflow *Tracking quickstart* |

**Best way to learn from this repo:** pick one file, read it top to bottom, then **break something on purpose** (remove a feature, change a threshold, shuffle the split) and see what happens to the numbers. Write what you learned in the log below.

## 📒 My learning log

| Date | What I learned | What confused me | How I solved it |
|---|---|---|---|
| | | | |
