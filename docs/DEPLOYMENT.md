# UNHOOK / CAPSTONE PROJECT

## Machine Learning Project Documentation — Deployment

*An AI-driven progressive bandwidth throttling and adaptive human-in-the-loop intervention concept*

### Document control

| FIELD | VALUE |
|---|---|
| Project | UNHOOK \| Android digital wellbeing capstone (SIC AI Capstone, Group 12) |
| Report date | 17 November 2026 |
| Scope | Overview; Model Serialization; Model Serving; API Integration; Security Considerations; Monitoring and Logging (template [1]) |
| Evidence base | Step 9 Final Technical Report [2] (primary source for the deployed model); UNHOOK repository code and reports [4]–[7] (repository evidence) |
| Model status | Offline prototype evaluated; no integrated Android/VPN deployment [2] |

### Executive summary

The model covered by this document is the final locked UNHOOK model, as documented in the Step 9 Final Technical Report [2]: a class-weighted Logistic Regression pipeline (median imputation → constant-feature filter → ANOVA top-10 feature selection → standardization → Logistic Regression, C = 2.0, `class_weight="balanced"`) with a frozen decision threshold of 0.55. It predicts `y_next_day_behavior_change_proxy`, an experimental label for an unusually large next-day smartphone-use deviation. On the independent held-out test (611 rows, 43 users) it reached Macro F1 0.6210 and positive-class F1 0.4089 [2].

According to [2], Step 8 delivered an **offline inference prototype, not an Android application**: a frozen, hash-verified joblib pipeline, a portable 10-feature Python scorer, and a Kotlin numeric core tested on the JVM. Android data collection, on-device feature extraction, intervention logic, and VPN throttling remain future work. No API server, authentication layer, or production monitoring exists. Every component below is therefore labelled **Delivered** (documented in [2] or present in the repository) or **Proposed only** (planned / proposed approach, not implemented).

> **Note on the two model streams.** The repository [4]–[7] contains a *different* modelling stream from the Step 9 locked model: a rule-based daily risk score (`src/risk.py`, thresholds in `config/risk_thresholds.json`) and a next-day forecast experiment (`src/forecast.py`) on 7,394 participant-days from 274 participants. That experiment uses a different target (next-day risk ≥ "Dikkat"), and its models are fitted only inside cross-validation and never saved. The Step 9 model uses 4,054 participant-days from 284 users. The Step 6–8 artifacts that hold the locked model (`unhook_step6_train_only_locked_pipeline.joblib`, `feature_columns.txt`, the portable JSON model, and the Kotlin source) are listed in [2], Appendix B, but they are **not included in this repository**. Repository files are cited below only as supporting evidence.

---

## 1. Overview

The deployment phase aimed to turn the locked Step 6 model into something reusable outside the training notebooks, without claiming real-world readiness. As documented in [2] (Test Submission §5–6), the steps taken were:

1. **Model locking and freezing.** The final pipeline was fitted on all 2,784 training rows (198 users) only. It was then stored as a joblib artifact and verified by SHA-256 hash.
2. **Frozen-model application.** The same artifact was applied unchanged to the 659-row validation split and the 611-row held-out test split. No refitting or threshold change was made.
3. **Portable export.** A mathematically equivalent 10-feature linear scoring definition was exported for pure-Python and Kotlin use.
4. **Interface definition.** A request/response contract and error-handling rules were documented. Input schemas, synthetic fixtures, and JSON/CSV command-line scoring were delivered.
5. **Parity testing.** The portable Python and Kotlin scorers were checked against the locked model.

| COMPONENT | STATUS | SOURCE |
|---|---|---|
| Frozen Python model (joblib pipeline, SHA-256 verified) | **Delivered** | [2] Test Submission §5 |
| Portable Python scorer (659/659 validation predictions matched) | **Delivered** | [2] Test Submission §5–6 |
| Kotlin numeric core (12/12 synthetic JVM parity tests) | **Delivered** | [2] Test Submission §5–6 |
| Android UsageStats data pipeline | **Proposed only** | [2] Test Submission §5 |
| In-app interventions / JITAI | **Proposed only** — not connected to model prediction | [2] Test Submission §5 |
| VpnService bandwidth throttling | **Proposed only** — not implemented or tested on-device | [2] Test Submission §5; [10] |
| HTTP prediction API | **Proposed only** — see Section 4 | this document |

The target deployment is **on-device (edge) inference on Android**, as planned in the concept note [3] (§8.1, §8.5). The planned architecture, as stated in [2] (Test Submission §5), is:

```
explicit permissions
  → local usage-event collection
  → validated feature extractor
  → portable next-day proxy scorer
  → separately designed and prospectively validated intervention logic
  → user feedback interface
  → independent VPN traffic-control module
```

Automatic interventions must remain disabled until false-positive burden, privacy protections, and operational behavior are tested [2].

## 2. Model Serialization

### 2.1 Delivered: joblib pipeline (primary model)

As documented in [2], the trained scikit-learn pipeline was serialized with **joblib** [8] as `unhook_step6_train_only_locked_pipeline.joblib`. The serialized object contains the full preprocessing chain (imputation medians, variance filter, ANOVA selector, scaler) and the fitted classifier. As a result, test-time inputs are transformed only with values learned from training data. As shown in the Step 7 runner [2] (code below), the 0.55 threshold is applied to the `predict_proba` output. The artifact is stored with a **SHA-256 hash**, and [2] reports that the frozen model hash matched the original Step 7 model.

The input feature order is fixed by `feature_columns.txt` (30 columns). `participant_id` and `local_date` are kept outside the predictor matrix [2].

The loading code below is quoted from [2] (Test Submission §3, Step 7 Python runner):

```python
import joblib
import pandas as pd
model = joblib.load('unhook_step6_train_only_locked_pipeline.joblib')
feature_columns = open('feature_columns.txt').read().splitlines()
X_test = pd.read_csv('test_features.csv')[feature_columns]
proxy_probability = model.predict_proba(X_test)[:, 1]
proxy_flag = (proxy_probability >= 0.55).astype(int)
# Test labels are used only for evaluation, never refitting.
```

### 2.2 Delivered: portable linear model (efficient storage)

The main storage consideration was portability to a mobile runtime. The joblib file requires Python and scikit-learn, which an Android device does not provide. For this reason, a **mathematically equivalent 10-feature linear scoring definition** was exported [2] (`portable_linear_model.json`, listed in [2], References [6]). [2] states that the portable logistic calculation uses training-only medians, scaler parameters, model coefficients, and the unchanged threshold. The scoring rule, as given in [2] (Test Submission §6):

```python
z = intercept + sum(
    coefficient[i] * ((value[i] or train_median[i]) - mean[i]) / scale[i]
    for i in range(10)
)
p = 1 / (1 + exp(-z))
predicted_next_day_proxy = int(p >= 0.55)
# Deliberate safety contract:
automatic_intervention_authorized = False
```

The 10 input features, in the order recorded by the locked configuration [2] (Appendix A):

```
1. night_unlock_events_00_06          6. has_notification_events
2. night_unlock_ratio                 7. prior20_median_night_unlock_ratio
3. night_nonstudy_app_events          8. prior20_unlock_delta
4. night_nonstudy_app_ratio           9. prior20_unlock_ratio
5. notification_posted_nonstudy      10. prior20_app_delta
```

### 2.3 Repository evidence

| ITEM | WHAT IT SHOWS | STATUS |
|---|---|---|
| `src/calibrate.py:81-84` | Writes `config/risk_thresholds.json`: thresholds (Dikkat 0.3404, Yüksek 0.5418, Kritik 0.7375) and all risk-score parameters as JSON | **Delivered** (rule-based risk score, not the Step 9 model) |
| `src/forecast.py:109-113` | Fits Logistic Regression and gradient boosting inside each GroupKFold fold; models are not saved | No serialized model in repository |
| `src/cluster_profiles.py:93` | Saves KMeans cluster labels to `data/processed/profiles.csv`; centroids are not saved | No serialized clustering model |
| `.gitignore` | Excludes `*.zip`, `data/`, raw data and participant-level CSVs | The Step 6–8 artifact archives are not versioned here |

### 2.4 Planned / proposed approach (not implemented)

- **Proposed only:** an ONNX Runtime or TFLite format for on-device inference, as named in the concept note [3] (§8.1). No ONNX/TFLite export exists.
- **Proposed only:** storing the joblib file, `feature_columns.txt`, the portable JSON model, and their SHA-256 hashes together in the repository (or in a release attachment), so that this documentation can be reproduced from the repository alone.

## 3. Model Serving

### 3.1 Delivered: offline batch scoring

As documented in [2], the model is currently served **offline only**:

- the strict 30-feature Python pipeline wrapper with JSON/CSV command-line scoring;
- the portable pure-Python scorer;
- the Kotlin numeric class, compiled and tested in a JVM environment.

[2] states that the Kotlin implementation was "not deployed as an Android APK or used with real device events". The checks performed did not measure battery usage, traffic handling, operating-system background limits, behavioral effectiveness, or real-time on-device inference latency [2].

The repository pipeline is likewise run as batch scripts from the project root (repository evidence, `README.md`, "Çalıştırma sırası"):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/calibrate.py          # writes config/risk_thresholds.json
python src/forecast.py           # writes reports/forecast_*.csv
```

These scripts require the raw dataset under `mobile_phone_use/data/`. The raw dataset is not part of the repository.

### 3.2 Choice of deployment platform

| OPTION | DECISION | REASON |
|---|---|---|
| On-device (Android, edge inference) | **Chosen — Proposed only** | The concept note [3] (§8.5) requires that profiling, inference, and HITL feedback run on-device and that no raw logs leave the device. The portable 10-feature scorer [2] was built for this. |
| Cloud service | Not chosen | It would require sending usage logs off the device, which conflicts with [3] (§8.5) and with the repository design notes (`README.md`, "Uygulama tarafı: tasarım kararları": data does not leave the device). |
| On-premises server | Not chosen | Same privacy constraint. No server component is planned in [3]. |

### 3.3 Planned / proposed approach (not implemented)

- **Proposed only:** the Kotlin numeric core is called from the Android app with features computed by an on-device feature extractor (UsageStats [9]). Scoring runs once per day for the next-day proxy, since many of the selected features are daily aggregates or rolling 20-day historical summaries [2] (Appendix A).
- **Proposed only:** the prediction output is shown to the user and is **not** connected to automatic throttling (`automatic_intervention_authorized = False` [2]).

## 4. API Integration

### 4.1 Current status

**No HTTP API exists.** The repository has no Flask, FastAPI, or Streamlit code and no endpoint. [2] reports that a request/response contract and error-handling rules were documented in Step 8, but those files are not in this repository. Their exact field names are therefore not reproduced here.

The only delivered integration surfaces are the JSON/CSV command-line scorer and the Kotlin class [2]. In the planned on-device design, the app calls the Kotlin class directly, so no network API is required.

### 4.2 Proposed contract (not implemented)

The contract below is a **proposed contract** for a local or demo HTTP wrapper around the portable scorer. It is illustrative only, and the values are **not** real predictions. The input fields are the 10 features listed in [2] (Appendix A). The output fields follow the variable names in the scoring rule in [2] (Test Submission §6).

**Endpoint (proposed):** `POST /v1/predict-next-day-proxy`

**Request (proposed):**

```json
{
  "features": {
    "night_unlock_events_00_06": 4,
    "night_unlock_ratio": 0.12,
    "night_nonstudy_app_events": 9,
    "night_nonstudy_app_ratio": 0.08,
    "notification_posted_nonstudy": 140,
    "has_notification_events": 1,
    "prior20_median_night_unlock_ratio": 0.10,
    "prior20_unlock_delta": 6.0,
    "prior20_unlock_ratio": 1.15,
    "prior20_app_delta": -3.0
  }
}
```

A missing value (`null`) is replaced by the training median, following `value[i] or train_median[i]` in [2]. Note: in Python, `0 or train_median[i]` also returns the median, so that expression would treat a literal `0` as missing. The proposed contract treats only `null` as missing; a real `0` value is kept.

**Response (proposed):**

```json
{
  "proxy_probability": 0.61,
  "threshold": 0.55,
  "predicted_next_day_proxy": 1,
  "automatic_intervention_authorized": false,
  "label": "y_next_day_behavior_change_proxy"
}
```

**Error response (proposed):** HTTP `422`. Returned when a field is unknown, or when a value is non-numeric (lists, objects, strings, or booleans in place of numbers):

```json
{
  "error": "invalid_input",
  "detail": "night_unlock_ratio must be a number"
}
```

**Run command (proposed, no such module exists yet):**

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`proxy_probability` is an **uncalibrated model score** for the experimental proxy. It is not a clinical addiction probability, and it does not authorize an automatic network restriction [2].

## 5. Security Considerations

### 5.1 Delivered measures

| MEASURE | WHERE | STATUS |
|---|---|---|
| Model integrity: the frozen joblib pipeline is SHA-256 hash-verified | [2] Test Submission §5–6 | **Delivered** |
| Safety contract: `automatic_intervention_authorized = False` in the scorer output | [2] Test Submission §6 | **Delivered** |
| No participant or date identifier used as a predictor; next-day audit fields excluded | [2] Executive summary, Test Submission §2 | **Delivered** |
| Raw data and participant-level tables are kept out of version control | `.gitignore` | **Delivered** (repository evidence) |
| Raw file names are mapped to sequential participant IDs (`P000`, `P001`, …) | `src/common.py:26-29` | **Delivered** (repository evidence) |
| No secrets, API keys, or passwords in the code | repository search | **Delivered** (repository evidence) |

### 5.2 Planned / proposed approach (not implemented)

- **Proposed only — data minimisation and on-device processing.** Usage logs, inference, and HITL feedback stay on the device, aiming at GDPR/KVKK compliance [3] (§8.5). Network monitoring through the local `VpnService` [10] reads only the **domain name** (DNS/SNI), never page content (`README.md`, "Uygulama tarafı: tasarım kararları").
- **Proposed only — authorization.** Data access is granted through explicit Android permissions (UsageStats [9], VPN [10]) requested from the user [2] (planned architecture). Any network restriction requires a manual override option [3] (§2, Objective 4).
- **Proposed only — encryption.** Local storage (SQLite/Room, [3] §8.1) holds HITL feedback. Encrypting it at rest is proposed but not designed yet.
- **Proposed only — API authentication.** No remote API is planned. If the demo wrapper in Section 4.2 were ever exposed beyond `127.0.0.1`, it would require an API key read from an environment variable (for example `UNHOOK_API_KEY`) and HTTPS. No key is stored in this document or in the repository.
- **Proposed only — input validation.** Reject booleans, lists, and objects in numeric fields (Section 4.2) before scoring.

## 6. Monitoring and Logging

### 6.1 Delivered: offline evaluation and verification metrics

No runtime monitoring exists because the model is not deployed. The metrics that were tracked offline, as documented in [2]:

| METRIC | VALIDATION (659 rows) | HELD-OUT TEST (611 rows) |
|---|---|---|
| Macro F1 | 0.6314 | 0.6210 |
| Positive-class F1 | 0.4351 | 0.4089 |
| Precision (+) | 0.3583 | 0.3274 |
| Recall (+) | 0.5537 | 0.5446 |
| Average precision | 0.3890 | 0.3053 |
| ROC-AUC | 0.7277 | 0.6928 |
| Brier score | 0.2253 | 0.2300 |

On the test set, the confusion counts were TN 397, FP 113, FN 46, TP 55. The participant-cluster bootstrap (3,000 resamples) gave a 95% interval of 0.569–0.672 for Macro F1 [2]. The project target of Macro F1 > 0.85 was not met [2].

Deployment verification checks [2] (Test Submission §6):

| SOFTWARE CHECK | DOCUMENTED RESULT |
|---|---|
| Frozen model hash | Matched original Step 7 model |
| Portable Python parity | 659/659; maximum \|Δp\| = 2.22e-16 |
| Decision parity | All 659 validation proxy flags identical |
| Kotlin JVM tests | 12 / 12 synthetic test cases passed |
| Python unit tests | 5 / 5 passed |

### 6.2 Repository evidence

- `src/calibrate.py:98-100` prints and saves `reports/calibration.txt`. The file reports the **warning burden** of the rule-based risk score: 47.8% of participants never receive a warning, 7.3% have at least one "Kritik" day. It also reports parameter sensitivity (mean Spearman 0.997) and component ablation.
- `src/forecast.py:182-183, 189-202` prints progress and saves `reports/forecast_metrics.csv`, `reports/forecast_importance.csv`, `reports/fig_forecast_pr.png`, and `reports/fig_forecast_importance.png`.
- Logging is done with `print` only. The Python `logging` module is not used, and there is no alerting.

### 6.3 Planned / proposed approach (not implemented)

| WHAT TO TRACK | WHY | ALERT (proposed) |
|---|---|---|
| Share of days flagged per user | False-alarm burden is a stated limitation [2]; it can be compared with the offline warning burden in `reports/calibration.txt` | Flag rate far above the offline false-positive rate observed on the held-out test (FP 113, TN 397 [2]) |
| User reactions: dismissal, app exit, manual override [3] (§4) | Measures whether warnings are accepted; an input for threshold review | Override rate rising week over week |
| Missing-feature rate (values replaced by training median) | The feature extractor mapping to live Android events is incomplete [2] | Any feature missing on most days |
| Input distribution drift vs. training medians and scaler means | The training data are old observational logs [2] | Large shift in a top feature |
| Score-version / model-hash in every log line | Ties each prediction to the hash-verified artifact | Hash mismatch at app start |

- **Proposed only:** logs stay on the device (Python `logging` for the batch tools; local Room/SQLite tables for the Android app). They contain no raw usage events, only aggregate counters, in line with [3] (§8.5).
- **Proposed only:** before any real-world use, a small participant-consented pilot with time-ordered evaluation and calibration checks, as recommended in [2] (Conclusion).

---

## References

[1] Course instructor (2026). Deployment template (instructor-provided structure): Overview; Model Serialization; Model Serving; API Integration; Security Considerations; Monitoring and Logging.

[2] UNHOOK Project Team (2026). `UNHOOK_ADIM9_Final_Model_Refinement_and_Test_Submission.docx`: Model Refinement and Test Submission, Final Technical Report, Step 9 of 9 (7 November 2026).

[3] UNHOOK Project Team (2026). `UNHOOK_ÖDEV2.pdf`: Capstone Project Concept Note and Implementation Plan (18 October 2026).

[4] UNHOOK repository. `README.md` (Kurulum; Çalıştırma sırası; Uygulama tarafı: tasarım kararları; Durum).

[5] UNHOOK repository. `src/calibrate.py`, `src/forecast.py`, `src/risk.py`, `src/common.py`, `src/cluster_profiles.py`.

[6] UNHOOK repository. `config/risk_thresholds.json`.

[7] UNHOOK repository. `reports/calibration.txt`, `reports/forecast_metrics.csv`, `reports/forecast_importance.csv`.

[8] scikit-learn developers. scikit-learn documentation: Pipeline, model selection, and model persistence. https://scikit-learn.org/stable/

[9] Android Developers. UsageStatsManager API. https://developer.android.com/reference/android/app/usage/UsageStatsManager

[10] Android Developers. VPN / per-app VPN and VpnService.Builder. https://developer.android.com/develop/connectivity/vpn
