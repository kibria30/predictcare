# PredictCare Knowledge Base

## Overview
PredictCare is a Streamlit application for outpatient no-show risk prediction. It trains a `DecisionTreeClassifier` on historical appointment data and then scores upcoming appointments into three operational risk bands: High Risk, Medium Risk, and Low Risk.

The app is organized as a four-step workflow in the sidebar:
1. Home and data format overview.
2. Upload and prepare historical data.
3. Train the model and review results.
4. Score upcoming appointments.

## Core Purpose
The application is designed to help clinic managers identify patients likely to miss appointments so that staff can intervene before the visit date. The implementation emphasizes interpretability over model complexity: the model is a decision tree, the feature set is small, and the UI exposes metrics, rules, and feature importance.

## Runtime Stack
- Streamlit for the UI and navigation.
- Pandas and NumPy for data handling.
- scikit-learn for preprocessing, train/test splitting, metrics, and the decision tree model.
- Matplotlib and Plotly for visualizations.
- Base64 for CSV download links.

## Session State
The app persists training artifacts in `st.session_state` so the user can move between pages without retraining immediately.

Stored keys:
- `model`: the trained decision tree.
- `label_encoders`: fitted label encoders for categorical training columns.
- `feature_cols`: declared but not actively used in the current code path.
- `X_train`, `X_test`, `y_train`, `y_test`: train/validation split.
- `trained`: boolean gate for later pages.
- `clean_log`: log of data preparation actions.
- `train_df`: last dataset used in training.

## Data Contract
### Required training columns
Historical data must contain these columns, case-insensitive:
- `patient_age_group`
- `previous_noshows`
- `lead_time_days`
- `appointment_type`
- `reminder_sent`
- `attended`

### Expected values
- `patient_age_group`: `Under 18`, `18-30`, `31-50`, `51-65`, `Over 65`
- `previous_noshows`: integer, expected range 0 to 20
- `lead_time_days`: integer, expected range 0 to 365
- `appointment_type`: `New`, `Follow-up`
- `reminder_sent`: `Yes`, `No`
- `attended`: binary target, `1` for attended and `0` for no-show

### Upcoming appointment columns
Upcoming scoring data should contain:
- `patient_ref`
- `patient_age_group`
- `previous_noshows`
- `lead_time_days`
- `appointment_type`
- `reminder_sent`

The upcoming file does not need `attended`.

## Data Preparation Pipeline
The `prepare_data()` helper performs the following:
1. Normalizes column names to lowercase snake case.
2. Verifies the required historical columns are present.
3. Removes duplicate rows.
4. Fills missing values with median for numeric columns or mode for categorical columns.
5. Filters invalid ranges for `lead_time_days` and `previous_noshows`.
6. Filters invalid target values outside `0` and `1`.
7. Label-encodes `patient_age_group`, `appointment_type`, and `reminder_sent`.
8. Returns `X`, `y`, the encoders, and a human-readable preparation log.

## Upcoming Data Encoding
The `encode_upcoming()` helper applies the saved label encoders to upcoming appointment data.

Important behavior:
- Unknown categorical values are replaced with the first known label before encoding.
- `previous_noshows` and `lead_time_days` are coerced to integers, with defaults of `0` and `14` if parsing fails.

## Model Logic
The model is a binary decision tree trained on the prepared feature matrix.

Training settings exposed in the UI:
- Maximum tree depth: slider from 2 to 8, default 4.
- Minimum samples per leaf: slider from 5 to 30, default 10.

Training details:
- The classifier uses `class_weight="balanced"`.
- The model is trained on a 80/20 train-test split with `random_state=42` and `stratify=y`.
- Validation metrics shown after training include accuracy, sensitivity, specificity, and validation set size.

## Risk Bucketing
The app does not train a true 3-class classifier. Instead, it uses the decision tree’s `predict_proba()` output for the attended class and maps probability to risk bands:
- `p < 0.50` -> High Risk
- `0.50 <= p < 0.75` -> Medium Risk
- `p >= 0.75` -> Low Risk

This means the risk label is derived after binary classification, not predicted as a native class.

## Feature Naming
The app maps the model’s feature columns to display names through `FACTOR_NAMES`:
- `patient_age_group` -> Age Group
- `previous_noshows` -> Previous No-Shows
- `lead_time_days` -> Lead Time (days)
- `appointment_type` -> Appointment Type
- `reminder_sent` -> Reminder Sent

The top factor displayed in the upcoming-schedule page is the single highest global feature importance from the trained model, not a row-level explanation.

## UI Flow
### Home
The home page provides:
- a short project description,
- key metrics cards,
- a four-step workflow explanation,
- the required data format,
- download buttons for sample training and upcoming CSV files.

### Step 1: Upload & Prepare Data
This page allows:
- loading built-in sample data,
- uploading a CSV or Excel file,
- previewing raw data,
- running the preparation pipeline,
- splitting into train and validation sets.

If the file is missing `attended`, preparation fails because the app expects a historical training dataset.

### Step 2: Train Model
This page:
- checks that Step 1 was completed,
- lets the user tune tree depth and minimum leaf size,
- trains the decision tree,
- displays accuracy, sensitivity, specificity, and validation record count.

### Step 3: View Results
This page contains four tabs:
- Decision Tree: tree diagram and rule text.
- Performance: confusion matrix and summary metrics.
- Risk Distribution: risk counts for the training data, shown as pie and bar charts.
- Feature Importance: bar chart of feature importance and the most influential predictor.

### Step 4: Score Upcoming
This page:
- loads sample or uploaded upcoming appointments,
- encodes the data using the fitted label encoders,
- predicts attendance probability,
- converts the probability into risk categories,
- adds recommended actions,
- displays a sorted, styled results table,
- provides a CSV download of the risk report.

## Visual Design Notes
The interface uses custom CSS to create:
- a branded top banner,
- gradient sidebar styling,
- card-based metric tiles,
- color-coded risk badges and table rows,
- helper info boxes and download buttons.

## Known Implementation Notes
- Several imports are unused in the current file, including `mpatches`, `io`, `classification_report`, `go`, and `UPCOMING_COLS`.
- `RISK_LABELS` is declared but not used directly in the UI flow.
- The comments in Step 2 mention a 3-class target, but the implementation trains a binary classifier and derives risk bands afterward.
- The risk summary on the upcoming page is based on the global feature importance of the trained model, so it is not an appointment-specific explanation.
- The app silences warnings globally with `warnings.filterwarnings("ignore")`, which can hide useful diagnostics during development.

## Output Artifacts
The app can produce:
- trained model state in memory,
- decision tree plot,
- confusion matrix,
- risk distribution charts,
- feature importance chart,
- downloadable CSV risk report for upcoming appointments.

## Practical Extension Ideas
- Persist the trained model to disk so users do not need to retrain each session.
- Add robust schema validation for uploaded files before preparation.
- Replace label encoding with a more explicit mapping layer for clearer handling of unseen values.
- Add row-level explanation logic for upcoming appointment scoring.
- Expose the cleaned training data and validation predictions as downloads for auditability.