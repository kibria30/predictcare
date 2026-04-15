# 🏥 PredictCare — Patient No-Show Risk Predictor

> **BU7081 – Programming for Business Analytics | Chester Business School**

PredictCare is a Streamlit-based data analytics platform that helps outpatient clinic managers predict which patients are at risk of missing their scheduled appointments, using a Decision Tree classification model.

---

## 🚀 Live Demo

Deploy instantly on **Streamlit Cloud** (see deployment instructions below).

---

## 📋 Features

| Feature | Description |
|---|---|
| 📂 Data Upload | Upload historical appointment CSV/Excel or use built-in sample data |
| 🔧 Auto Cleaning | Handles missing values, duplicates, and out-of-range entries automatically |
| 🤖 Model Training | Trains a Decision Tree classifier with configurable depth |
| 🌳 Tree Visualisation | Visual flowchart of the full decision tree |
| 📈 Performance Metrics | Accuracy, sensitivity, specificity, confusion matrix |
| 📊 Risk Distribution | Interactive charts showing High / Medium / Low risk proportions |
| 🔑 Feature Importance | Which variables drive no-show risk in your patient population |
| 🔮 Appointment Scoring | Upload upcoming schedule → get risk category + recommended action per slot |
| ⬇ Download | Export full risk report as CSV |

---

## 📁 Required Data Format

### Historical Data (Training)
| Column | Type | Values |
|---|---|---|
| `patient_age_group` | Categorical | Under 18 / 18-30 / 31-50 / 51-65 / Over 65 |
| `previous_noshows` | Integer | 0–20 |
| `lead_time_days` | Integer | Days from booking to appointment |
| `appointment_type` | Categorical | New / Follow-up |
| `reminder_sent` | Categorical | Yes / No |
| `attended` | Binary | 1 (attended) or 0 (no-show) |

### Upcoming Appointments (Scoring)
Same columns as above **except** `attended`, plus a `patient_ref` column.

---

## 🛠 Local Installation

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/predictcare.git
cd predictcare

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. **Push this folder to a GitHub repository** (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"**
5. Select your repository, branch (`main`), and set **Main file path** to `app.py`
6. Click **"Deploy"** — your app will be live in ~2 minutes

Your live URL will be: `https://YOUR_APP_NAME.streamlit.app`

---

## 📦 Project Structure

```
predictcare/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🎓 Academic Context

This prototype was developed as part of the BU7081 Portfolio Assessment at Chester Business School. The platform addresses the real-world problem of patient no-shows in outpatient clinics, using a Decision Tree classification approach selected after critical evaluation of alternative analytical methods including association rule mining, clustering, and Monte Carlo simulation.

**Module:** BU7081 – Programming for Business Analytics  
**Module Leader:** Kelvin Leong  
**Institution:** Chester Business School, University of Chester
