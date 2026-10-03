# Smart Queue Management System

A Flask + SQLite queue management system with priority scheduling, round-robin
load balancing, concurrency-safe ticket assignment, retry/dead-letter handling
for no-shows, rate limiting, and wait-time estimation — with an optional
Temporal Fusion Transformer (TFT) model for quantile-based forecasts (P10/P50/P90).

## Project Structure
```
smart_queue/
├── app.py
├── requirements.txt
├── Procfile
├── .gitignore
├── README.md
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── ticket.html
│   ├── admin.html
│   └── error.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
└── ml/
    ├── requirements-ml.txt
    ├── generate_synthetic_data.py
    ├── train_tft.py
    ├── predict.py
    └── saved_model/   (trained model checkpoint goes here)
```

---

## A. Running the Normal Flask Application on Windows

1. **Install Python** (3.10 or later) from python.org if you don't have it.
   During installation, check "Add Python to PATH".

2. **Open Command Prompt** and navigate to the project folder:
   ```
   cd path\to\smart_queue
   ```

3. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

4. **Activate the virtual environment:**
   ```
   venv\Scripts\activate
   ```
   You should see `(venv)` appear at the start of your command prompt line.

5. **Install the requirements:**
   ```
   pip install -r requirements.txt
   ```
   This installs only Flask and gunicorn — no heavy ML libraries. The app
   runs fully functional with the baseline wait-time estimator at this point.

6. **Start the Flask server:**
   ```
   python app.py
   ```
   You should see output like `Running on http://127.0.0.1:5000`, plus a
   console message confirming the baseline estimator is in use (see Section
   D below on what the TFT vs baseline log messages look like).

7. **Open the application in your browser:**
   - Customer view: http://127.0.0.1:5000
   - Admin view: http://127.0.0.1:5000/admin

   The SQLite database (`queue.db`) and 2 sample counters ("Counter 1",
   "Counter 2") are created automatically the first time the app runs.

8. **To stop the server:** press `CTRL+C` in the terminal.

9. **To deactivate the virtual environment** when done: `deactivate`

---

## B. Installing the Optional ML Dependencies

The TFT model requires PyTorch and pytorch-forecasting, which are **not**
part of `requirements.txt` (they are too large for free-tier deployment).
Install them separately, only on a machine where you intend to train or
test the TFT model:

```
cd ml
pip install -r requirements-ml.txt
```

This installs: `torch`, `pytorch-lightning`, `pytorch-forecasting`, `pandas`,
`numpy`, `matplotlib`.

---

## C. Generating Synthetic Training Data

A brand-new queue system has no real history to learn from, so a synthetic
dataset is generated first to simulate several weeks of realistic queue
activity (with morning/evening peaks and weekday/weekend variation), broken
down per counter **and per service type** (since `service_type` is a static
covariate the model needs to learn separate patterns for):

```
cd ml
python generate_synthetic_data.py
```

This produces `ml/queue_history.csv`. Once your deployed app has
accumulated a few real weeks of ticket data, you can instead run:
```
python -c "from generate_synthetic_data import export_real_data; export_real_data()"
```
to export real historical data in the same format, for retraining on
actual usage patterns.

---

## D. Training the TFT Model

```
cd ml
python train_tft.py
```

This builds a `TimeSeriesDataSet` grouped by `(counter_id, service_type)`,
trains a `TemporalFusionTransformer` with `QuantileLoss` (so it outputs
P10/P50/P90-style predictions instead of one number), and saves:

- `ml/saved_model/tft_model.ckpt` — the trained model checkpoint
- `ml/training_forecast_sample.png` — a plot of actual vs predicted queue
  length with the quantile band, useful as evidence of the model working

Training runs for up to 20 epochs with early stopping, and should take a
few minutes on a normal laptop CPU (no GPU required for this dataset size).

---

## E. Testing TFT Predictions

Once trained, you can sanity-check the prediction interface directly:

```
cd ml
python -c "
from predict import predict_wait_tft, minutes_from_queue_length
import pandas as pd
df = pd.read_csv('queue_history.csv')
sample = df[(df.counter_id=='1') & (df.service_type=='General Service')].tail(96)
result = predict_wait_tft(sample)
print('P10:', result['p10'][:3])
print('P50:', result['p50'][:3])
print('P90:', result['p90'][:3])
print('Example wait (minutes):', minutes_from_queue_length(result['p50'][0], 180, 2))
"
```

If this prints quantile lists without errors, the model and interface are
working correctly.

---

## F. Running the Flask App With the Trained TFT Model

Once `ml/saved_model/tft_model.ckpt` exists and the ML dependencies are
installed in the **same** Python environment the Flask app runs in:

```
venv\Scripts\activate
pip install -r ml\requirements-ml.txt
python app.py
```

Now, when a ticket's wait time is requested, the console will show:
```
[WAIT-ESTIMATE] Using TFT prediction: {'p10': 8, 'p50': 15, 'p90': 24}
```
instead of the baseline message. The customer's Ticket Status page will
display a range (e.g. "Estimated wait: 8-24 minutes, Most likely: 15
minutes") instead of a single fixed number.

**Note:** the app needs at least 96 real 15-minute time buckets of ticket
history (24 hours' worth) before it has enough data to call the TFT model
live — until then, it automatically uses the baseline estimator, with a
log message explaining why ("Not enough real ticket history yet").

### How the TFT output is converted into a wait time

The TFT model forecasts **queue length**, not wait time directly. The
conversion (implemented in `ml/predict.py`'s `minutes_from_queue_length()`
and mirrored by the baseline formula in `app.py`) is:

```
estimated_wait_seconds = (predicted_queue_length * avg_service_seconds) / active_counters
estimated_wait_minutes = estimated_wait_seconds // 60
```

This is applied separately to the P10, P50, and P90 predicted queue-length
values, producing the optimistic / most-likely / conservative wait times
shown to the customer.

---

## G. Deploying the Main Application to Render.com (without ML dependencies)

1. Create a free account at https://render.com (sign up with GitHub).

2. Push the project folder to a GitHub repository — all files including
   `templates/`, `static/`, and the `ml/` folder's **code** files are fine
   to include, but do NOT commit `queue.db`, `venv/`, or any trained model
   checkpoint `.ckpt` files (already excluded via `.gitignore`). Render's
   free tier will only install `requirements.txt` (Flask + gunicorn), so
   the ML code sits there as source but its heavy dependencies are never
   installed — the app will correctly fall back to the baseline estimator
   on Render, exactly as designed.

3. On the Render dashboard, click **New +** → **Web Service**.

4. Connect your GitHub account and select your repository.

5. Configure the service:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free

6. Click **Create Web Service**. Render installs dependencies and starts
   the app — this takes 2–5 minutes on the first deploy.

7. Once deployed, Render gives you a live URL like
   `https://smart-queue.onrender.com`.

   Note: On Render's free tier, the SQLite database resets whenever the
   service restarts or spins down from inactivity. This is fine for a
   demo, but not for permanent production data.

---

## Testing Checklist

**1. Normal and priority tickets**
- [ ] Join the queue as a normal customer (leave "Priority" unchecked).
- [ ] Join again as a priority/VIP customer (check the box).
- [ ] On the Admin page, confirm the VIP ticket appears above the normal
      ticket in the "Waiting Queue" table, even though it joined later.

**2. FIFO ordering within the same priority**
- [ ] Join 3 normal customers one after another.
- [ ] Confirm they appear in the Waiting Queue in the exact order they
      joined (first joined = first in list).

**3. Multiple counters and round-robin distribution**
- [ ] On the Admin page, add a 3rd counter using the "Add Counter" form.
- [ ] Click "Call Next Ticket" multiple times and confirm customers are
      distributed across different counters, not all sent to the same one.
- [ ] With counters at equal load, confirm the tie-break alternates
      (check the Counter column across several consecutive "Call Next" calls).

**4. Concurrent "Call Next" requests**
- [ ] With at least 2 people waiting, open the Admin page in two browser
      tabs and click "Call Next Ticket" in both at nearly the same time.
- [ ] Confirm two different tickets get served (no duplicate assignment).

**5. No-show retries and dead-letter handling**
- [ ] Call a ticket to a counter, then click "No-show".
- [ ] Confirm the ticket returns to the Waiting Queue with "Retries: 1".
- [ ] Repeat two more times and confirm it disappears from the Waiting
      Queue (moved to dead-letter) after exceeding the retry limit.
- [ ] Check the "Dead-lettered" stat card increased by 1.

**6. Rate limiting**
- [ ] Submit the Join form 6 times quickly (within 60 seconds) from the
      same browser/IP.
- [ ] Confirm the 6th attempt shows a "Too many requests" error message.

**7. Ticket status polling**
- [ ] Join the queue and stay on your Ticket Status page.
- [ ] From another tab, call that ticket to a counter via Admin.
- [ ] Confirm your Ticket Status page automatically updates to "It's your
      turn!" within about 5 seconds, without manually refreshing.

**8. Wait-time estimation (baseline and TFT)**
- [ ] Without the TFT model present: confirm the Ticket Status page shows
      a single "~N min" estimate, and the console logs the baseline message.
- [ ] With the TFT model trained and present (and ML deps installed):
      confirm the page shows a P10-P90 range and "Most likely: Npp min",
      and the console logs the TFT message.
- [ ] Temporarily rename `ml/saved_model/tft_model.ckpt` and confirm the
      app falls back to the baseline estimator without crashing.

**9. Service type**
- [ ] Join with each of the 4 service type options and confirm each is
      correctly stored and displayed on the Admin dashboard and the
      customer's own Ticket Status page.

**10. Mobile responsiveness**
- [ ] Open the app on a phone (or use your browser's responsive mode).
- [ ] Confirm the Join page, Ticket Status page, and Admin dashboard all
      remain readable and usable without horizontal scrolling.

---

## Notes on Authentication

Admin authentication is intentionally not implemented in this prototype.
Every admin route is wrapped with an `admin_required` decorator in
`app.py` that currently does nothing — real authentication (session-based
login, an API key, etc.) can be added there later without restructuring
the routes.
