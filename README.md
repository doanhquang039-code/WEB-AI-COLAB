# HAR CSV Web

This is a standalone web app for uploading CSV files, reviewing data, calculating
column statistics, and predicting activities with the saved UCI HAR MLP model. It
does not depend on the `Learn-AI` GitHub repository at runtime.

## Input format

The web accepts CSV files up to 4 MB, 10,000 rows, and 1,000 columns. A header row
is optional. It shows a data preview, dataset-level findings (missing values,
non-numeric cells, duplicate rows, and constant columns), and per-column statistics
(type, unique values, min/max, quartiles, median, mean, standard deviation, and
IQR-based outlier counts or common text values).

Only a CSV with exactly 561 numeric UCI HAR features in the original feature order
is sent to the saved MLP for activity predictions. Other files are still analyzed
and previewed, but are not given HAR predictions. Random or unrelated values can
produce model predictions when they have 561 columns, but those predictions are
not meaningful HAR results. Raw accelerometer/gyroscope readings must first be
converted to the 561 UCI HAR features.

To evaluate the models, include those 561 feature columns and one `activity` column
with the actual activity name (for example `WALKING`) or UCI class ID (1–6). The app
reports accuracy, Macro-F1, precision/recall/F1 per class, and confusion matrices
for both MLP and Logistic Regression. Without true labels it compares predictions
and class distributions only; it does not report accuracy.

The “Tải CSV HAR mẫu có nhãn thật” link downloads a balanced 180-row sample selected
from the UCI HAR test split (30 rows per activity), with features and ground-truth
labels. It is for demonstrating evaluation, not a replacement for the complete
held-out test set. Dataset citation and license: Anguita et al., *A Public Domain
Dataset for Human Activity Recognition using Smartphones*, ESANN 2013; UCI HAR,
DOI 10.24432/C54S4K, CC BY 4.0.

## Run locally

From this folder:

```bash
python -m pip install -r requirements.txt
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The API documentation is at
`http://127.0.0.1:8000/docs`.

Run the tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The saved model was created with scikit-learn 1.6.1; the requirements pin that
version so predictions are consistent with the Colab model.

## Deploy to Vercel

The Vercel Python entrypoint is explicitly set to `web_app:app` in `pyproject.toml`,
so Vercel does not mistake the test module for the production FastAPI app.

1. Push this project to the `WEB-AI-COLAB` GitHub repository.
2. In Vercel, import that repository and set the project Root Directory to `.` (the
   repository root containing `pyproject.toml`, `web_app.py`, `index.html`, and
   `outputs/mlp_har.joblib`).
3. Use the Python/FastAPI framework preset or let Vercel detect it, then deploy.

To stay within Vercel's request-body limit, uploads are capped at 4 MB. The generated
deployment URL is available in the Vercel project dashboard after a successful deployment.

## Run in Google Colab

Upload this folder as a ZIP to the Colab session, then extract it:

```python
from google.colab import files
uploaded = files.upload()  # Select the ZIP made from this folder

import zipfile
with zipfile.ZipFile(next(iter(uploaded))) as archive:
    archive.extractall("/content/har-web")
```

Install dependencies and start the web app:

```python
%cd /content/har-web
!pip install -r requirements.txt
```

```python
import subprocess
import time
import urllib.error
import urllib.request
from google.colab.output import eval_js

server = subprocess.Popen([
    "python", "-m", "uvicorn", "web_app:app",
    "--host", "0.0.0.0", "--port", "8000"
])
for _ in range(60):
    if server.poll() is not None:
        raise RuntimeError("The web API stopped during startup.")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1):
            pass
        break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(1)
else:
    raise RuntimeError("The web API did not start within 60 seconds.")

print(eval_js("google.colab.kernel.proxyPort(8000)"))
```

Open the printed URL while the Colab runtime is active. This is suitable for a demo,
not for a continuously available public service.
