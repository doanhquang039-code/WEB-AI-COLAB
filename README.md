# HAR CSV Web

This is a standalone web app for uploading CSV files, reviewing data, calculating
column statistics, and predicting activities with the saved UCI HAR MLP model. It
does not depend on the `Learn-AI` GitHub repository at runtime.

## Input format

The web accepts CSV files with up to 10,000 rows and 1,000 columns. A header row is
optional. It shows a data preview and per-column statistics (numeric type, missing
values, unique values, min/max/mean or common text values).

Only a CSV with exactly 561 numeric UCI HAR features in the original feature order
is sent to the saved MLP for activity predictions. Other files are still analyzed
and previewed, but are not given HAR predictions. Random or unrelated values can
produce model predictions when they have 561 columns, but those predictions are
not meaningful HAR results. Raw accelerometer/gyroscope readings must first be
converted to the 561 UCI HAR features.

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
python -m unittest discover -s tests -v
```

The saved model was created with scikit-learn 1.6.1; the requirements pin that
version so predictions are consistent with the Colab model.

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
