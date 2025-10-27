# OLID Project — Offensive Language Identification (Training + API)

This repository contains an end‑to‑end workflow for detecting offensive/foul language in short texts (e.g., tweets) using classic NLP with scikit‑learn. It includes:
- A training script that builds a TF‑IDF + linear classifier pipeline and saves artifacts
- A FastAPI service that serves single and batch predictions using the trained model
- A minimal static UI for batch predictions
- Notebooks used during exploration

## Live Deployment
>
- The API and web interface are fully deployed on **Google Cloud Run**.  
- **Base API:** [https://olid-api-1078757655479.europe-west1.run.app](https://olid-api-1078757655479.europe-west1.run.app)
- **Interactive Web UI:** [https://olid-api-1078757655479.europe-west1.run.app/ui/](https://olid-api-1078757655479.europe-west1.run.app/ui/)
- The Cloud Run instance automatically loads the trained model (`artifacts/model.pkl`) at startup and serves both the REST API and the UI.  
- Local runs are still supported via `uvicorn` or Docker.


## Tech stack
- Language: Python (3.11 in Docker; 3.9+ should work locally)
- ML/NLP: scikit‑learn, numpy, pandas, nltk, matplotlib (for evaluation plots)
- Web API: FastAPI, Pydantic
- App server: uvicorn (ASGI)
- Packaging: pip via `requirements.txt` (under `OLID_Project/`)
- Containerization: Docker (Python 3.11 slim base)


## Repository layout
```
/Users/muratkorkmaz/Desktop/YL/NLP/Assignment 1
├── OLID_Project/
│   ├── app/
│   │   └── app.py
│   ├── artifacts/
│   │   ├── confusion_matrix.png
│   │   ├── metrics.json
│   │   └── model.pkl
│   ├── data/
│   │   ├── labeled_data.csv
│   │   └── olid.csv
│   ├── static/
│   │   └── index.html
│   └── train_olid_V1.py
│   ├── Dockerfile
│   ├── README.md
│   ├── OLID_Training_Pipeline.ipynb
│   ├── requirements.txt


```


## Data
The training script expects a CSV at `OLID_Project/data/labeled_data.csv` (default) with columns:
- `text` — input text
- `label` — integer class label (0 = proper, 1 = foul)

`OLID_Project/data/olid.csv` also exists and can be adapted if needed. Ensure the file you intend to use matches the expected columns.


## Training
Run from inside the `OLID_Project/` directory so relative paths resolve correctly:

```bash
cd OLID_Project
python train_olid_V1.py
```

What the script does:
- Loads `data/labeled_data.csv`
- Splits data into train/validation (stratified, `random_state=42`)
- Preprocesses text using a custom `clean_tweet` function (lowercasing, URL/mention/emoji removal, basic normalization, and stopword removal using NLTK with an extended list)
- Builds a scikit‑learn pipeline (Char/word TF‑IDF with a linear classifier; see code for exact configuration)
- Trains the model and computes metrics over a probability threshold grid
- Saves artifacts to `artifacts/`:
  - `model.pkl` — a dict like `{ "pipeline": sklearn.Pipeline, "threshold": float }`
  - `metrics.json` — metrics summary
  - `confusion_matrix.png` — validation confusion matrix

Notes:
- On first run, NLTK stopwords may be downloaded automatically if missing.
- The selected threshold stored in the artifact is used by the API for decisioning.


## Serving the model (FastAPI)
The service can be run locally or directly accessed via its deployed instance on **Google Cloud Run**:  
- **Base API:** [https://olid-api-1078757655479.europe-west1.run.app](https://olid-api-1078757655479.europe-west1.run.app)  
- **Interactive Web UI:** [https://olid-api-1078757655479.europe-west1.run.app/ui/](https://olid-api-1078757655479.europe-west1.run.app/ui/)

Make sure `OLID_Project/artifacts/model.pkl` exists if running locally (created by the training step).
Then run the API from the `OLID_Project/` directory.


### Local run (pip)
```bash
cd OLID_Project
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger docs: http://localhost:8000/docs
- Health/version: `GET /` returns `{ ok, model_loaded, threshold, version, preprocessor }`
- Single prediction: `POST /predict` with body `{ "text": "..." }`
- Batch prediction: `POST /predict/batch` with body `{ "texts": ["...", "..."] }` (1–100 items)
- Admin: `POST /admin/reload` attempts to reload `artifacts/model.pkl`

Example curl:
```bash
curl -s http://localhost:8000/ -H 'accept: application/json'

curl -s http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"I love you but this is terrible"}'

curl -s http://localhost:8000/predict/batch \
  -H 'content-type: application/json' \
  -d '{"texts":["nice day","you are stupid"]}'
```

### Docker
A Dockerfile is provided in `OLID_Project/`.

Build and run (from the `OLID_Project/` directory):
```bash
cd OLID_Project
# Ensure artifacts/ contains model.pkl before building, or mount later
docker build -t olid-api:latest .
# Map local artifacts and choose a port (container listens on 8080)
docker run --rm -p 8080:8080 olid-api:latest
# If you need to inject artifacts at runtime instead of baking:
# docker run --rm -p 8080:8080 -v "$PWD/artifacts":/app/artifacts:ro olid-api:latest
```
- The image sets `ENV PORT=8080` and starts uvicorn with `--port 8080`.
- API docs (container): http://localhost:8080/docs

### Static UI
`OLID_Project/static/index.html` is an optional web UI. It currently points to a deployment URL in the code:
```js
const API_URL = "https://olid-api-1078757655479.europe-west1.run.app";
// const API_URL = ""; // Use "" for local testing (window.location.origin)
```
For local testing, edit the file and set `API_URL = ""` so it calls your local API origin.

Note: The UI references a `/model/info` endpoint that is not implemented in the current FastAPI app; the rest of the functionality (batch predictions) works with `/predict/batch`.


## Scripts and entry points
- Training: `python OLID_Project/train_olid_V1.py` (run with CWD = `OLID_Project/`)
- API: `uvicorn app.app:app` (CWD = `OLID_Project/`)
- Docker: `docker build -t olid-api . && docker run -p 8080:8080 olid-api`

There are no additional package scripts or a task runner; commands are run directly.


## Requirements
Install with pip using the provided requirements file:
```bash
cd OLID_Project
pip install -r requirements.txt
```
Contents of `requirements.txt`:
```
fastapi
uvicorn
scikit-learn
pandas
numpy
pydantic
nltk
```


## Environment variables
- Not required for local runs.
- Container: `PORT=8080` is set in the Docker image and used by uvicorn. Most PaaS platforms read `PORT` automatically.
- CORS: currently open to `*` in the API code; consider restricting in production.

Paths (relative to `OLID_Project/`):
- Data: `data/labeled_data.csv`
- Artifacts: `artifacts/`


## Tests

Three integration tests confirm that the API behaves as expected both locally and on Google Cloud Run:

| Test | Input | Expected Result |
|------|--------|-----------------|
| **1. Happy Path** | `"You are amazing!"` | Returns `prediction = 0` (proper) with low foul probability. (0.022) |
| **2. Foul Example** | `"You're such an idiot"` | Returns `prediction = 1` (foul) with high foul probability. (0.639) |
| **3. Invalid Input** | `{"text": ""}` | Returns **HTTP 422** with validation message: `"Please enter at least one line."`. |

All tests passed successfully on the deployed service:  
**Web UI:** [https://olid-api-1078757655479.europe-west1.run.app/ui/](https://olid-api-1078757655479.europe-west1.run.app/ui/)  
**Base API:** [https://olid-api-1078757655479.europe-west1.run.app](https://olid-api-1078757655479.europe-west1.run.app)

## Example test calls
**Happy Path**

```bash
curl -s https://olid-api-1078757655479.europe-west1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are amazing!"}'
```

**Foul Example**

```bash
curl -s https://olid-api-1078757655479.europe-west1.run.app/predict \
 -H "Content-Type: application/json" \
-d '{"text": "You are such an idiot"}'
```

**Invalid Input**

```bash
curl -s https://olid-api-1078757655479.europe-west1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'
```


## License
This project was developed as part of the ARI5501 Natural Language Processing course at Bahçeşehir University.
It is intended for educational and academic use only.

--------------------------

**Contact:** murat.korkmaz1@bahcesehir.edu.tr, suleyman.saritas@bahcesehir.edu.tr
