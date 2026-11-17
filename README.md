# Typing Fingerprint

Typing Fingerprint is a local development project for additional behavioral verification using typing dynamics. Phase 1 established a React/Vite frontend, FastAPI backend, and PostgreSQL connection foundation. Phase 2 adds an in-browser fixed-passage typing interface and ephemeral timing capture. The sample stays in browser memory and is not sent to the API or persisted. No identity prediction is made.

## Architecture

- `frontend/`: React and Vite UI; `/api` requests proxy to FastAPI in development.
- `frontend/src/hooks/`: browser-only keystroke timing collector.
- `frontend/src/components/`: typing practice interface and timing display.
- `backend/`: FastAPI routes, environment settings, SQLAlchemy engine/session.
- PostgreSQL: local service through Docker Compose.
- `ml/`: session-safe feature extraction, preprocessing, and JSONL dataset contract.
- `models/`: reserved for model checkpoints in later phases.

## Windows setup

Requirements: Git, Node.js 20+, Python 3.11+, and Docker Desktop with Compose.

1. Copy `.env.example` to `.env`. Generate `JWT_SECRET_KEY` with `python -c "import secrets; print(secrets.token_urlsafe(48))"` and `BIOMETRIC_ENCRYPTION_KEY` with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`, then put both values in `.env`. Never commit `.env`.
2. Start PostgreSQL: `docker compose up -d db`.
3. In a PowerShell window: `py -3.11 -m venv .venv`, then `.\.venv\Scripts\Activate.ps1` and `python -m pip install -r backend/requirements.txt -r ml/requirements.txt`.
4. Apply database migrations from the repository root: `alembic -c alembic.ini upgrade head`.
5. Start the API from the repository root: `uvicorn app.main:app --app-dir backend --reload --port 8000`.
6. In another window: `cd frontend`, `npm install`, then `npm run dev`.
7. Open `http://localhost:5173`. API docs are at `http://localhost:8000/docs`; health is at `http://localhost:8000/api/health`.

To try Phase 2, enable consent, select **Start capture**, and type the displayed passage into the field. Select **Finish sample** to validate it and review local timing bars. Reset clears the sample and input. Paste is disabled so the passage is typed on the keyboard.

## Phase 3 data pipeline

The feature pipeline consumes timing-only JSONL records as described in [ml/data/README.md](ml/data/README.md). `ml.features.keystroke_features.extract_sample` validates event timings, rejects invalid records, clips within-sample IQR outliers, then extracts statistical, typing, correction, and coarse key-family transition features plus a variable-length `(dwell, flight, one-hot key-family)` sequence. `ml.preprocessing.pipeline` imputes missing sequence values, pads and masks sequences, fits a robust scaler on explicitly supplied training data, and splits records by `session_id`. `python -m ml.data.prepare_dataset INPUT.jsonl OUTPUT_DIR` writes feature CSV and compressed sequence/mask/split arrays. `python -m ml.data.export_enrollment --output ml/data/enrollment-samples.jsonl` decrypts consented enrollment sequences from the database and exports the timing-only JSONL format for local model training. Exported data is ignored by Git. No fabricated biometric dataset is included.

Run the unit suite from the repository root with `python -m unittest discover -s tests`.

## Phase 4 Random Forest baseline

After creating a prepared dataset from consented samples belonging to at least two users, train and evaluate the closed-set classifier with:

```powershell
python -m ml.training.train_random_forest ml/data/processed/features.csv ml/data/processed/sequences.npz --model-path models/random_forest.joblib
```

This verifies sample ordering and session-disjoint splits before fitting. It saves a joblib pipeline and a separate `.metrics.json` report with validation/test accuracy, balanced accuracy, macro precision/recall/F1, confusion matrix, and multiclass ROC-AUC when each training class appears in the evaluated split. The model predicts among known training labels; these classifier metrics and probabilities are not calibrated biometric match scores or FAR/FRR/EER. The unit suite trains only on synthetic fixtures in temporary directories; no model or performance claim is produced from those fixtures.

## Phase 5 LSTM sequence model

Train the LSTM on the same prepared dataset with `python -m ml.training.train_lstm ml/data/processed/features.csv ml/data/processed/sequences.npz --checkpoint-path models/keystroke_lstm.pt`. It uses padded sequences with their original lengths, a packed-sequence LSTM, dropout, class-weighted cross-entropy, Adam, gradient clipping, validation-loss early stopping, and a best-validation checkpoint. The checkpoint includes the class mapping, sequence feature names, and training-only robust-scaler parameters. The `.metrics.json` sidecar reports validation/test classification metrics; outputs remain closed-set classifier scores, not authentication match scores. Pass `--device cpu` to force CPU or omit it to use CUDA when available.

Both trainers require actual consented labeled sessions. No production model is generated by the tests, which use temporary synthetic fixtures only.

### Optional Siamese verifier

The modular shared-weight verifier is available with `python -m ml.training.train_siamese ml/data/processed/features.csv ml/data/processed/sequences.npz --checkpoint-path models/siamese.pt`. It builds balanced genuine/impostor pairs separately inside the session-disjoint train, validation, and test splits, learns normalized sequence embeddings with contrastive loss, and reports ROC-AUC, EER, FAR, FRR, and an evaluation-only EER threshold. The saved checkpoint is an experimental artifact: the API's supported deployment path remains the LSTM encoder plus held-out distance calibrator. Do not use the Siamese report's test threshold as a production threshold; calibrate and validate it on representative data before wiring it into authentication.

## Phase 6 biometric enrollment and verification engine

`ml.inference.lstm_encoder.LSTMEncoder` loads a trained checkpoint, applies its training-only sequence scaler, and returns an LSTM embedding. `ml.biometrics.profile.create_profiles` builds an in-memory per-user median/MAD template from enrollment embeddings (five samples per user by default). `ml.biometrics.calibration.fit_distance_calibrator` pairs validation embeddings with genuine and impostor profiles, fits a logistic distance calibrator, and selects an equal-error operating threshold. Calibration must achieve at least 0.55 validation ROC-AUC or it fails closed. `AuthenticationEngine.verify` handles unknown users, insufficient sequences, model mismatch, unavailable calibration, decisive accept/reject results, and scores near the threshold as low confidence.

Run the held-out verification evaluation with:

```powershell
python -m ml.training.calibrate_verifier ml/data/processed/features.csv ml/data/processed/sequences.npz models/keystroke_lstm.pt --output models/verification.metrics.json
```

The report includes validation calibration and test FAR, FRR, EER, ROC-AUC, Brier score, and confusion matrix. The calibrator is fit only on validation pairs; the decision threshold is then evaluated on test pairs. The evaluation command keeps profiles in memory; the API builds and encrypts per-account profiles from consented enrollment sessions. Real consented samples are required for meaningful calibration.

## Phase 7 API, database, and security integration

The FastAPI app now implements the authentication, biometric, typing, analytics, and model routes listed in the project brief. Passwords use Argon2; access JWTs expire and are revocable through a database session token. Biometric sample summaries, sequences, embeddings, and profile templates are AES-GCM encrypted with user/row-bound associated data. Requests reject raw key text and require an explicit consent flag. Each account can delete its stored biometric samples and authentication history. The login limiter is in-process, so a multi-instance deployment needs a shared rate-limit store.

Alembic manages the PostgreSQL schema. Local development applies migrations with `alembic -c alembic.ini upgrade head`. Auth and biometric endpoints require non-empty secrets in `.env`; an unset key returns a configuration error instead of using a hard-coded secret.

## Phase 8 dashboard and workflow

The responsive React dashboard calls the API for account state, profile status, timing history, authentication history, and model metrics. Enrollment and verification capture only the fixed passage's physical key categories and timing. The exact text remains in the browser. Match scores remain empty until the backend returns a real verification attempt. The dashboard shows a model-unavailable state until a trained checkpoint and distance calibrator are installed.

## Phase 9 tests and Docker deployment

Run the Python unit and API integration suite from the repository root with `python -m unittest discover -s tests`. The tests use temporary synthetic fixtures and SQLite; they do not generate production model files or dashboard results. Build the frontend with `cd frontend; npm run build`.

Docker Compose starts PostgreSQL, the FastAPI service, and the Nginx-served frontend. After copying and configuring `.env`, run `docker compose up --build`. The frontend is at `http://localhost:5173`, the API docs at `http://localhost:8000/docs`, and PostgreSQL data is in the Compose volume. Docker uses the same migrations automatically before starting the API. The local development credentials in `.env.example` must be changed before use outside a local machine.

## Phase 10 real-data demonstration

1. Create several accounts and collect at least five consented enrollment samples per account. More separate sessions improve the session-based splits.
2. Export encrypted enrollment sequences with `python -m ml.data.export_enrollment --output ml/data/enrollment-samples.jsonl`.
3. Build model arrays with `python -m ml.data.prepare_dataset ml/data/enrollment-samples.jsonl ml/data/processed`.
4. Train and evaluate the closed-set RF and LSTM classifiers using the commands in Phases 4 and 5.
5. Calibrate the LSTM embedding verifier with the Phase 6 command, keeping the generated checkpoint and `distance_calibrator.joblib` in `models/`.
6. Restart FastAPI so it loads the new artifacts. Sign in, confirm the profile reports ready, then perform a verification attempt. Review the returned calibrated score and test metrics in the dashboard.

The demonstration requires consented, labeled samples from multiple users. This workspace contains no real biometric dataset or trained production checkpoint.

For local development, the Compose defaults are intentionally convenient and must not be reused as production credentials. Set `DATABASE_URL` and `CORS_ORIGINS` in `.env` as needed. Schema migrations and domain tables arrive in later phases.

