# Customer Churn Explorer

## 1. Project overview

Explore how a binary model estimates historical bank-customer churn and how the decision threshold changes predictions. Python trains the model and JavaScript runs it in the browser.

[Live demo](https://subhashdixit.github.io/customer-churn-explorer/) · [Source repository](https://github.com/subhashdixit/customer-churn-explorer)

![Customer Churn Explorer main screen](docs/images/application.png)

## 2. Motivation and problem statement

A single churn score hides the decision policy behind a flag. An analyst needs to compare missed churners with false positives and check whether changing the threshold improves the tradeoff. This explorer makes the historical model, threshold, and held-out errors visible in the same interface.

## 3. Solution and scope

Enter a single record or score a CSV. Inspect the positive-label score, adjust the decision threshold, and export results. The shipped test report uses the validation-selected threshold.

## 4. Quick start and usage

1. Open the GitHub Pages address linked from the repository About section after deployment.
2. Review the synthetic starting profile. Defaults combine training medians and common categories; they do not identify a source record.
3. Change fields and calculate the model result. Blank values use the frozen missing-data representation.
4. Adjust the positive-label threshold to explore the precision/recall tradeoff. Published test metrics remain fixed at the validation-selected threshold.
5. Download the sample CSV, retain all feature headers and upload up to **500 rows / 2 MiB**. The browser worker computes the output locally.
6. Download predictions. The output contains row numbers and results rather than echoing your financial, assessment or sensor inputs.
7. Read the model-selection chart, test results and full machine-readable evaluation in the Evidence section.



**Privacy:** this app has no inference/upload endpoint, analytics script or browser-persistent input store. The static host receives normal asset requests/connection metadata. The local batch worker is terminated after completion or timeout. Reset restores the synthetic form; it does not promise forensic erasure of browser memory. Inputs are validated, not interpreted as executable code.

## 5. High-level architecture

```mermaid
flowchart LR
    A["Bank churn dataset"] --> B["Offline Python training"] --> C["Versioned numeric model"] --> D["Browser inference"]
```

## 6. End-to-end request workflow

```mermaid
sequenceDiagram
    actor Visitor
    participant UI as Application
    participant Logic as Processing layer
    Visitor->>UI: Enter a profile or upload a CSV
    UI->>Logic: Validate and score feature rows
    alt Valid input
        Logic-->>UI: Churn scores and threshold decisions
        UI-->>Visitor: Render or export the result
    else Invalid input
        Logic-->>UI: Input error
        UI-->>Visitor: Show actionable feedback
    end
```

## 7. Data sources, inputs, and outputs

- Source file: `data/Churn_Modelling.csv`.
- Original source rows: **10,000**; valid-target rows used: **10,000**.
- Source SHA-256: `ee2a5c9d8daf3aed0d91c4d883a84ebb77d271447bebc0f0c80aafb0f288ceab`.
- Selected features: **9**. [Full names, units, categories and bounds](docs/feature-contract.md).
- Attribution/provenance: Original repository dataset; upstream annotation/collection provenance is not independently established.

Source numeric values outside the documented serving contract are counted and treated as missing in training, rather than being silently converted to valid measurements. The counts are recorded in [reports/metrics.json](reports/metrics.json). Unsupported targets are excluded. Categories are normalized using the documented source adapter.

## 8. Data flow and ETL pipeline

```mermaid
flowchart LR
    A["Source CSV"] --> B["Validate features and split labeled rows"] --> C["Fit and select on validation"] --> D["Evaluate on held-out data"] --> E["Export model, metrics, and parity fixtures"]
```

## 9. Low-level design and interfaces

```mermaid
flowchart LR
    A["app.js: input and UI state"] --> B["model-worker.js: local scoring"] --> C["model-engine.js: numeric inference"] --> D["model.json: fitted parameters"]
```

The static app validates inputs and scores them locally. It exposes a form and CSV interface.

## 10. Concepts and algorithms

**Missing values:** numeric medians are learned from training rows. A median is robust to outliers but ignores relationships between fields. Missing categories have an explicit token. All-empty profiles are rejected.

**Encoding:** numeric values become `(x − training_mean) / training_scale` after imputation. A zero-variance scale becomes 1. Categories use a training-only one-hot vocabulary, not an invented numeric ordering. Unsupported categories are rejected by serving; a new source category needs a reviewed adapter/model update.

**Linear candidate:** regularized logistic regression learns a weighted sum of transformed features, then maps it through the sigmoid `1/(1+exp(−margin))`. It minimizes regularized log loss. It is fast and compact but assumes a linear decision boundary in transformed feature space. A score is not a calibrated probability for a current person or vehicle.

**Random forest candidate:** 64 decision trees learn threshold splits and average their class distributions. Depth and leaf-size limits reduce complexity. Forests capture nonlinear interactions without a neural network, but use larger artifacts, can overfit a small dataset and generally cannot extrapolate reliably outside observed target ranges. Class-balanced weights are fitted from training labels.

The serving tree implementation reproduces scikit-learn's float32 split inputs with `Math.fround`. Linear inference uses ordinary double-precision arithmetic. Node parity fixtures compare JavaScript outputs with real held-out Python predictions to an absolute tolerance of `1e-7`. They are arithmetic checks, not extra accuracy measurements.

## 11. Technology choices and dependencies

| Technology | Role and rationale |
| --- | --- |
| Python and scikit-learn | Train and compare classifiers against a majority baseline. |
| JavaScript ES modules | Reproduce the scoring pipeline using exported parameters without a model server. |
| Web Worker | Keep batch scoring off the interface thread. |
| Node and GitHub Actions | Verify parity, input contracts, and release readiness before publishing. |

[Dependency versions and reproducible setup](DEPENDENCIES.md)

## 12. Training, inference, and evaluation

GroupShuffleSplit; identical selected profiles share a group. Fixed seeds **42 and 43** produce training / validation / test rows **6,000 / 2,000 / 2,000**. Grouping limits the documented form of leakage; this is not time-separated or externally validated deployment performance.

Preprocessing and both candidates fit **training rows only**. Validation average precision selects **random_forest**. The selected model is not refitted on test data. The held-out set supplies the final reported metrics.

| Validation candidate | Average precision: higher is better |
|---|---:|
| linear | 0.4353 |
| random_forest | 0.6614 |

| Held-out test metric | Selected model | Training-only baseline |
|---|---:|---:|
| Accuracy | 0.8185 | 0.7910 |
| Macro F1 | 0.7455 | 0.4417 |
| Average precision | 0.6964 | 0.2090 |
| ROC AUC | 0.8483 | 0.5000 |

Positive-class precision: **0.5538**; recall: **0.6770**; F1: **0.6093** at validation-selected threshold **0.51**. Confusion matrix, rows actual and columns predicted: `[[1354, 228], [135, 283]]`.

**Thresholds:** model scores are ranked before choosing a decision threshold. Validation chooses the default; test labels do not choose it. Lower thresholds generally increase recall and false positives. Accuracy can look strong under class imbalance. Average precision emphasizes positive-class ranking but depends on prevalence; ROC AUC can hide poor precision. Precision and recall describe different error costs, while F1 balances them without encoding every real-world cost.



Full measurements and integrity metadata are in [reports/metrics.json](reports/metrics.json). No synthetic performance figures are used.

### Reproduce training

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes --only-binary=:all: -r requirements-training.lock.txt
python scripts/train.py
python scripts/check_release.py
python -m black --check scripts
python -m pylint scripts
npm test
```

On Windows, activate `.venv\Scripts\activate` instead. Training reads the included source file, reconstructs the split and writes `site/model.json`, `site/sample.json`, `site/metrics.json` and `reports/`. Source data is not fetched live by an end user. Review source rights, model results and hash changes before committing a retrained release.

The two insurance apps deliberately share source/training design; their results must not be counted as independent experiments. Other legacy preprocessing/model objects are preserved but are not silently mixed with this release.

## 13. Limitations and data handling

Churn is a historical engagement outcome, not a credit-default or loan-approval label. Customer identifiers, surnames and gender are not model inputs. Age and geography remain source predictors; fairness and proxy effects have not been established.

The target is the source Exited flag. Names, customer IDs and gender are excluded. Historical churn associations do not establish why a customer left or justify lending decisions.

The original repository was `Customer_Churn_Prediction_and_GUI`. Existing notebooks, Python applications and model files are preserved for context; previous documentation is in `archive/`. The published `site/` uses the documented new model, not an implicitly trusted legacy pickle.

## 14. Installation and local setup

Use Node.js **22+** for JavaScript development/checks and Python **3.10 or 3.11** for training and the simple local web server.

```bash
git clone https://github.com/subhashdixit/customer-churn-explorer.git
cd customer-churn-explorer
npm ci
npm test
npm run lint
npm run format
python3 -m http.server 8000 --directory site
```

Open `http://127.0.0.1:8000`. Serve over HTTP rather than opening `index.html` with `file://`, because model fetching, modules and workers need an appropriate origin. Production uses HTTPS on Pages. No API key, Docker container or backend account is needed to use this browser app.

## 15. Repository structure

```text
site/index.html          accessible HTML interface
site/style.css           responsive local styling
site/app.js              form, results, plots and batch orchestration
site/model-engine.js     validated framework-free inference and CSV parsing
site/model-worker.js     bounded local batch work
site/model.json          frozen numeric model; no executable pickle
site/metrics.json        evaluation and source/artifact hashes
site/project.json        purpose, limitations and view configuration
scripts/train.py         reproducible project-specific source adapter/training
scripts/check_release.py source/model identity checks
tests/model.test.js      real-model parity, validation, CSV and metadata checks
reports/                 full metrics and held-out parity fixtures
docs/feature-contract.md names, units, categories and supported bounds
archive/                 preserved previous documentation
.github/workflows/       test and GitHub Pages release pipeline
```

## 16. Deployment and operations

```mermaid
flowchart LR
    A["GitHub main"] --> B["GitHub Actions checks"] --> C["Publish site directory"] --> D["GitHub Pages"]
```

```mermaid
flowchart LR
 A[Feature branch and PR] --> B[Black + Pylint + ESLint + Prettier]
 B --> C[Source hashes + Node behavior/parity tests]
 C --> D[Review and merge main]
 D --> E[Repeat checks; upload only site folder]
 E --> F[GitHub Pages HTTPS deployment]
 F --> G[Check form, CSV and published model identity]
```

**CI** repeats code and behavior checks for changes, including a real Python/JavaScript model comparison. It catches regressions and packaging mistakes; it does not certify the statistical validity of the dataset. **CD** publishes the checked static artifact from `main`. PRs run checks without publishing. Actions are pinned to reviewed commit hashes, and only the deployment job gets Pages publishing permissions.

Enable **Settings → Pages → Source → GitHub Actions**. The workflow `.github/workflows/pages.yml` builds/tests and uploads `site/`; it does not publish legacy datasets, Python environments or reports as website assets. The repository About/README link points to the verified live app after rollout.

**Hosting** supplies the public asset server; **deployment** replaces its files with a particular revision. GitHub Pages needs no server process, paid inference quota or container here. Docker can be useful for a Python training environment, but is unnecessary for serving these static browser files.

**Versioning:** Git records code, configuration, model weights and evaluation together. Tag known-good releases. Update preprocessing and the exported model as one reviewed change. The model's schema version and report hashes identify the inference contract.

**Rollback:** revert a faulty change through a PR and let the same workflow publish the previous compatible artifacts. Restore model, schema and metrics together, then repeat a synthetic form/CSV check. Deleting a review branch does not delete its merged changes. Main is the maintained branch, with merged branches removed automatically.

**Retraining versus redeployment:** retraining changes learned parameters and needs a new evaluation review; redeploying unchanged files does not create a new model. Useful future work includes time-separated validation, subgroup/coverage analysis, better source documentation and calibrated uncertainty. Those are not claimed as completed.

References: [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages), [scikit-learn evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), [common leakage pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

## 17. License, credits, and support

Project by **Subhash Dixit**. See the [repository license](LICENSE) for code terms. Report bugs or documentation issues with synthetic reproduction inputs.

[GitHub Issues](https://github.com/subhashdixit/customer-churn-explorer/issues)
