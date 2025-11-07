# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLflow spike project for experimenting with MLflow's experiment tracking and model management capabilities. The project demonstrates different approaches to logging machine learning experiments using scikit-learn models trained on the Iris dataset.

## Development Setup

This project uses `uv` for Python dependency management (Python 3.13.4+).

### Installation
```bash
# Install dependencies
pip install mlflow

# Or if using uv (recommended based on uv.lock presence)
uv sync
```

### Starting MLflow Server

Two options for running the MLflow tracking server:

**Option A: Full server mode**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Option B: UI only mode**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Or simpler:
mlflow ui --port 5000
```

Access the UI at http://localhost:5000

### Running the Code
```bash
python main.py
```

## Code Architecture

### MLflow Configuration
- **Tracking URI**: `sqlite:///mlflow.db` (local SQLite database)
- **Default Experiment**: "my-first-experiment" (set at main.py:9)
- **Artifacts Storage**: Local `mlruns/` directory

### Main Components (main.py)

The codebase demonstrates three different MLflow usage patterns:

1. **`train_model()`** (lines 12-53)
   - Uses MLflow autologging with `mlflow.sklearn.autolog()`
   - Automatically logs parameters, metrics, and model artifacts without explicit logging calls
   - Creates runs under "MLflow Quickstart" experiment
   - Simplest approach with minimal code changes

2. **`train_model2()`** (lines 55-91)
   - Manual logging approach using `mlflow.start_run()` context manager
   - Explicitly logs:
     - Parameters via `mlflow.log_params()`
     - Model via `mlflow.sklearn.log_model()`
     - Metrics via `mlflow.log_metric()`
     - Tags via `mlflow.set_tag()`
   - Provides fine-grained control over what gets logged

3. **`load_model()`** (lines 92-120)
   - Demonstrates loading a logged model using `mlflow.pyfunc.load_model()`
   - Note: This function has a bug - `model_info` is undefined (line 110)
   - Shows how to make predictions with loaded models and format results

### Key Dependencies
- **mlflow**: Experiment tracking and model registry
- **pandas**: Data manipulation and result formatting
- **scikit-learn**: Model training (LogisticRegression) and dataset loading
- **Python 3.13.4+**: Required runtime version

## MLflow Concepts in Use

- **Experiments**: Logical groupings of related runs
- **Runs**: Individual executions with logged parameters, metrics, and artifacts
- **Autologging**: Automatic capture of model training details
- **Model Registry**: Storage for trained models (via `log_model()`)
- **Tags**: Custom metadata for organizing runs
