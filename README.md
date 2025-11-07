# mlflow-spike
MLflow spike project for experimenting with MLflow's experiment tracking and model management capabilities.

## Installation

### Prerequisites
- Python 3.13.4+
- `uv` package manager (recommended) or `pip`

### Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install mlflow
```

## Development

### Code Formatting and Linting

This project is configured with automated code formatting and linting using `isort` and `ruff`.

**Manual formatting:**
```bash
# Format imports with isort
isort .

# Format code with ruff
ruff format .

# Lint and auto-fix issues
ruff check . --fix
```

**Quick format (all at once):**
```bash
isort . && ruff format . && ruff check . --fix
```

### Building the Project

The project uses `hatchling` as the build backend with a custom build hook that automatically runs `isort` and `ruff` during the build process.

**Build the project:**
```bash
uv build
```

This will:
1. Run `isort` to sort imports
2. Run `ruff format` to format code
3. Run `ruff check` to lint and auto-fix issues
4. Create distribution packages in the `dist/` directory

**Build artifacts:**
- `dist/mlflow_spike-<version>-py3-none-any.whl` - Wheel distribution
- `dist/mlflow_spike-<version>.tar.gz` - Source distribution

## Running the Project

```bash
python main.py
```

## MLflow Setup

### Starting MLflow Server

**Option A: Full server mode**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Option B: UI only mode**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Simplified UI mode:**
```bash
mlflow ui --port 5000
```

### Access MLflow UI
Navigate to http://localhost:5000 in your browser.

## Project Configuration

- **Python Version:** 3.13.4+
- **Build System:** hatchling with custom build hooks
- **Code Formatting:** isort + ruff (line length: 88)
- **MLflow Backend:** SQLite (`mlflow.db`)
- **Artifacts Storage:** Local `mlruns/` directory




