"""
Tests for main.py MLflow functions.
"""

import tempfile

import mlflow
import pytest

from main import load_model, train_model, train_model2


@pytest.fixture
def temp_mlflow_tracking():
    """Create a temporary MLflow tracking directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"file://{tmpdir}/mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test-experiment")
        yield tracking_uri
        # Cleanup is automatic with TemporaryDirectory


def test_train_model(temp_mlflow_tracking):
    """Test that train_model executes without errors."""
    # This should complete without raising exceptions
    train_model()

    # Verify that a run was created
    experiment = mlflow.get_experiment_by_name("test-experiment")
    assert experiment is not None

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0


def test_train_model2(temp_mlflow_tracking):
    """Test that train_model2 executes and logs expected artifacts."""
    train_model2()

    # Verify that a run was created
    experiment = mlflow.get_experiment_by_name("test-experiment")
    assert experiment is not None

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0

    # Check that the run has logged metrics
    latest_run = runs.iloc[0]
    assert "metrics.accuracy" in runs.columns
    assert latest_run["metrics.accuracy"] > 0

    # Check that params were logged
    assert "params.solver" in runs.columns
    assert latest_run["params.solver"] == "lbfgs"


def test_load_model(temp_mlflow_tracking):
    """Test that load_model can load a trained model and make predictions."""
    # First train a model
    train_model2()

    # Get the run ID
    experiment = mlflow.get_experiment_by_name("test-experiment")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    run_id = runs.iloc[0].run_id

    # Load the model and make predictions
    model_uri = f"runs:/{run_id}/iris_model"
    result = load_model(model_uri)

    # Verify the result DataFrame has expected columns
    assert "actual_class" in result.columns
    assert "predicted_class" in result.columns
    assert len(result) == 30  # 20% of 150 samples with random_state=42

    # Verify predictions are valid iris classes (0, 1, or 2)
    assert result["predicted_class"].isin([0, 1, 2]).all()


def test_model_accuracy(temp_mlflow_tracking):
    """Test that the trained model achieves reasonable accuracy."""
    train_model2()

    experiment = mlflow.get_experiment_by_name("test-experiment")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    latest_run = runs.iloc[0]

    # Iris dataset is relatively easy, expect high accuracy
    accuracy = latest_run["metrics.accuracy"]
    assert accuracy > 0.8, f"Model accuracy {accuracy} is too low"
