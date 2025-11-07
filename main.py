import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my-first-experiment")


def train_model():
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()

    # Just train the model normally
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)


def train_model2():
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Train the model
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)

        # Log the model
        mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

        # Predict on the test set, compute and log the loss metric
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Optional: Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for iris data")


def load_model(model_uri):
    """
    Load a trained model and make predictions on test data.

    Args:
        model_uri (str): MLflow model URI (e.g., "runs:/<run_id>/iris_model"
            or "models:/<model_name>/<version>")

    Returns:
        pd.DataFrame: DataFrame with features, actual classes, and predicted classes
    """
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets (using same random_state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    predictions = loaded_model.predict(X_test)

    iris_feature_names = datasets.load_iris().feature_names

    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    return result


def main():
    experiment = mlflow.get_experiment_by_name("my-first-experiment")
    if experiment:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs.empty:
            run_id = runs.iloc[0].run_id
            result = load_model(f"runs:/{run_id}/iris_model")
            print("\nPredictions on test data:")
            print(result.head())
        else:
            print("No runs found. Please run train_model2() first to create a model.")
    else:
        print(
            "Experiment 'my-first-experiment' not found. "
            "Please run train_model2() first."
        )


if __name__ == "__main__":
    main()
