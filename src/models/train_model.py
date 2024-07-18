from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore
import toml


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    df_train, df_test = train_test_split(df, test_size=config["test_size"])

    smote = SMOTE(random_state=42)
    df_train = pd.concat(
        smote.fit_resample(df_train[config["features"]], df_train[config["target"]]),
        axis=1,
    )
    with open("config.toml", "r") as config_file:
        config = toml.load(config_file)
    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }
    rf_estimator = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rf_estimator, param_grid=params, cv=5, n_jobs=-1, scoring="f1_micro"
    )
    grid_search.fit(df_train[config["features"]], df_train[config["target"]])

    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    config["random_forest"].update(best_params)  # config update

    with open("config.toml", "w") as config_file:
        toml.dump(config, config_file)

    model = SklearnClassifier(best_rf_model, config["features"], config["target"])
    model.train(df_train)

    metrics = model.evaluate(df_test)

    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)


if __name__ == "__main__":
    main()
