"""Hyperparameter search to find the best SetFit training parameters"""


from setfit import SetFitModel, SetFitTrainer

from src.config import DATASET_PATH
from src.data.make_dataset import load_dataset_from_file


def get_model_params(params):
    """Extract model parameters from the given dictionary.

    Args:
        params (dict): A dictionary containing the model parameters.

    Returns:
        tuple: A tuple containing the model ID and the model parameters.
    """
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    model_id = params.get("model_id", "sentence-transformers/all-mpnet-base-v2")
    model_params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return model_id, model_params


def get_hp_space(trial) -> dict:
    """Define the hyperparameter search space.

    Args:
        trial (Trial): An object representing a single trial in the hyperparameter search.

    Returns:
        dict: A dictionary containing the hyperparameters for the trial.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
        "num_iterations": trial.suggest_categorical("num_iterations", [5, 10, 20, 40, 80]),
        "seed": trial.suggest_int("seed", 1, 40),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
        "model_id": trial.suggest_categorical(
            "model_id",
            [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L12-v1",
            ],
        ),
    }


def create_model(params) -> SetFitModel:
    """Create a SetFit model using the given parameters.

    Args:
        params (dict): A dictionary containing the model parameters.

    Returns:
        SetFitModel: A SetFit model created from the given parameters.
    """
    model_id, model_params = get_model_params(params)
    return SetFitModel.from_pretrained(model_id, **model_params)


def run_hyperparameter_search():
    """Run the hyperparameter search.

    Returns:
        Trial: An object representing the best run of the hyperparameter search.
    """
    dataset = load_dataset_from_file(DATASET_PATH)
    trainer = SetFitTrainer(
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        model_init=create_model,  # type: ignore
        column_mapping={"entry": "text", "label": "label"},
    )
    best_run = trainer.hyperparameter_search(
        direction="maximize", hp_space=get_hp_space, n_trials=100
    )

    return best_run


if __name__ == "__main__":
    best_run = run_hyperparameter_search()
