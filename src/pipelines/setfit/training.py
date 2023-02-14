from datasets import DatasetDict  # type: ignore
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer


def model_init(model: str) -> SetFitModel:
    """Initialize a new SetFit model from our base model

    Args:
        model (str): HuggingFace model name (Sentence Transformers model)

    Returns:
        SetFitModel: SetFit model
    """
    return SetFitModel.from_pretrained(model)


def get_setfit_trainer(model: SetFitModel, dataset: dict) -> SetFitTrainer:
    """Get the SetFit trainer

    Returns:
        SetFitTrainer: SetFit trainer
    """
    # NOTE: See `scripts/search_setfit_hp.py` for hyperparameter search
    return SetFitTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        column_mapping={"entry": "text", "label": "label"},
        metric="accuracy",  # if using "f1" may have to specify average="macro" if you get a warning
        loss_class=CosineSimilarityLoss,
        learning_rate=2.67e-5,
        num_epochs=5,
        batch_size=4,
        num_iterations=10,
    )


def train_setfit_model(dataset: DatasetDict) -> tuple[SetFitModel, SetFitTrainer]:
    """Train the SetFit model

    Returns:
        dict: {
            "model": model name,
            "trainer": trainer,
        }
    """

    if "train" not in dataset.keys() or "test" not in dataset.keys():
        raise ValueError("Dataset must contain 'train' and 'test' keys.")

    # We need to start with a base model to fine-tune from the Sentence Transformers library
    BASE_MODEL = "sentence-transformers/all-MiniLM-L12-v1"
    model = model_init(BASE_MODEL)
    trainer = get_setfit_trainer(model, dataset)
    trainer.train()
    return model, trainer
