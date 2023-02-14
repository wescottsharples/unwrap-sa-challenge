import datetime
import os
import pickle
import time

from src.config import CUSTOM_MODELS_DIR, SETFIT_MODEL_PATH
from src.data.make_dataset import load_latest_test_dataset
from src.pipelines.setfit.training import train_setfit_model

SETFIT_TRAINER_PATH = CUSTOM_MODELS_DIR / "setfit_trainer.pkl"


def main():
    # Check if the model has already been trained
    if not os.path.exists(SETFIT_MODEL_PATH):

        # Train the model
        print(f"Training model and saving to {SETFIT_MODEL_PATH}")
        model, trainer = load_latest_test_dataset()

        # Save the model and trainer
        # Use today's date for versioning
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        with open(CUSTOM_MODELS_DIR / f"setfit_model_{now}.pkl", "wb") as f:
            pickle.dump(model, f)
            os.symlink(f.name, SETFIT_MODEL_PATH)
        with open(CUSTOM_MODELS_DIR / f"setfit_trainer_{now}.pkl", "wb") as f:
            pickle.dump(trainer, f)
            os.symlink(f.name, SETFIT_TRAINER_PATH)

        # Print accuracy on test set
        metric = trainer.evaluate()  # type: ignore
        if metric and "accuracy" in metric:
            print(f"Accuracy: {metric['accuracy']:0.4f}")
    else:
        print(f"Model already exists at {SETFIT_MODEL_PATH}")


if __name__ == "__main__":
    # NOTE: I trained this model on my local 1080 Ti GPU
    # NOTE: It took about 26 minutes for me to train
    # +-----------------------------------------------------------------------------+
    # | NVIDIA-SMI 525.65       Driver Version: 527.56       CUDA Version: 12.0     |
    # |-------------------------------+----------------------+----------------------+
    # | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    # | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    # |                               |                      |               MIG M. |
    # |===============================+======================+======================|
    # |   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  N/A |
    # | 24%   42C    P8    19W / 250W |   6008MiB / 11264MiB |      3%      Default |
    # |                               |                      |                  N/A |
    # +-------------------------------+----------------------+----------------------+
    # Can also be easily trained on Google Colab
    start_time = time.time()
    main()
    print(f"Total time: {datetime.timedelta(seconds=time.time() - start_time)}")
