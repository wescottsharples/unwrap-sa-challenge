"""Compares all models under consideration for the challenge and outputs a table of results."""

import asyncio
import datetime
import os
import pickle
import time

import numpy as np
import pandas as pd
from datasets import Dataset  # type: ignore
from sklearn.metrics import classification_report

from src.config import DATA_DIR, DATASET_PATH
from src.data.make_dataset import load_dataset_from_file, load_latest_test_dataset
from src.pipelines import get_all_pipelines


async def run_comparison(all_pipelines: dict, dataset: Dataset) -> dict:
    """Compare the performance of multiple sentiment analysis models

    Args:
        all_pipelines (dict): A nested dictionary of pipelines to compare
        dataset (Dataset): The dataset to use for model evaluation

    Returns:
        dict: A dictionary with model names as keys and a dictionary
            with the following key-value pairs as values:
                - "pipeline": The sentiment analysis model used.
                - "report": The classification report generated by the model.
                - "accuracy": The accuracy score of the model.
                - "time": The time taken for inference.
                - "df": The dataframe of the model's predictions and actual labels.
    """

    results = {}
    for category, pipelines in all_pipelines.items():
        results[category] = {}
        for name, pipeline in pipelines.items():

            # Run the pipeline
            print(f"Running {category} - {name}...")
            start_time = time.time()

            if getattr(pipeline, "expects_titles", False):
                # If the pipeline supports titles, pass them in as well
                pipe_inputs = zip(dataset["title"], dataset["entry"])
            else:
                # Otherwise, just pass in the entries
                pipe_inputs = dataset["entry"]

            if getattr(pipeline, "is_async", False):
                # If the pipeline is async, run it asynchronously
                preds = await pipeline(pipe_inputs)
            else:
                # Otherwise, run it synchronously
                preds = pipeline(pipe_inputs)

            assert len(preds) == len(dataset)
            if isinstance(preds[0], dict):
                # If the pipeline returns a dictionary, extract the label
                preds = [pred["label"] for pred in preds]

            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Finished {name} in {time_taken:.2f} seconds.")

            # Evaluate the pipeline
            df = dataset.to_pandas()
            assert isinstance(df, pd.DataFrame)  # to get better type hints
            df["preds"] = preds
            report_str = classification_report(dataset["label_str"], preds)
            report = classification_report(dataset["label_str"], preds, output_dict=True)
            accuracy = np.mean(df["preds"] == df["label_str"])

            # Print the results
            print(report_str)

            # Save the results
            results[category][name] = {
                "pipeline": str(pipeline),
                "report": report,
                "report_str": report_str,
                "accuracy": accuracy,
                "time": time_taken,
                "df": df,
            }

    return results


def save_readable_results_file(results: dict, now_str: str):
    """Save an easy-to-read high-level summary of the results."""

    flattened_results = {}
    for category, category_results in results.items():
        for name, result in category_results.items():
            key = f"{category} - {name}"
            flattened_results[key] = result

    # sort results absed on accuracy
    flattened_results_sorted = sorted(
        flattened_results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    with open("results.txt", "w") as f:
        f.write("Sentiment Analysis Results\n")
        f.write(f"Date: {now_str}\n\n\n")
        f.write(f"Total models evaluated: {len(flattened_results)}\n\n\n")
        for name, result in flattened_results_sorted:
            f.write(f"Results for {name}: \n")
            f.write(f"Accuracy: {result['accuracy']:.2f}\n")
            f.write(f"Inference time: {result['time']:.2f} seconds\n\n")
            # use rich to print the classification report
            f.write(f"{result['report_str']}\n")
            f.write(f"Example predictions: \n")
            df = result["df"][["entry", "label_str", "preds"]].head(3)
            # truncate the entries
            df["entry"] = df["entry"].apply(lambda x: x[:45] + "...")
            f.write(f"{df.to_markdown()}\n")
            f.write(f"--------------------------------------------\n\n\n")

        f.write(f"See data/results/results_{now_str}.pkl for more details.\n")


async def main(use_local_test_data: bool = False):
    """Main function

    Args:
        use_local_test_data (bool, optional): Whether to use the test data for evaluation.
            Defaults to False.
    """

    if use_local_test_data:
        dataset = load_dataset_from_file(DATASET_PATH)["test"]
    else:
        # Grab the latest dataset from the database
        dataset = load_latest_test_dataset()

    # Get the pipelines
    print("Loading the pipelines...")
    pipelines = get_all_pipelines()

    # Run the comparison
    print("Running the comparison...")
    results = await run_comparison(pipelines, dataset)

    # Save the results
    print("Saving the results...")
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{now_str}.pkl"
    results_dir = os.path.join(DATA_DIR, "results")  # type: ignore
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    pickle.dump(results, open(path, "wb"))
    save_readable_results_file(results, now_str)

    print("Done!")


if __name__ == "__main__":
    # Run the comparison
    asyncio.run(main())
