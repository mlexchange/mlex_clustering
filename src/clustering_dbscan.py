import argparse
import logging

import pandas as pd
import yaml

from clustering import run_dbscan
from utils.parameters import DBSCANParameters, IOParameters
from utils.tiled_dataset import TiledDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    # Load parameters
    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    # Validate parameters
    io_parameters = IOParameters(**parameters["io_parameters"])
    model_parameters = DBSCANParameters(**parameters["model_parameters"])
    logger.info(f"Parameters loaded: {model_parameters}")

    dataset = TiledDataset(
        io_parameters.root_uri,
        io_parameters.results_tiled_uri,
        io_parameters.data_tiled_api_key,
        io_parameters.results_tiled_api_key,
    )
    data = dataset.load_data_from_tiled(io_parameters.data_uris[0])

    labels = run_dbscan(data, model_parameters.eps, model_parameters.min_samples)

    # Convert labels to a pandas DataFrame
    labels_df = pd.DataFrame(labels, columns=["cluster_label"])

    # Save results
    results_path = f"{io_parameters.results_dir}/{io_parameters.uid_save}.parquet"
    labels_df.to_parquet(results_path)

    dataset.write_results(labels_df, io_parameters, results_path, parameters)
