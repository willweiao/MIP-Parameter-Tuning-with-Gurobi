import argparse
from src.collect_dataset import collect_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["collect"], required=True)
    args = parser.parse_args()

    if args.mode == "collect":
        collect_dataset(
            instance_dir="data/instances",
            param_file="data/params/param_sets.json",
            output_path="data/processed/dataset.csv"
        )