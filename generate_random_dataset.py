import argparse
from sklearn.datasets import make_classification
import pandas as pd


def generate_random_dataset(n_samples=1000, n_features=10, n_classes=2, random_state=42, out_path="random_dataset.csv"):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, int(n_features/2)),
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["label"] = y
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate a random classification dataset")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--features", type=int, default=10, help="Number of features")
    parser.add_argument("--classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--output", type=str, default="random_dataset.csv", help="Output CSV file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    path = generate_random_dataset(
        n_samples=args.samples,
        n_features=args.features,
        n_classes=args.classes,
        random_state=args.seed,
        out_path=args.output,
    )
    print(f"Dataset saved to {path}")


if __name__ == "__main__":
    main()
