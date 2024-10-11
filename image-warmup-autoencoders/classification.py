import argparse
from anomalib.data import MVTec
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.data import Folder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or inference using Padim."
    )
    parser.add_argument(
        "mode",
        choices=["train", "inference"],
        help="Mode to run the script: 'train' or 'inference'.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create the datamodule
    datamodule = Folder(
        name="autoencoder_img",
        root=args.data_root,
        normal_dir="normal",
        abnormal_dir="flip",
        task="classification",
    )

    # Setup the datamodule
    datamodule.setup()

    engine = Engine(max_epochs=10)

    if args.mode == "train":
        model = Padim()
        print("Training the model...")
        engine.train(datamodule=datamodule, model=model)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()