import argparse
import os
import torch
import pandas as pd

#lsfjalsdghlksghklghsaklghsadklghas fkldwgasglasgjslkfjsfklas 

import constants
from datasets.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train


SUMMARIES_PATH = "training_summaries"


def main():
    # Get command line arguments
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size}

    # Create path for training summaries
    summary_path = None
    if args.logdir is not None:
        summary_path = f"{SUMMARIES_PATH}/{args.logdir}"
        os.makedirs(summary_path, exist_ok=True)

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Summary path:", summary_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    train_csv = pd.read_csv('cassava-leaf-disease-classification/train.csv')
    

    # Initalize dataset and model. Then train the model!
    full_dataset = StartingDataset('cassava-leaf-disease-classification/train.csv', 'cassava-leaf-disease-classification/train_images')
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    model = StartingNetwork(3, 5).to(device)
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=args.n_eval,
        summary_path=summary_path,
        device=device
    )

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--n_eval", type=int, default=constants.N_EVAL)
    parser.add_argument("--logdir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
