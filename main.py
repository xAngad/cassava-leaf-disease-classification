import argparse
import os
import torch
import pandas as pd
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter



import constants
from dataset import StartingDataset
from networks import StartingNetwork
from networks import TransferNetwork
from train import starting_train, evaluate


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
    train_size = int(0.05 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    model = TransferNetwork(3, 5).to(device)
    name = "test run"
    tf_writer = SummaryWriter(os.path.join('log', name))

    USE_SAVED_MODEL = True

    if USE_SAVED_MODEL:
        # Load saved state dict to model
        model.load_state_dict(torch.load('model_weights.pth'))

        # load paramters for evaluation loop
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        epoch = "EVAL_MODE"

        # Evaluate model using saved weights
        model.eval()
        acc, loss = evaluate(val_loader, model, loss_fn, epoch, device)
        print(f"Accuracy: {acc}\t\tLoss: {loss}")

    else:
        starting_train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            hyperparameters=hyperparameters,
            n_eval=args.n_eval,
            summary_path=summary_path,
            device=device,
            name=name,
            writer=tf_writer
        )
        tf_writer.close()

    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--n_eval", type=int, default=constants.N_EVAL)
    parser.add_argument("--logdir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
