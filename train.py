import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as tqdm


def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path, device, name, writer
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), weight_decay = 0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    if summary_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    best_test_acc = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        
        n_correct = 0
        loss_sum = 0
        n_total = 0

        model.train()

        loop = tqdm(train_loader, desc=f"Train {epoch}")

        # Loop over each batch in the dataset
        for images, labels in loop:
            # print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            images = images.to(device)
            labels = labels.to(device)

            # Forward propogation
            outputs = model.forward(images)

            # Backpropogation with Gradient Descent
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # collect statistics
            n_correct += torch.sum(outputs.argmax(dim=1) == labels).item()
            loss_sum += loss.item()
            n_total += len(labels)

            # update loop
            train_acc = n_correct / n_total
            train_loss = loss_sum / n_total
            loop.set_postfix({"acc:": f"{train_acc : .03f}", "loss:": f"{train_loss : .03f}"})

            # # Periodically evaluate our model + log to Tensorboard
            # if step % n_eval == 0:
            #     # TODO:
            #     # Compute training loss and accuracy.
            #     # Log the results to Tensorboard.

            #     # TODO:
            #     # Compute validation loss and accuracy.
            #     # Log the results to Tensorboard.
            #     # Don't forget to turn off gradient calculations!
            step += 1

        
        test_acc, test_loss = evaluate(val_loader, model, loss_fn, epoch, device)
        best_test_acc = max(test_acc, best_test_acc)
        writer.add_scalar('acc/train', train_acc, epoch)
        writer.add_scalar('acc/test', test_acc, epoch)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('acc/best_test', best_test_acc, epoch)
        writer.flush()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, epoch, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    n_correct = 0
    loss_sum = 0
    n_total = 0

    model.eval()

    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Valid: {epoch}")
        for images, labels in loop:
            # move data to gpu
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model.forward(images)
            loss = loss_fn(outputs, labels).mean()

            # collect statistics
            n_correct += torch.sum(outputs.argmax(dim=1) == labels).item()
            loss_sum += loss.item()
            n_total += len(labels)

            loop.set_postfix({"acc:": f"{n_correct / n_total : .03f}", "loss:": f"{loss_sum / n_total : .03f}"})

            return n_correct / n_total, loss_sum / n_total