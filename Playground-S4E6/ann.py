from torch import nn
from torch.utils.data import DataLoader
import torch
from module.AcademicDataset import AcademicDataset
from module.model import NeuralNetwork
from torchvision.transforms import ToTensor

categories = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
num_classes = 3


def transform_y(y: str):
    label = categories[y]
    one_hot_y = torch.zeros(num_classes, dtype=torch.float).scatter_(dim=0, index=torch.tensor(label), value=1)
    return one_hot_y


def reverse_map(value: int):
    for key, val in categories.items():
        if val == value:
            return key
    return -1


def train(dataloader, model_, loss, optim, bsize):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model_.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        predict = model_(X)
        loss = loss(predict, y)

        # Backpropagation
        loss.backward()
        optim.step()
        optim.zero_grad()

        if batch % 200 == 0:
            loss, current = loss.item(), batch * bsize + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model_, loss, title: str = "Validation"):
    model_.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            predict = model_(X)
            test_loss += loss(predict, y).item()
            correct += (predict.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"{title} -- Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")


if __name__ == "__main__":
    training_data = AcademicDataset(root="asdata/train.csv", transform=ToTensor(),
                                    target_transform=transform_y)
    val_data = AcademicDataset(root="asdata/val.csv", transform=ToTensor(),
                               target_transform=transform_y)

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")
    print(f"Using {device} device")

    model = NeuralNetwork(num_classes=3)

    learning_rate = 1e-4
    batch_size = 32
    epochs = 100

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, batch_size)
        test(train_dataloader, model, loss_fn, title="Train")
        test(val_dataloader, model, loss_fn)
    print("Done!")
