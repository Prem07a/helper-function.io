import torch
from torch import nn
from torch.utils.data import DataLoader


def train_model(model, data, target, criterion, optimizer, num_epochs):
    """
    A function to train a PyTorch model on a given dataset

    Parameters:
        - model : a PyTorch model
        - data (torch.Tensor) : input data
        - target (torch.Tensor) : target data
        - criterion : loss function
        - optimizer : optimizer function
        - num_epochs (int) : number of training epochs
    """
    for epoch in range(num_epochs):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


def test_model(model, dataloader, criterion):
    """
    A function to test a PyTorch model on a given dataset using dataloader and calculate accuracy

    Parameters:
        - model : a PyTorch model
        - dataloader (torch.utils.data.DataLoader) : dataloader for the dataset
        - criterion : loss function
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    for i, (data, target) in enumerate(dataloader):
        # Move input and target data to the GPU if available
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()

        # calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    # Print the average loss and accuracy
    print(f'Test Loss: {test_loss / len(dataloader):.4f}')
    print(f'Test Accuracy: {(correct / total) * 100:.2f}%')


def train_cnn(model, dataloader, criterion, optimizer, num_epochs):
    """
    A function to train a CNN model on a given dataset using dataloader

    Parameters:
        - model : a PyTorch CNN model
        - dataloader (torch.utils.data.DataLoader) : dataloader for the dataset
        - criterion : loss function
        - optimizer : optimizer function
        - num_epochs (int) : number of training epochs
    """
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (data, target) in enumerate(dataloader):
            # Move input and target data to the GPU if available
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Print the average loss per epoch
        if (epoch+1) % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')


def test_cnn(model, dataloader, criterion):
    """
    A function to test a CNN model on a given dataset using dataloader and calculate accuracy

    Parameters:
        - model : a PyTorch CNN model
        - dataloader (torch.utils.data.DataLoader) : dataloader for the dataset
        - criterion : loss function
    """
    # Set the model to evaluation mode
    model.eval()
    with torch.inference_mode():
        correct = 0
        total = 0
        test_loss = 0.0
        for i, (data, target) in enumerate(dataloader):
            # Move input and target data to the GPU if available
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            # calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Print the average loss and accuracy
        print(f'Test Loss: {test_loss / len(dataloader):.4f}')
        print(f'Test Accuracy: {(correct / total) * 100:.2f}%')


def create_dataloader(dataset, batch_size, num_workers, shuffle):
    """
    A function to create a PyTorch DataLoader

    Parameters:
        - dataset : a PyTorch dataset
        - batch_size (int) : the number of samples per batch
        - num_workers (int) : the number of worker threads for loading the data
        - shuffle (bool) : whether to shuffle the data before each epoch
    """
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def cnn_model(input_shape, num_classes):
    """
    A function to create a CNN model

    Parameters:
        - input_shape : the shape of the input data (i.e., the shape of a single input image)
        - num_classes (int) : the number of classes in the dataset
    """
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()

            self.conv1 = nn.Conv2d(
                in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.25)
            self.fc = nn.Linear(in_features=64*8*8, out_features=num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*8*8)
            x = self.dropout(x)
            x = self.fc(x)
            return x
    return CNN()


def save_model(model, path):
    """
    A function to save a PyTorch model

    Parameters:
        - model : a PyTorch model
        - path (str) : the file path where the model should be saved
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    A function to load a PyTorch model

    Parameters:
        - model : a PyTorch model
        - path (str) : the file path of the saved model
    """
    model.load_state_dict(torch.load(path))
    return model
