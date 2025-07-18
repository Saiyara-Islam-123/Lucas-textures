import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_texture_data  # Importing the data loading function from dataset.py
from NeuralNetwork import Net, SupervisedNet  # Import the autoencoder and supervised model from Net.py
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import numpy as np
from dist import *
import pandas as pd

def train_supervised(model, trainloader, device, lr, epochs=15):
    """
    Trains a given model using supervised learning with a provided dataloader, device,
    and a specified number of epochs. The function uses the CrossEntropyLoss for
    classification tasks, and the Adam optimizer for parameter updates. During training,
    it visualizes the loss and accuracy over epochs using real-time plots and saves the
    model's weights after each epoch. Additionally, the computed loss and accuracy values
    are saved for post-training analysis.

    :param model: Neural network model to be trained
    :type model: torch.nn.Module
    :param trainloader: DataLoader providing batches of training data
    :type trainloader: torch.utils.data.DataLoader
    :param device: Device on which the model and data will be loaded (e.g., 'cpu' or 'cuda')
    :type device: torch.device
    :param epochs: Number of training epochs; defaults to 15
    :type epochs: int, optional
    :return: None
    :rtype: None
    """
    # Define the loss function specific for supervised learning
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    model.train()

    # Create the folder for saving results if it doesn't exist


    # Create plot lines for loss and accuracy
    loss_values = []
    accuracy_values = []
    #loss_line, = ax1.plot([], [], label="Loss", color="blue")
    #accuracy_line, = ax2.plot([], [], label="Accuracy", color="green")
    #ax1.legend()
    #ax2.legend()

    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []


    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        batch = 0
        for images, labels in trainloader:
            # Prepare the images and labels
            images = images.to(device)  # Move input images to the same device as the model

            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)


            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            encoder_outputs = model.encoder_output
            zero, zero_one, one = sampled_all_distance(encoder_outputs, labels)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            accuracy_values.append(accuracy)
            print(accuracy)


            torch.save(model.state_dict(), f"weights/sup/sup_net_weights_ lr={lr} "+str(epoch)+  " " + str(batch) +".pth")
            print("sup_net model weights saved as sup_net_weights.pth'")
            batch += 1

        # Compute average loss and accuracy for the epoch
        avg_loss = running_loss / len(trainloader)
        loss_values.append(avg_loss)

        print(f"Supervised epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f},")

    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0, 0)]
    df["within 1"] = avg_distances[(1, 1)]
    df["between"] = avg_distances[(0, 1)]
    df["acc"] = accuracy_values
    df.to_csv(f"LR={lr}, Distance every batch sup", index=False)


if __name__ == "__main__":
    file = "filtered.csv"

    # Load the data
    trainloader, valloader, testloader = load_texture_data(file, batch_size=64)

    # Initialize the net and load the lastest encoder weights
    unsup_net = Net()

    weight_path = "weights/unsup/lr=0.001 0 31.pth"

    unsup_net.load_state_dict(torch.load(weight_path))


    # Initialize the supervised model using the encoder from the trained autoencoder
    sup_net = SupervisedNet(unsup_net)

    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sup_net.to(device)

    # Train the supervised model
    train_supervised(sup_net, trainloader, device, epochs=1, lr=0.005)