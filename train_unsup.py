import torch
import torch.nn as nn
import torch.optim as optim
from NeuralNetwork import *
import os
import numpy as np
from dist import *
import pandas as pd
from dataset import *

def train_unsupervised(model, trainloader, device, lr, epochs=5):

    # Define the loss function specific for autoencoder
    criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    loss_values = []


    avg_distances = {}
    avg_distances[(0,0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []

    for epoch in range(epochs):
        running_loss = 0.0
        batch = 0
        for images, labels in trainloader:
            images = images.to(device)  # Move input images to the same device as the model

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            zero, zero_one, one = sampled_all_distance(model.encoded, labels)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)

            avg_loss = running_loss / len(trainloader)
            loss_values.append(avg_loss)


            torch.save(model.state_dict(), "weights/unsup/" + "lr=" + str(lr) + " " +str(epoch)+ " " + str(batch) +".pth")
            print("unsup_net model weights saved as 'unsup_net_weights.pth'")

            batch += 1

        print(f"Unsupervised epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0,0)]
    df["within 1"] = avg_distances[(1,1)]
    df["between"] = avg_distances[(0,1)]
    df.to_csv(f"LR={lr}, Distance every batch unsup.csv", index=False)

if __name__ == "__main__":

    file = "filtered.csv"

    # Load the data
    trainloader, valloader, testloader = load_texture_data(file,batch_size=64)

    # Initialize the autoencoder model
    unsup_net = Net()
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unsup_net.to(device)

    # Train the model
    train_unsupervised(unsup_net, trainloader, device, lr=0.001, epochs=1)
