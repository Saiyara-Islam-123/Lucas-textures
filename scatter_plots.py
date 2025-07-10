import torch
from NeuralNetwork import *
from sklearn.manifold import TSNE
from dataset import load_texture_data

import matplotlib.pyplot as plt
import os
import pandas as pd

label_colors = {
    1: "limegreen",
    0: "green"

}


def of_two(matrix):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(matrix)

def scatter_plot(train_type, weights, lr, batch, epoch, loc):

    if train_type == "sup":
        unsup_net = Net()
        unsup_weights_path = "weights/unsup/lr=0.001 0 31.pth"
        unsup_net.load_state_dict(torch.load(unsup_weights_path))

        m = SupervisedNet(unsup_net)
        m.load_state_dict(torch.load(weights))

    elif train_type == "unsup":
        m = Net()
        m.load_state_dict(torch.load(weights))

    m_outputs = []

    excel_file = "filtered.csv"
    dataset, _, _ = load_texture_data(excel_file, batch_size=500)

    for images, labels in dataset:

        if train_type == "sup":
            _ = m(images)
            encoder_outputs = m.encoder_output
            m_outputs.append(encoder_outputs.detach().numpy())
            labels_arr=(labels.detach().numpy())

        elif train_type == "unsup":
            _ = m(images)
            encoder_outputs = m.encoded
            m_outputs.append(encoder_outputs.detach().numpy())
            labels_arr=(labels.detach().numpy())
        break

    #standard_dataset = pd.read_csv("Standard Dataset").to_numpy()


    X_tnse = of_two(m_outputs[0])

    #_, mx, _ = procrustes(standard_dataset, X_tnse)
    mx = X_tnse
    pc1 = mx[:, 0]
    pc2 = mx[:, 1]

    for i in range(mx.shape[0]):
        x_axis = pc1[i]
        y_axis = pc2[i]

        color = label_colors[labels_arr[i]]

        plt.scatter(x_axis, y_axis, color=color, s=10)

    plt.scatter([], [], color= "limegreen", label ="Cat1" )
    plt.scatter([], [], color= "green", label='Cat2')


    plt.title( train_type +f"ervised training scatterplot, Epoch: {epoch} Batch: {batch}" )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.legend()
    plt.savefig(f"plots/scatter_plots/{loc}/{train_type} lr = {lr}, {epoch} {batch}.png")
    plt.show()

def plot_raw_data():
    excel_file = "filtered.csv"
    dataset, _, _ = load_texture_data(excel_file, batch_size=500)

    im = []

    for images, labels in dataset:
        im.append(images)
        labels_arr = (labels.detach().numpy())
        break

    print(im[0].shape)
    X_tnse = of_two(im[0].reshape(500, 128*128*3))
    pc1 = X_tnse[:, 0]
    pc2 = X_tnse[:, 1]

    for i in range(len(labels_arr)):
        x_axis = pc1[i]
        y_axis = pc2[i]

        color = label_colors[labels_arr[i]]

        plt.scatter(x_axis, y_axis, color=color, s=10)

    plt.scatter([], [], color= "limegreen", label ="Cat1" )
    plt.scatter([], [], color= "green", label='Cat2')


    plt.title( "No training scatterplot" )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.legend()
    plt.savefig("plots/scatter_plots/no_train/no_training.png")
    plt.show()



if __name__ == '__main__':

    #plot_raw_data()
    '''
    for i in range(32):
        w = f"weights/unsup/lr=0.001 0 {i}.pth"
        scatter_plot(train_type="unsup", weights=w, lr=0.0001, batch=i, epoch=0, loc="unsup")


    '''
    for i in range(32):
            w = f"weights/sup/sup_net_weights_ lr=0.001 0 {i}.pth"
            scatter_plot(train_type="sup", weights=w, lr=0.001, batch=i, epoch=0, loc="sup")
