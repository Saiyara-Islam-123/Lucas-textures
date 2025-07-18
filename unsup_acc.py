import pandas as pd
from NeuralNetwork import *
from dataset import *

def acc():
    accs = []
    file = "filtered.csv"
    trainloader,_,_ = load_texture_data(file, batch_size=64)

    for i in range(32):
        network = Net()
        w = f"weights/unsup/lr=0.001 0 {i}.pth"
        network.load_state_dict(torch.load(w))
        sup_network = SupervisedNet(network)
        for images, labels in trainloader:
            outputs = sup_network(images)
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            accs.append(accuracy)
            break
        print(i)
    df = pd.DataFrame()
    df["accuracy"] = accs

    df.to_csv("unsup_accs")

acc()