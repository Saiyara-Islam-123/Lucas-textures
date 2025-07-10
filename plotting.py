import matplotlib.pyplot as plt
import pandas as pd

def plot_batch(time_step, csv_unsup, csv_sup, num_unsup_rows, num_sup_rows, lr, loc):
    df_no_train = pd.read_csv("Distance no train.csv")

    df_unsup = pd.read_csv(csv_unsup)
    df_unsup = df_unsup.tail(num_unsup_rows)


    df_sup = pd.read_csv(csv_sup)
    df_sup = df_sup.head(num_sup_rows)


    accs = df_sup["acc"]
    df_sup.drop(columns=["acc"], inplace=True)

    df_whole = pd.concat([df_no_train, df_unsup , df_sup])
    x = []
    for k in range(num_unsup_rows+num_sup_rows+1):
        x.append(k)
    x2 = []
    for k in range(num_unsup_rows+1, num_unsup_rows+num_sup_rows+1):
        x2.append(k)

    fig, ax1 = plt.subplots()
    ax1.plot(x, df_whole['within 1'], color="limegreen", label="within Cat1")
    ax1.plot(x, df_whole['within 0'], color="green", label="within Cat2")
    ax1.plot(x, df_whole['between'], color="blue", label="between")
    ax1.axvline(x=num_unsup_rows, color='r', linestyle='--')
    ax1.axvline(x=1, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="coral", label="accuracy")
    ax1.axvline(x=time_step, color='black', linestyle='dashed')
    ax2.set_ylabel('Accuracy')


    plt.xlabel("Epoch")
    plt.title("Texture categorization accuracy and distances across training batches")

    plt.savefig(loc + f"/Lr={lr} " + str(time_step) + ".png")

    plt.show()

if __name__=="__main__":

    for i in range(0, 65):
        plot_batch(time_step=i, csv_unsup="LR=0.001, Distance every batch unsup.csv", csv_sup="LR=0.001, Distance every batch sup", num_unsup_rows=32, num_sup_rows=32, lr=0.001, loc="plots/blue-green/")