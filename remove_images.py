import pandas as pd
from PIL import Image, UnidentifiedImageError


def filter_dataset():
    df = pd.read_csv("dataset.csv")
    indices_to_keep = []
    labels_to_keep = []
    for i in range(len(df["index"])):
        img_path = "images/"+ str(df["index"].iloc[i])+".png"
        try:
            image = Image.open(img_path).convert('RGB')
            indices_to_keep.append(df["index"].iloc[i])
            labels_to_keep.append(df["label"].iloc[i])
            print(i)

        except UnidentifiedImageError:
            continue


    df_filtered = pd.DataFrame()
    df_filtered["index"] = indices_to_keep
    df_filtered["label"] = labels_to_keep
    df_filtered.to_csv("filtered.csv", index=False)

filter_dataset()