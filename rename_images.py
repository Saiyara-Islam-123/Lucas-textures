import os

import shutil
import pandas as pd

def rename():
    folders = os.listdir("StimuliTextures")
    count = 0
    labels = []
    image_indices = []

    for folder in folders:
        images = os.listdir("StimuliTextures/" + folder)
        for image in images:
            shutil.copyfile(f"StimuliTextures/{folder}/{image}", f"images/{count}.png")
            image_indices.append(count)
            if "kalamite" in image:
                labels.append(1)
            else:
                labels.append(0)

            print(count, image)
            count += 1

    df = pd.DataFrame()
    df["index"] = image_indices
    df["label"] = labels
    df.to_csv("dataset.csv", index=False)

