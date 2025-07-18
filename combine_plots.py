import os

from PIL import Image

def merge(im1, im2, title, loc):
    images = [Image.open(x) for x in [im1,im2]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(f"{loc}/"+title+" Combined.png")

if __name__ == "__main__":

    merge(f"plots/blue-green/Lr=0.001 0.png", f"plots/scatter_plots/no_training.png", f"no_train",
          "plots/combined")

    for i in range(32):
        merge(f"plots/blue-green/Lr=0.001 {i+1}.png", f"plots/scatter_plots/unsup lr = 0.0001, 0 {i}.png", f"unsup {i}", "plots/combined")


    for i in range(32):
        merge(f"plots/blue-green/Lr=0.001 {i+33}.png", f"plots/scatter_plots/sup lr = 0.001, 0 {i}.png", f"z sup {i}", "plots/combined")
