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

    merge("", "", "", "")