import imageio
from pathlib import Path
import os
from typing import List


def make_gif(images_path: Path, gif_filename: Path):
    """
    Creates a gif from a set of images. The images are located
    in images_path. Note that only images should be located in the
    given directory
    :return:
    """

    # collect all the files in the
    # proposed directory

    filenames = os.listdir(images_path)

    images = []

    for filename in filenames:
        images.append(imageio.imread(str(images_path) + "/" + filename))

    imageio.mimsave(gif_filename, images)


def make_gif_from_images(filenames: List[str], gif_filename: Path):

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))

    imageio.mimsave(gif_filename, images)