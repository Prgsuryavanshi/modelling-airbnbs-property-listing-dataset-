import os
import glob
import imutils
from PIL import Image
import numpy as np


def open_image(image_path: str) -> np.ndarray:
    """ This function takes the path of the images and open as numpy array

    Args:
        image_path (str): Path of the images

    Returns:
        image (np.ndarray): Returns the array of the image read
    """
        
    image = Image.open(image_path, 'r')
    image = np.asarray(image)
    
    return image


def save_images(image_path: str, image: np.ndarray):
    """This function is used to:
       -> Save the processed image in the data/processed_image folder

    Args:
        image_path (str): path of the images in RGB format
        image (np.ndarray): array of the resized image
    """

    processed_image_path = os.path.join('data','processed_images', os.path.basename(image_path))
    image = Image.fromarray(image)
    image.save(processed_image_path)

    
def resize_images(image_data_folder: str):
    """ This function is used to:
        -> Check that every image is in RGB format.
        -> Set the height of the smallest image as the height for all of the other images.
        -> Resize the images to maintain the aspect ratio of the image and adjust the width 
        proportionally to the change in height.

    Args:
        image_data_folder (str): Path of the image folder
    """
    
    images_height = list()
    path_of_images = list()
    images = os.path.join(image_data_folder, '**', '*.png')
    for image_path in glob.glob(images, recursive=True):
        image = open_image(image_path)
        if len(image.shape) == 3:
            images_height.append(image.shape[0])
            path_of_images.append(image_path)
            
    min_image_height = min(images_height)
    
    for image_path in path_of_images:
        image = open_image(image_path)
        image = imutils.resize(image, height=min_image_height)
        save_images(image_path,image)


if __name__ == "__main__":
    image_data_folder = 'data/images'
    resize_images(image_data_folder)
