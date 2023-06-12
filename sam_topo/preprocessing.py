from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def tif_to_png(tif_path, png_path):
    image = Image.open(tif_path)
    # Convert to RGB mode
    image = image.convert("RGB")
    image.save(png_path, "PNG")


def tif_to_colormap(tif_path, png_path, colormap_name="rainbow"):
    # Open the grayscale image
    grayscale_image = Image.open(tif_path)
    # Convert the image to a NumPy array
    grayscale_array = np.array(grayscale_image)
    # Apply the colormap to the grayscale array
    colormap_image = plt.get_cmap(colormap_name)(grayscale_array)
    # Convert the colormap image to PIL Image
    colormap_image = Image.fromarray((colormap_image * 255).astype(np.uint8))
    # Save the colormap image
    colormap_image.save(png_path)
    
    
