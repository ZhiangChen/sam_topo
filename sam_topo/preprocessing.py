from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def tif_to_png(tif_path, png_path):
    grayscale_image = Image.open(tif_path)
    grayscale_array = np.array(grayscale_image)    
    grayscale_array[grayscale_array<-9000] = np.mean(grayscale_array[grayscale_array>-9000])
    array_min = np.min(grayscale_array)
    array_max = np.max(grayscale_array)
    grayscale_array = (grayscale_array - array_min) / (array_max - array_min)*255
    image = Image.fromarray(grayscale_array.astype(np.uint8))
    image.save(png_path, "PNG")

def tif_to_colormap(tif_path, png_path, colormap_name="rainbow"):    
    grayscale_image = Image.open(tif_path)
    grayscale_array = np.array(grayscale_image)    
    grayscale_array[grayscale_array<-9000] = np.mean(grayscale_array[grayscale_array>-9000])
    array_min = np.min(grayscale_array)
    array_max = np.max(grayscale_array)
    grayscale_array = (grayscale_array - array_min) / (array_max - array_min)
    colormap_image = plt.get_cmap(colormap_name)(grayscale_array)
    colormap_image = Image.fromarray((colormap_image * 255).astype(np.uint8))
    colormap_image.save(png_path)
        
def tif_to_png_batch(folder_path):
    assert os.path.exists(folder_path)
    tif_files = [file for file in os.listdir(folder_path) if file.endswith('tif')]
    for tif_file in tif_files:
        tif_path = os.path.join(folder_path, tif_file)
        png_path = tif_path[:-4]+"_grayscale.png"
        tif_to_png(tif_path, png_path)
        
def tif_to_colormap_batch(folder_path, colormap_name="rainbow"):
    assert os.path.exists(folder_path)
    tif_files = [file for file in os.listdir(folder_path) if file.endswith('tif')]
    for tif_file in tif_files:
        tif_path = os.path.join(folder_path, tif_file)
        png_path = tif_path[:-4]+"_color.png"
        tif_to_colormap(tif_path, png_path, colormap_name)
        
if __name__ == "__main__":
    tif_to_png_batch('/home/shakebot/sam_topo/data/data/sample_data')
    tif_to_colormap_batch('/home/shakebot/sam_topo/data/data/sample_data')