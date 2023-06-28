import numpy as np
from scipy.optimize import linear_sum_assignment
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes
import os
import pickle
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import cv2


class Evaluate(object):
    def __init__(self, predicted_masks, gt_masks):
        #iou_matrix = self.compute_iou_matrix(predicted_masks, gt_masks)
        #matches = self.match_instances(iou_matrix)
        #AP, AR = 
        #return AP, AR
        _, _, pred_N = predicted_masks.shape
        _, _, gt_N = gt_masks.shape
        
        self.predicted_masks = []
        self.gt_masks = []
        for i in range(pred_N):
            self.predicted_masks.append(predicted_masks[:,:,i])
        
        for i in range(gt_N):
            self.gt_masks.append(gt_masks[:,:,i])
            
        self.iou_matrix = self._compute_iou_matrix()
        self.matches = self._match_instances(self.iou_matrix) # matches[pred_idx] = gt_idx
        
        
    
    def _calculate_iou_mask(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou
        
    def _compute_iou_matrix(self):
        num_pred = len(self.predicted_masks)
        num_gt = len(self.gt_masks)

        iou_matrix = np.zeros((num_pred, num_gt))

        for i in range(num_pred):
            for j in range(num_gt):
                iou_matrix[i, j] = self._calculate_iou_mask(self.predicted_masks[i], self.gt_masks[j])

        return iou_matrix

    
    def _match_instances(self, iou_matrix):
        num_pred, num_gt = iou_matrix.shape

        # Convert IoU matrix to a dissimilarity matrix
        dissimilarity_matrix = 1 - iou_matrix

        # Solve the matching problem using the Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(dissimilarity_matrix)

        # Create an array to store the matches
        matches = np.zeros(num_pred, dtype=np.int32)

        # Assign matches based on the optimal assignment
        for pred_idx, gt_idx in zip(row_indices, col_indices):
            matches[pred_idx] = gt_idx

        return matches
    
    def compute_precision_recall(self, iou_threshold):
        _, num_gt_instances = self.iou_matrix.shape
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_idx, gt_idx in enumerate(self.matches):
            if gt_idx == 0:  # No match for predicted instance
                false_positives += 1
            else:
                iou = self.iou_matrix[pred_idx, gt_idx] #self.calculate_iou_mask(self.predicted_masks[pred_idx], self.gt_masks[gt_idx])
                if iou >= iou_threshold:
                    true_positives += 1
                else:
                    false_positives += 1

        false_negatives = num_gt_instances - true_positives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        return precision, recall
    
    def compute_average_precision_recall(self, iou_thresholds):
        num_thresholds = len(iou_thresholds)

        average_precision = 0.0
        average_recall = 0.0

        for iou_threshold in iou_thresholds:
            precision, recall = self.compute_precision_recall(iou_threshold)
            average_precision += precision
            average_recall += recall

        average_precision /= num_thresholds
        average_recall /= num_thresholds

        return average_precision, average_recall
    
    
    def eval_prediction_masks(self, iou_threshold):
        _, num_gt_instances = self.iou_matrix.shape
        gt_instance_ids = list(range(num_gt_instances))
        
        pred_evals = []
        
        true_positive_pred = []
        false_positive_pred = []
        false_negative_pred = []
        

        for pred_idx, gt_idx in enumerate(self.matches):
            if gt_idx == 0:  # No match for predicted instance
                pred_evals.append(-1)
                false_positive_pred.append(self.predicted_masks[pred_idx])
            else:
                iou = self.iou_matrix[pred_idx, gt_idx] #self.calculate_iou_mask(self.predicted_masks[pred_idx], self.gt_masks[gt_idx])
                if iou >= iou_threshold:
                    pred_evals.append(gt_idx)
                    true_positive_pred.append(self.predicted_masks[pred_idx])
                    if gt_idx in gt_instance_ids:
                        gt_instance_ids.remove(gt_idx)
                    
                else:
                    pred_evals.append(-1)
                    false_positive_pred.append(self.predicted_masks[pred_idx])

        true_positives = self._convert_mask_list_to_ndarray_masks(true_positive_pred)
        false_positives = self._convert_mask_list_to_ndarray_masks(false_positive_pred)
        
        for gt_idx in gt_instance_ids:
            false_negative_pred.append(self.gt_masks[gt_idx])
        false_negatives = self._convert_mask_list_to_ndarray_masks(false_negative_pred)
        
        return pred_evals, true_positives, false_positives, false_negatives 
    
    def _convert_mask_list_to_ndarray_masks(self, mask_list):
        output_height, output_width = mask_list[0].shape
        masks = np.zeros((output_height, output_width, len(mask_list)), dtype=np.uint8)
        for i,d in enumerate(mask_list):
            masks[:, :, i] = d
        return masks>0
    
    
def shp_to_masks(shp_file, tif_file):
    assert os.path.exists(shp_file)
    assert os.path.exists(tif_file)

    # Load the shapefile using GeoPandas
    gdf = gpd.read_file(shp_file)

    # Remove invalid or empty geometries
    gdf = gdf.dropna(subset=['geometry'])

    if gdf.empty:
        raise ValueError('No valid geometries found for rasterization.')

    # Load the TIFF file using rasterio
    tiff_dataset = rasterio.open(tif_file)
    
    # Get the necessary information from the TIFF file
    output_width = tiff_dataset.width
    output_height = tiff_dataset.height
    output_transform = tiff_dataset.transform

    # Create an empty raster array with individual channels
    raster_array = np.zeros((output_height, output_width, len(gdf)), dtype=np.uint8)
    
    # Rasterize the valid polygons with different integer values
    for idx, geometry in enumerate(gdf['geometry']):
        value = idx + 1  # Assign unique integer value to each polygon
        mask = rasterize([(geometry, value)], out_shape=raster_array.shape[:2],
                         transform=output_transform, fill=0)
        raster_array[:, :, idx] = mask

    return raster_array>0

def pkl_to_masks(pkl_file):
    with open(pkl_file, 'rb') as f:
        # Load the contents of the pickle file
        data = pickle.load(f)
    
    output_height, output_width = data[0]['segmentation'].shape
    masks = np.zeros((output_height, output_width, len(data)), dtype=np.uint8)
    for i,d in enumerate(data):
        masks[:, :, i] = d['segmentation']
    return masks>0

def plot_masks(masks):
    output_height, output_width, N = masks.shape
    anns = []
    for i in range(N):
        mask = masks[:,:,i]
        ann = {'segmentation':mask, 'area': np.count_nonzero(mask)}
        anns.append(ann)

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    plt.imshow(img)
    plt.show()


    
def masks_to_shapefile(mask_array, tiff_file, output_shapefile):
    # Convert mask array to polygons
    polygons = []
    _, _, N = mask_array.shape
    
    # Read the TIFF file to get the geospatial information
    tiff_dataset = rasterio.open(tiff_file)
    crs = tiff_dataset.crs  # Get the CRS
    
    h_start, _, _, v_start = tiff_dataset.bounds
    mask_h_size, mask_v_size = tiff_dataset.res
    
    for i in range(N):
        mask = mask_array[:,:,i]
        # Get the contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to shapely polygons
        for contour in contours:
            contour = contour.squeeze()  # Remove unnecessary dimensions
            if len(contour.shape) < 2:
                continue
            if contour.shape[0] < 3:
                continue
            
            coords = [(h_start + pixel[0]*mask_h_size, 
                       v_start - pixel[1]*mask_v_size) 
                      for pixel in contour]
            coords = np.asarray(coords)
            poly = Polygon(zip(coords[:, 0].tolist(), coords[:, 1].tolist()))
            
            polygons.append(poly)

    # Create a GeoDataFrame with the polygons
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    # Reproject the GeoDataFrame to match the CRS of the TIFF file
    gdf = gdf.to_crs(crs)

    # Save the GeoDataFrame as a shapefile
    gdf.to_file(output_shapefile)



def batch_evaluation(folder):
    pass

if __name__ == "__main__":
    shp_file = "../data/data/sample_data/DTM_slopeshade_1.shp"
    tif_file = "../data/data/sample_data/DTM_slopeshade_1.tif"
    pkl_file = "../data/data/sample_data/DTM_slopeshade_1_grayscale.pkl"
    gt_masks = shp_to_masks(shp_file, tif_file)
    pred_masks = pkl_to_masks(pkl_file)
    #plot_masks(gt_masks)
    #plot_masks(pred_masks)
    
    eval = Evaluate(pred_masks, gt_masks)
    
    print(eval.compute_precision_recall(0.5))
    print(eval.compute_average_precision_recall([0.5, 0.6, 0.7, 0.8, 0.9]))
    
    pred_evals, true_positives, false_positives, false_negatives = eval.eval_prediction_masks(0.5)
    
    plot_masks(true_positives)
    
    masks_to_shapefile(true_positives, tif_file, "../data/data/true_positives.shp")
    masks_to_shapefile(false_positives, tif_file, "../data/data/false_positives.shp")
    masks_to_shapefile(false_negatives, tif_file, "../data/data/false_negatives.shp")
    
    
    



