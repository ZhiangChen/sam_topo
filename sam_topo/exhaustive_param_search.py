from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os, cv2, pickle
import numpy as np
import matplotlib.pyplot as plt
from sam_topo.evaluation import shp_to_masks, Evaluate, pkl_to_masks, masks_to_shapefile

def form_string_scores(points_per_side, pred_iou_thresh, stability_score_thresh):
    a = str(points_per_side).split('.')[0].zfill(4)
    b = str(pred_iou_thresh*100).split('.')[0].zfill(4)
    c = str(stability_score_thresh*100).split('.')[0].zfill(4)
    return "_".join([a, b, c]) 

def form_pickle_file_by_scores(img_file, string_scores):
    return "_".join([img_file[:-4], string_scores]) +'.pkl'

def range_score(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
        
    results = []

    current = start
    while current < stop:
        results.append(current)
        current += step
        
    return results

class Exhaustive_Param_Search(object):
    def __init__(self, img_path, shp_path, tif_path, sam_checkpoint, model_type, is_cuda):
        assert os.path.exists(img_path), "The image file does not exist."
        assert os.path.exists(sam_checkpoint), "The model file does not exist."
        assert os.path.exists(shp_path), "The shapefile does not exist."
        assert os.path.exists(tif_path), "The tiff file does not exist."
        
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.shp_path = shp_path
        self.tif_path = tif_path
        
        if is_cuda:
            device = "cuda"
        else:
            device = "cpu"

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        
        
    def start_search(self, param_ranges):
        gt_masks = self._extract_gt_masks()
        
        precisions = {}
        recalls = {}
        
        for points_per_side in param_ranges["points_per_sides"]:
            for pred_iou_thresh in param_ranges["pred_iou_threshs"]:
                for stability_score_thresh in param_ranges["stability_score_threshs"]:
                    pred_results = self._segment_everything(points_per_side, pred_iou_thresh, stability_score_thresh)
                    
                    a = str(points_per_side).split('.')[0].zfill(4)
                    b = str(pred_iou_thresh*100).split('.')[0].zfill(4)
                    c = str(stability_score_thresh*100).split('.')[0].zfill(4)
                    save_path = "_".join([self.img_path[:-4], a, b, c]) +'.pkl'
                    
                    with open(save_path, 'wb') as f:
                        pickle.dump(pred_results, f)
                    pred_masks = self._extract_pred_masks(pred_results)
                    eval = Evaluate(pred_masks, gt_masks)
                    precision, recall = eval.compute_precision_recall(0.5)
                    print("points_per_side ({}), pred_iou_thresh ({}), and stability_score_thresh ({}): precision ({}) and recall ({})".format(points_per_side, pred_iou_thresh, stability_score_thresh, precision, recall))
                    
                    key = "_".join([a, b, c])
                    precisions[key] = precision
                    recalls[key] = recall
         
                    precisions_save_path = self.img_path[:-4]+'_precisions.pkl'
                    recalls_save_path = self.img_path[:-4]+'_recalls.pkl'
                                
                    with open(precisions_save_path, 'wb') as f:
                        pickle.dump(precisions, f)    
                        
                    with open(recalls_save_path, 'wb') as f:
                        pickle.dump(recalls, f)
                    

        
    def _segment_everything(self, points_per_side, pred_iou_thresh, stability_score_thresh):
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh)
        
        results = mask_generator.generate(self.img)   
        return results
    
    def _extract_pred_masks(self, results):
        output_height, output_width = results[0]['segmentation'].shape
        masks = np.zeros((output_height, output_width, len(results)), dtype=np.uint8)
        for i,d in enumerate(results):
            masks[:, :, i] = d['segmentation']
        return masks>0
    
    def _extract_gt_masks(self,):
        return shp_to_masks(self.shp_path, self.tif_path)
    
    
def extract_best_params(precisions_f, recalls_f):
    with open(precisions_f, 'rb') as file:
        precisions = pickle.load(file)
    with open(recalls_f, 'rb') as file:
        recalls = pickle.load(file)
    
    M_precision = -1
    M_recall = -1
    best_param = ""
    for key in precisions:
        precision = precisions[key]
        recall = recalls[key]
        if precision <= recall:
            if M_precision < precision:
                M_precision = precision
                M_recall = recall
                best_param = key
            elif M_precision == precision:
                if M_recall < recall:
                    M_precision = precision
                    M_recall = recall
                    best_param = key
                else:
                    None
            else:
                None
        else:
            None
    
    if len(best_param) != 0:
        print("precision: ", M_precision)
        print("recall: ", M_recall)
        
        params = [int(a) for a in best_param.split('_')]
        points_per_side = params[0]
        pred_iou_thresh = params[1]/100.
        stability_score_thresh = params[2]/100.
    
    return points_per_side, pred_iou_thresh, stability_score_thresh


def save_score_plot(points_per_side, 
                    pred_iou_thresh, 
                    stability_score_thresh, 
                    axis, 
                    ranges, 
                    values, 
                    x_label, 
                    y_label,
                    save_dir):
    assert axis in [0, 1, 2]
    assert os.path.exists(save_dir)
    
    scores_ndarray = np.asarray([[points_per_side, pred_iou_thresh, stability_score_thresh]])
    scores_ndarray = np.repeat(scores_ndarray, len(ranges), axis=0)
    scores_ndarray[:, axis] = ranges
    
    score_values = []
    for i in range(len(ranges)):
        scores = scores_ndarray[i, :].tolist()
        string_scores = form_string_scores(scores[0], scores[1], scores[2])
        score_values.append(values[string_scores])
    
    plt.plot(ranges, score_values)
    
    string_scores = form_string_scores(points_per_side, pred_iou_thresh, stability_score_thresh)
    best_value = values[string_scores]
    if axis==0:
        plt.scatter(points_per_side, best_value, marker='*', s=200, c='red')
    elif axis==1:
        plt.scatter(pred_iou_thresh, best_value, marker='*', s=200, c='red')
    else:
        plt.scatter(stability_score_thresh, best_value, marker='*', s=200, c='red')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(save_dir, "_".join([x_label, y_label]) + ".png"))
    plt.clf()
        
        


if __name__ == "__main__":
    
    param_ranges = {"points_per_sides": range_score(8, 32, 1),
                    "pred_iou_threshs": range_score(0.8, 0.98, 0.02), 
                    "stability_score_threshs": range_score(0.8, 0.98, 0.02) 
    }
    
    
    PARAM_SEARCH = False
    if PARAM_SEARCH:
        eps = Exhaustive_Param_Search(img_path="../data/data/validation_1/DTM_1_grayscale.png", 
                                    shp_path="../data/data/validation_1/DTM_1.shp",
                                    tif_path="../data/data/validation_1/DTM_1.tif",
                                    sam_checkpoint="../data/models/sam_vit_h_4b8939.pth",
                                    model_type="vit_h",
                                    is_cuda=True,
                                    )
        
        
        print(param_ranges)
        
        eps.start_search(param_ranges)
        
        eps = Exhaustive_Param_Search(img_path="../data/data/validation_4/DTM_4_grayscale.png", 
                                    shp_path="../data/data/validation_4/DTM_4.shp",
                                    tif_path="../data/data/validation_4/DTM_4.tif",
                                    sam_checkpoint="../data/models/sam_vit_h_4b8939.pth",
                                    model_type="vit_h",
                                    is_cuda=True,
                                    )
        
        eps.start_search(param_ranges)
    
    PARAM_EXTRACT = False
    if PARAM_EXTRACT:
        img_file = "../data/data/validation_1/DTM_1_grayscale.png"
        tif_file = "../data/data/validation_1/DTM_1.tif"
        shp_file = "../data/data/validation_1/DTM_1.shp"
        precisions_f = "../data/data/validation_1/DTM_1_grayscale_precisions.pkl"
        recalls_f = "../data/data/validation_1/DTM_1_grayscale_recalls.pkl"
        points_per_side, pred_iou_thresh, stability_score_thresh = extract_best_params(precisions_f, recalls_f)
        print(points_per_side, pred_iou_thresh, stability_score_thresh)
        string_scores = form_string_scores(points_per_side, pred_iou_thresh, stability_score_thresh)
        pkl_file = form_pickle_file_by_scores(img_file, string_scores)
        gt_masks = shp_to_masks(shp_file, tif_file)
        pred_masks = pkl_to_masks(pkl_file)        
        eval = Evaluate(pred_masks, gt_masks)
        pred_evals, true_positives, false_positives, false_negatives = eval.eval_prediction_masks(0.5)
        masks_to_shapefile(true_positives, tif_file, "../data/data/validation_1/true_positives.shp")
        masks_to_shapefile(false_positives, tif_file, "../data/data/validation_1/false_positives.shp")
        masks_to_shapefile(false_negatives, tif_file, "../data/data/validation_1/false_negatives.shp")
        
        img_file = "../data/data/validation_4/DTM_4_grayscale.png"
        tif_file = "../data/data/validation_4/DTM_4.tif"
        shp_file = "../data/data/validation_4/DTM_4.shp"
        precisions_f = "../data/data/validation_4/DTM_4_grayscale_precisions.pkl"
        recalls_f = "../data/data/validation_4/DTM_4_grayscale_recalls.pkl"
        points_per_side, pred_iou_thresh, stability_score_thresh = extract_best_params(precisions_f, recalls_f)
        print(points_per_side, pred_iou_thresh, stability_score_thresh)
        string_scores = form_string_scores(points_per_side, pred_iou_thresh, stability_score_thresh)
        pkl_file = form_pickle_file_by_scores(img_file, string_scores)
        gt_masks = shp_to_masks(shp_file, tif_file)
        pred_masks = pkl_to_masks(pkl_file)        
        eval = Evaluate(pred_masks, gt_masks)
        pred_evals, true_positives, false_positives, false_negatives = eval.eval_prediction_masks(0.5)
        masks_to_shapefile(true_positives, tif_file, "../data/data/validation_4/true_positives.shp")
        masks_to_shapefile(false_positives, tif_file, "../data/data/validation_4/false_positives.shp")
        masks_to_shapefile(false_negatives, tif_file, "../data/data/validation_4/false_negatives.shp")
    
    
    PARAM_ANALYSIS = True
    if PARAM_ANALYSIS:
        img_file = "../data/data/validation_1/DTM_1_grayscale.png"
        tif_file = "../data/data/validation_1/DTM_1.tif"
        shp_file = "../data/data/validation_1/DTM_1.shp"
        precisions_f = "../data/data/validation_1/DTM_1_grayscale_precisions.pkl"
        recalls_f = "../data/data/validation_1/DTM_1_grayscale_recalls.pkl"
        points_per_side, pred_iou_thresh, stability_score_thresh = extract_best_params(precisions_f, recalls_f)
        with open(precisions_f, 'rb') as file:
            precisions = pickle.load(file)
        with open(recalls_f, 'rb') as file:
            recalls = pickle.load(file)
        
        # precision
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=0, 
                   ranges=param_ranges["points_per_sides"], 
                   values=precisions, 
                   x_label="points_per_sides", 
                   y_label="precision",
                   save_dir="../data/data/validation_1/")
        
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=1, 
                   ranges=param_ranges["pred_iou_threshs"], 
                   values=precisions, 
                   x_label="pred_iou_thresh", 
                   y_label="precision",
                   save_dir="../data/data/validation_1/")   
        
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=2, 
                   ranges=param_ranges["stability_score_threshs"], 
                   values=precisions, 
                   x_label="stability_score_thresh", 
                   y_label="precision",
                   save_dir="../data/data/validation_1/")     
        
        # recall
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=0, 
                   ranges=param_ranges["points_per_sides"], 
                   values=recalls, 
                   x_label="points_per_sides", 
                   y_label="recall",
                   save_dir="../data/data/validation_1/")
        
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=1, 
                   ranges=param_ranges["pred_iou_threshs"], 
                   values=recalls, 
                   x_label="pred_iou_thresh", 
                   y_label="recall",
                   save_dir="../data/data/validation_1/")   
        
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=2, 
                   ranges=param_ranges["stability_score_threshs"], 
                   values=recalls, 
                   x_label="stability_score_thresh", 
                   y_label="recall",
                   save_dir="../data/data/validation_1/") 
        
        #########################################################       
        
        img_file = "../data/data/validation_4/DTM_4_grayscale.png"
        tif_file = "../data/data/validation_4/DTM_4.tif"
        shp_file = "../data/data/validation_4/DTM_4.shp"
        precisions_f = "../data/data/validation_4/DTM_4_grayscale_precisions.pkl"
        recalls_f = "../data/data/validation_4/DTM_4_grayscale_recalls.pkl"
        points_per_side, pred_iou_thresh, stability_score_thresh = extract_best_params(precisions_f, recalls_f)
        with open(precisions_f, 'rb') as file:
            precisions = pickle.load(file)
        with open(recalls_f, 'rb') as file:
            recalls = pickle.load(file)
        
        # precision
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=0, 
                   ranges=param_ranges["points_per_sides"], 
                   values=precisions, 
                   x_label="points_per_sides", 
                   y_label="precision",
                   save_dir="../data/data/validation_4/")
        
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=1, 
                   ranges=param_ranges["pred_iou_threshs"], 
                   values=precisions, 
                   x_label="pred_iou_thresh", 
                   y_label="precision",
                   save_dir="../data/data/validation_4/")   
        
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=2, 
                   ranges=param_ranges["stability_score_threshs"], 
                   values=precisions, 
                   x_label="stability_score_thresh", 
                   y_label="precision",
                   save_dir="../data/data/validation_4/")     
        
        # recall
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=0, 
                   ranges=param_ranges["points_per_sides"], 
                   values=recalls, 
                   x_label="points_per_sides", 
                   y_label="recall",
                   save_dir="../data/data/validation_4/")
        
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=1, 
                   ranges=param_ranges["pred_iou_threshs"], 
                   values=recalls, 
                   x_label="pred_iou_thresh", 
                   y_label="recall",
                   save_dir="../data/data/validation_4/")   
        
        save_score_plot(points_per_side, 
                   pred_iou_thresh, 
                   stability_score_thresh, 
                   axis=2, 
                   ranges=param_ranges["stability_score_threshs"], 
                   values=recalls, 
                   x_label="stability_score_thresh", 
                   y_label="recall",
                   save_dir="../data/data/validation_4/")
    