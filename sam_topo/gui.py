import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib as mpl
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import os
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons


class SAM_Topo_GUI(object):
    def __init__(self, img_path, sam_checkpoint, model_type, is_cuda):
        assert os.path.exists(img_path), "The image file does not exist."
        assert os.path.exists(sam_checkpoint), "The model file does not exist."
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
        if is_cuda:
            device = "cuda"
        else:
            device = "cpu"

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        
        
        # create a matplotlib interactive figure
        slider_bottom = 0.2
        
        fig_w = 10 #* (self.img.shape[1] / max(self.img.shape))
        fig_h = 10 #* (self.img.shape[0] / max(self.img.shape))
        self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h))
        self.fig.suptitle(f'SAM_TOPO GUI', fontsize=16)
        self.fig.subplots_adjust(bottom=slider_bottom+0.2)
        self.im = self.ax.imshow(self.img)
        self.ax.autoscale(False)
        
        
        # 1. create a slider for each parameter input, three parameters
        
        self.points_per_side = 32
        ax_points_per_side= plt.axes([0.2, slider_bottom, 0.5, 0.03])  # [left, bottom, width, height]
        self.slider_points_per_side = Slider(ax_points_per_side, 'points_per_side', 16, 64, valinit=self.points_per_side, valstep=1)
        self.slider_points_per_side.on_changed(self._update_points_per_side)
        
        self.pred_iou_thresh = 0.86
        ax_pred_iou_thresh = plt.axes([0.2, slider_bottom+0.05, 0.5, 0.03])  # [left, bottom, width, height]
        self.slider_pred_iou_thresh = Slider(ax_pred_iou_thresh, 'pred_iou_thresh', 0.1, 1.0, valinit=self.pred_iou_thresh)
        self.slider_pred_iou_thresh.on_changed(self._update_pred_iou_thresh)
        
        self.stability_score_thresh = 0.85
        ax_stability_score_thresh = plt.axes([0.2, slider_bottom+0.1, 0.5, 0.03])  # [left, bottom, width, height]
        self.slider_stability_score_thresh = Slider(ax_stability_score_thresh, 'stability_score_thresh', 0.1, 1.0, valinit=self.stability_score_thresh)
        self.slider_stability_score_thresh.on_changed(self._update_stability_score_thresh)
        
        # 2. create a button for automatic segmentation generation
        ax_button_sam = plt.axes([0.3, 0.05, 0.2, 0.05])  # [left, bottom, width, height]
        self.button_sam = Button(ax_button_sam, 'Segment Everything')
        self.button_sam.on_clicked(self._push_button_sam)
        
        # 3. add a status text box        
        ax_status_text = plt.axes([0.3, 0.12, 0.5, 0.03])  # [left, bottom, width, height]
        self.status_text = TextBox(ax_status_text, '', initial='Wait after click "Segment Everything"', color='white', hovercolor='white')

        
        
        
        # ==== once the automatic segmentation generation botton is pushed, ...
        # create a SAM predictor
        # create a new interactive figure
        # 3. create radio buttons for the mask display 
        # 4. create a button to save annotations
        # 5. create a button to add an annotation
        # 6. create a button to delete an annotation
        # 7. create a button to modify an annotation
        # 9. create two radio buttons to include and exclude points respectively 
        
        plt.show()
        
    def _push_button_sam_confirm(self, _):
        pass
        
        
    def _push_button_sam(self, _):
        self.mask_generator = SamAutomaticMaskGenerator(
        model=self.sam,
        points_per_side=self.points_per_side,
        pred_iou_thresh=self.pred_iou_thresh,
        stability_score_thresh=self.stability_score_thresh)
        
        self.masks = self.mask_generator.generate(self.img)
        print(len(self.masks))
        
        # create a visualize or not radio button
        ax_visualize_radio = plt.axes([0.1, 0.05, 0.1, 0.1])  # [left, bottom, width, height]
        self.radio_buttons_visualize = RadioButtons(ax_visualize_radio, ['orignal', 'masks'], active=0,)
        self.radio_buttons_visualize.on_clicked(self._on_radio_button_visualize_clicked)
        
        # create a confirm button
        self.status_text.set_val("Ready to confirm results")
        ax_button_sam_confirm = plt.axes([0.55, 0.05, 0.2, 0.05])  # [left, bottom, width, height]
        self.button_sam_confirm = Button(ax_button_sam_confirm, 'Confirm')
        self.button_sam_confirm.on_clicked(self._push_button_sam_confirm)
        self.fig.canvas.draw_idle()
    
    def _on_radio_button_visualize_clicked(self, label):
        if label=="masks":
            self.show_anns(self.masks)
            
        else:
            self.im = self.ax.imshow(self.img)
            self.fig.canvas.draw_idle()
    
    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        #ax = plt.gca()
        self.ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()
        
    
    def _update_points_per_side(self, val):
        self.points_per_side = self.slider_points_per_side.val
        #print(self.points_per_side)
        
    def _update_pred_iou_thresh(self, val):
        self.pred_iou_thresh = self.slider_pred_iou_thresh.val
        #print(self.pred_iou_thresh)
        
    def _update_stability_score_thresh(self, val):
        self.stability_score_thresh = self.slider_stability_score_thresh.val
        #print(self.stability_score_thresh)
    
    
    
if __name__ == "__main__":
    sam_topo_gui = SAM_Topo_GUI('../data/data/sample_data/beach_hillshade_grayscale.png', 
                                sam_checkpoint = "../data/models/sam_vit_h_4b8939.pth", 
                                model_type = "vit_h", 
                                is_cuda=True)