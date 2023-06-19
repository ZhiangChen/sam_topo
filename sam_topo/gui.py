import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib as mpl
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import os
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons 
import pickle

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
        self.slider_bottom = 0.2
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.suptitle(f'SAM_TOPO GUI', fontsize=16)
        self.fig.subplots_adjust(bottom=self.slider_bottom+0.2)
        self.im = self.ax.imshow(self.img)
        self.ax.autoscale(False)
        
        
        # create a slider for each parameter input, three parameters
        self.points_per_side = 32
        self.ax_points_per_side= plt.axes([0.2, self.slider_bottom, 0.5, 0.03])  # [left, bottom, width, height]
        self.slider_points_per_side = Slider(self.ax_points_per_side, 'points_per_side', 16, 64, valinit=self.points_per_side, valstep=1)
        self.slider_points_per_side.on_changed(self._update_points_per_side)
        
        self.pred_iou_thresh = 0.86
        self.ax_pred_iou_thresh = plt.axes([0.2, self.slider_bottom+0.05, 0.5, 0.03])  # [left, bottom, width, height]
        self.slider_pred_iou_thresh = Slider(self.ax_pred_iou_thresh, 'pred_iou_thresh', 0.1, 1.0, valinit=self.pred_iou_thresh)
        self.slider_pred_iou_thresh.on_changed(self._update_pred_iou_thresh)
        
        self.stability_score_thresh = 0.85
        self.ax_stability_score_thresh = plt.axes([0.2, self.slider_bottom+0.1, 0.5, 0.03])  # [left, bottom, width, height]
        self.slider_stability_score_thresh = Slider(self.ax_stability_score_thresh, 'stability_score_thresh', 0.1, 1.0, valinit=self.stability_score_thresh)
        self.slider_stability_score_thresh.on_changed(self._update_stability_score_thresh)
        
        # create a button for automatic segmentation generation
        self.ax_button_sam = plt.axes([0.3, 0.05, 0.2, 0.05])  # [left, bottom, width, height]
        self.button_sam = Button(self.ax_button_sam, 'Segment Everything')
        self.button_sam.on_clicked(self._push_button_sam)
        
        # add a status text box        
        self.ax_status_text = plt.axes([0.3, 0.12, 0.2, 0.03])  # [left, bottom, width, height]
        self.status_text = self.ax_status_text.text(0,0, 'Ready to SAM', fontsize=12, )
        self.ax_status_text.axis('off')
        
        # add a mask number text box        
        self.ax_mask_nummber = plt.axes([0.55, 0.12, 0.2, 0.03])  # [left, bottom, width, height]
        self.mask_number_text = self.ax_mask_nummber.text(0,0, '  ', fontsize=12, )
        self.ax_mask_nummber.axis('off')
        
        # add a mouse click callback function
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = [], [], [], [], []
        self.add_plot, = self.ax.plot([], [], 'o', markerfacecolor='green', markeredgecolor='black', markersize=5)
        self.rem_plot, = self.ax.plot([], [], 'x', markerfacecolor='red', markeredgecolor='red', markersize=5)
        
        # set a flag between segment everything and interactive segmentation
        self.flag_interactive_segmentation = False
        
        plt.show()
        
    def _push_button_sam_confirm(self, _):
        if self.flag_interactive_segmentation is False:
            self.status_text.set_text('Initializing ... wait')
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            
            self.flag_interactive_segmentation = True
            self.ax_points_per_side.remove()
            self.ax_pred_iou_thresh.remove()
            self.ax_stability_score_thresh.remove()
            self.ax_status_text.remove()
            self.ax_mask_nummber.remove()
            self.ax_button_sam.remove()
            
            self.predictor = SamPredictor(self.sam)
            self.predictor.set_image(self.img)
            
            # create slider for the individual mask display 
            self.mask_id = 0
            self.ax_mask_id= plt.axes([0.2, self.slider_bottom, 0.5, 0.03])  # [left, bottom, width, height]
            self.slider_mask_id = Slider(self.ax_mask_id, 'mask_id', 0, len(self.masks)-1, valinit=self.mask_id, valstep=1)
            self.slider_mask_id.on_changed(self._update_mask_id)
            
            # create a button to add a mask
            self.ax_button_add = plt.axes([0.3, 0.12, 0.2, 0.05])  # [left, bottom, width, height]
            self.button_add = Button(self.ax_button_add, 'Add')
            self.button_add.on_clicked(self._push_button_add)
            
            
            # create a button to delete a mask
            self.ax_button_delete = plt.axes([0.3, 0.05, 0.2, 0.05])  # [left, bottom, width, height]
            self.button_delete = Button(self.ax_button_delete, 'Delete')
            self.button_delete.on_clicked(self._push_button_delete)
            
            # create a button to modify an annotation
            self.ax_button_modify = plt.axes([0.55, 0.12, 0.2, 0.05])  # [left, bottom, width, height]
            self.button_modify = Button(self.ax_button_modify, 'Modify')
            self.button_modify.on_clicked(self._push_button_modify)
            
            # save annotation button
            self.button_sam_confirm.label.set_text("Save Annotations") 
            
            # annotation guide text
            self.ax_annotation_guide_text = plt.axes([0.2, self.slider_bottom + 0.05, 0.2, 0.03])  # [left, bottom, width, height]
            self.annotation_guide_text= self.ax_annotation_guide_text.text(0,0, 'left click to include points, right click to exclude, press c to clear', fontsize=12, )
            self.ax_annotation_guide_text.axis('off')
            
            # status text
            self.ax_annotation_status = plt.axes([0.2, self.slider_bottom + 0.1, 0.2, 0.03])  # [left, bottom, width, height]
            self.annotation_status_text = self.ax_annotation_status.text(0,0, '  ', fontsize=12, )
            self.ax_annotation_status.axis('off')
            
            
            # key press 
            self.fig.canvas.mpl_connect('key_press_event', self._on_key)
            
            self.fig.canvas.draw_idle()
            
            
        else:
            # save the masks    
            self.annotation_status_text.set_text('Saving ... ')
            self.annotation_status_text.set_color('black')
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
                
            save_path = self.img_path[:-3]+'pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(self.masks, f)
                
            self.annotation_status_text.set_text('Saved to path ' + save_path)
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            
            
        
    def _push_button_add(self, _):
        if len(self.add_xs)>0:
            mask, score, _ = self.predictor.predict(point_coords=np.array(list(zip(self.add_xs, self.add_ys)) + 
                                                                    list(zip(self.rem_xs, self.rem_ys))), 
                                                point_labels=np.array([1] * len(self.add_xs) + [0] * len(self.rem_xs)), 
                                                multimask_output=False)
            
            
            add_mask = [{'segmentation': mask[0], 'area': np.count_nonzero(mask), 'predicted_iou':score, 'stability_score':1.0, 'bbox':None, 'crop_box':None}]
            self.masks += add_mask
            
            # update slider
            self.slider_mask_id.valmax = len(self.masks)-1 
            self.slider_mask_id.ax.set_xlim(0, len(self.masks)-1)
            self.slider_mask_id.valinit = len(self.masks)-1 
            self.mask_id = len(self.masks)-1
            self.slider_mask_id.reset() 
            self.im = self.ax.imshow(self.img)
            self._show_anns([self.masks[self.mask_id]])
        else:
            self.annotation_status_text.set_text('Click points to add a mask')
            self.annotation_status_text.set_color('red')
            
            self.fig.canvas.draw_idle()
    
        
    
    def _push_button_delete(self, _):
        if len(self.masks)>0:
            del self.masks[self.mask_id]
            self.slider_mask_id.valmax = len(self.masks)-1 
            self.slider_mask_id.ax.set_xlim(0, len(self.masks)-1)
            
            if len(self.masks)>0: # only update slider when there is still any mask
                if self.mask_id == len(self.masks): # the slider is at the right end
                    self.mask_id -=1
                    
                self.slider_mask_id.valinit = self.mask_id
                self.slider_mask_id.reset() 
                
                self.im = self.ax.imshow(self.img)
                self._show_anns([self.masks[self.mask_id]])
            
            self.fig.canvas.draw_idle()
    
    
    def _push_button_modify(self, _):
        if len(self.add_xs)>0 or len(self.rem_xs)>0:
            mask, score, _ = self.predictor.predict(point_coords=np.array(list(zip(self.add_xs, self.add_ys)) + 
                                                                    list(zip(self.rem_xs, self.rem_ys))), 
                                                point_labels=np.array([1] * len(self.add_xs) + [0] * len(self.rem_xs)), 
                                                mask_input=self.masks[self.mask_id]['segmentation'][None, :, :].resize((1, 256, 256)),
                                                multimask_output=False)
            
            new_mask = [{'segmentation': mask, 'area': np.count_nonzero(mask), 'predicted_iou':score, 'stability_score':1.0, 'bbox':None, 'crop_box':None}]
            self.masks[self.mask_id] = new_mask
            
            # update slider
            self.im = self.ax.imshow(self.img)
            self._show_anns([self.masks[self.mask_id]])
            
        else:
            self.annotation_status_text.set_text('Click points to modify')
            self.annotation_status_text.set_color('red')
            
            self.fig.canvas.draw_idle()
    
    
    def _on_click(self, event):
        if event.inaxes != self.ax and (event.button in [1, 3]): return
        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))

        if event.button == 1: # left click
            self.trace.append(True)
            self.add_xs.append(x)
            self.add_ys.append(y)
            self._show_points(self.add_plot, self.add_xs, self.add_ys)
            
        else: # right click
            self.trace.append(False)
            self.rem_xs.append(x)
            self.rem_ys.append(y)
            self._show_points(self.rem_plot, self.rem_xs, self.rem_ys)
            
    def _show_points(self, plot, xs, ys):
        plot.set_data(xs, ys)
        self.fig.canvas.draw()
        
    def _on_key(self, event):
        if event.key == 'c':
            self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = [], [], [], [], []
            self.add_plot.set_data([], [])
            self.rem_plot.set_data([], [])
            self.fig.canvas.draw()
        
    
    def _update_mask_id(self, val):
        self.mask_id = self.slider_mask_id.val
        
        self.im = self.ax.imshow(self.img)
        self._show_anns([self.masks[self.mask_id]])
        self.fig.canvas.draw_idle()
        
        
    def _push_button_sam(self, _):
        self.status_text.set_text('Segmenting ... wait')
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
        
        self.mask_generator = SamAutomaticMaskGenerator(
        model=self.sam,
        points_per_side=self.points_per_side,
        pred_iou_thresh=self.pred_iou_thresh,
        stability_score_thresh=self.stability_score_thresh)
        
        self.masks = self.mask_generator.generate(self.img)
        
        # create a visualize or not radio button
        self.ax_visualize_radio = plt.axes([0.1, 0.05, 0.1, 0.1])  # [left, bottom, width, height]
        self.radio_buttons_visualize = RadioButtons(self.ax_visualize_radio, ['orignal', 'masks'], active=0,)
        self.radio_buttons_visualize.on_clicked(self._on_radio_button_visualize_clicked)
        
        # create a confirm button
        #self.status_text.set_val("Ready to confirm results")
        self.ax_button_sam_confirm = plt.axes([0.55, 0.05, 0.2, 0.05])  # [left, bottom, width, height]
        self.button_sam_confirm = Button(self.ax_button_sam_confirm, 'Confirm')
        self.button_sam_confirm_cid = self.button_sam_confirm.on_clicked(self._push_button_sam_confirm)
        self.status_text.set_text('Ready to SAM')
        self.mask_number_text.set_text("Mask number is " + str(len(self.masks)))
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    
    def _on_radio_button_visualize_clicked(self, label):
        if label=="masks":
            self._show_anns(self.masks)
            
        else:
            self.im = self.ax.imshow(self.img)
            self.fig.canvas.draw_idle()
    
    def _show_anns(self, anns):
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