import random
import numpy as np
import torch
import os
import cv2
import timm
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.ndimage as ndimage
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb
from skimage.segmentation import quickshift
from scipy import ndimage as ndi
from skimage.segmentation import slic
from PIL import Image
from FeatUp.featup.util import norm, unnorm
from FeatUp.featup.plotting import plot_feats, plot_lang_heatmaps
from featup.upsamplers import JBUStack
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   



class DINOv2SegHead(nn.Module):
    def __init__(self, backbone_name='vit_base_patch14_dinov2.lvd142m', num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        self.num_levels = 3
        in_channels_per_level = 768 
        projection_dim = 128
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_per_level, projection_dim, kernel_size=1),
                nn.BatchNorm2d(projection_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels)
        ])
        total_decoder_in_channels = projection_dim * self.num_levels  # = 128 * 3 = 384
        self.decoder = nn.Sequential(
            nn.Conv2d(total_decoder_in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.upsampler = JBUStack(feat_dim=num_classes)
 
    def forward(self, x):
        features = self.backbone(x)
        projected_feats = [proj(feat) for feat, proj in zip(features, self.proj_layers)]
        fused = torch.cat(projected_feats, dim=1)  
        out = self.decoder(fused)
        return  F.interpolate(self.upsampler(out,x), size=(518, 518), mode='bilinear', align_corners=False) 

class DINOv2SegHeadV2(nn.Module):
    def __init__(self, backbone_name='vit_base_patch14_dinov2.lvd142m', num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        self.num_levels = 3
        in_channels_per_level = 768 
        projection_dim = 128
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_per_level, projection_dim, kernel_size=1),
                nn.BatchNorm2d(projection_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels)
        ])
        total_decoder_in_channels = projection_dim * self.num_levels  # = 128 * 3 = 384
        self.decoder = nn.Sequential(
            nn.Conv2d(total_decoder_in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False)

    def forward(self, x):
        features = self.backbone(x)
        projected_feats = [proj(feat) for feat, proj in zip(features, self.proj_layers)]
        fused = torch.cat(projected_feats, dim=1)  # Shape: [B, 384, 37, 37]
        out = self.decoder(fused)
        return self.upsample(out)
    

class DinoFeatureMatching():
    def __init__(self, refine_mask_auto = True,refine_mask_manual= True,superpixel="felzenszwalb", debug=True,show_scale_percentage=30):
        self.refine_mask_auto = refine_mask_auto
        self.refine_mask_manual = refine_mask_manual
        self.superpixel = superpixel
        self.debug = debug
        self.show_scale_percentage = show_scale_percentage
        self.inputsize = 518
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upsampler = JBUStack(feat_dim=768).to(self.device)
        self.model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True).to(self.device)
        self.model.eval()
        self.upsampler.eval()
        self.transform = T.Compose([T.ToPILImage(),T.Resize((self.inputsize, self.inputsize)),T.ToTensor(),norm])
    
    def get_roi(self,image,show_scale = 300):
        IMAGE = image.copy()
        def draw_rectangle(event, x, y, flags, params):
            global rectangle_points, img, img_orig
            if event == cv2.EVENT_LBUTTONDOWN:
                rectangle_points = [(x, y)]
            elif event == cv2.EVENT_LBUTTONUP:
                rectangle_points.append((x, y))
                cv2.rectangle(img, rectangle_points[0], rectangle_points[1], (0, 255, 0), 2)
                cv2.imshow("image", img)
        global img,img_orig,rectangle_points
        rectangle_points = []
        
        w = int(image.shape[1] * show_scale / 100)
        h = int(image.shape[0] * show_scale / 100)
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        img = image.copy()
        img_orig = img.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_rectangle)
        while True:
            text_0 = "Press \"r\" to reset the croping"
            text_1 = "Hold the left click and draw the box, then press \"c\" to crop"
            cv2.putText(img, text_0, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(img, text_1, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"): # If 'r' is pressed, reset the cropping region
                img = img_orig.copy()
            elif key == ord("c"): # If 'c' is pressed, break from the loop and do the cropping
                break
            elif key == ord("q"):
                break    
        cv2.destroyAllWindows()
        if len(rectangle_points)!=0:
            roi =  [(int(rectangle_points[0][0]*100/show_scale),int(rectangle_points[0][1]*100/show_scale)),(int(rectangle_points[1][0]*100/show_scale),int(rectangle_points[1][1]*100/show_scale))]
            return [roi[0][0],roi[0][1], roi[1][0],roi[1][1]]
        else:
            raise Exception("No roi has selected")
    
    def get_patch_image(self, image, roi,width,height):
        x1,y1,x2,y2 = roi
        scale_x = self.inputsize / width
        scale_y = self.inputsize / height
        x1_r = int(x1 * scale_x)
        x2_r = int(x2 * scale_x)
        y1_r = int(y1 * scale_y)
        y2_r = int(y2 * scale_y)
        patch_image = image[y1_r:y2_r, x1_r:x2_r]
        return patch_image
    
    def get_patch_feature(self, roi,width,height, image_embedding):
        H_prime = int(np.sqrt(image_embedding.shape[0]))
        W_prime = H_prime
        C = image_embedding.shape[1]
        image_embedding = image_embedding.reshape(H_prime, W_prime, C)  # shape: [H', W', C]
        # Scale ROI to match embedding resolution
        scale_x = W_prime / width
        scale_y = H_prime / height
        x1, y1, x2, y2 = roi
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        patch_features = image_embedding[y1:y2, x1:x2, :]  # shape: [h, w, C]
        patch_features = patch_features.mean(axis=(0, 1)) 
        return patch_features
    
    def get_image_feature(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(self.device) # [1, 3, 518, 518]
            image_embedding =self.model.forward_features(image_tensor)  # [1, 1370, 768]
            image_embedding = image_embedding[:, 1:, :] # [1, 1369, 768]
            image_embedding = image_embedding.reshape(1,int(np.sqrt(image_embedding.shape[1])), int(np.sqrt(image_embedding.shape[1])), image_embedding.shape[2])
            image_embedding = image_embedding.permute(0, 3, 1, 2)
            image_embedding = self.upsampler(image_embedding, image_tensor)
            image_embedding = image_embedding.contiguous().view(1,image_embedding.shape[1],-1) # [1, 768, 350464] --> 768,592*592
            image_embedding = image_embedding.permute(0,2,1).squeeze().cpu().numpy()  # [1, 350464, 768]
            image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=1, keepdims=True)
        return image_embedding
   
    def get_similarity_map(self,image_embedding,patch_embedding,width, height):
        similarity = image_embedding @ patch_embedding.T  # [350464,]
        sim_map = similarity.reshape(int(np.sqrt(similarity.shape[0])), int(np.sqrt(similarity.shape[0]))) # [592,592]
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())

        sim_map_resized = cv2.resize(sim_map, (width, height), interpolation=cv2.INTER_CUBIC)
        return sim_map_resized
    
    def get_overlay_heatmap(self, sim_map_resized,img_org):
        heatmap = cv2.applyColorMap(np.uint8(255 * sim_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img_org.shape[1], img_org.shape[0]))
        overlay = cv2.addWeighted(img_org, 0.5, heatmap, 0.5, 0)
        return overlay
    def show_image(self, image):
        w = int(image.shape[1] * self.show_scale_percentage / 100)
        h = int(image.shape[0] * self.show_scale_percentage / 100)
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def adaptive_mask(self, sim_map, sigma=2, k=2):
    #     blurred = ndimage.gaussian_filter(sim_map, sigma=sigma)
    #     mean_val = blurred.mean()
    #     std_val = blurred.std()
    #     threshold = mean_val + k * std_val
    #     binary_mask = ((blurred > threshold) * 255).astype(np.uint8)
    #     return binary_mask
    
    def adaptive_mask(self, sim_map, sigma=2, threshold=0.5):
        blurred = ndimage.gaussian_filter(sim_map, sigma=sigma)
        binary_mask = ((blurred > threshold) * 255).astype(np.uint8)
        return binary_mask
    
    def refine_mask_with_superpixels(self, image_bgr, binary_mask, n_segments=700, compactness=10):
        image_lab = img_as_float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))
        scale_factor = 0.5
        small_image = cv2.resize(image_lab, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        if self.superpixel == "quickshift":
            segments_small = quickshift(small_image, kernel_size=3, max_dist=6, ratio=0.5)
        elif self.superpixel == "slic":
            segments_small = slic(small_image, n_segments=n_segments, compactness=10, start_label=1)
        elif self.superpixel == "felzenszwalb":
            segments_small = felzenszwalb(small_image, scale=3.0, sigma=0.95, min_size=10)
        else:
            raise Exception("superpixel method must be one of the following:  felzenszwalb, quickshift,slic")

        segments = cv2.resize(segments_small.astype(np.uint16), (image_lab.shape[1], image_lab.shape[0]), interpolation=cv2.INTER_NEAREST)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.debug:
            boundary_img = mark_boundaries(image_rgb, segments, color=(1, 0, 0)) 
            self.show_image(boundary_img)
        refined_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        fg_ratio = ndi.mean(binary_mask == 255, labels=segments, index=np.unique(segments))
        accepted = np.unique(segments)[fg_ratio > 0.5]
        mask_lookup = np.isin(segments, accepted)
        refined_mask = (mask_lookup * 255).astype(np.uint8)
        return refined_mask
    
    def refine_mask_manually(self, mask,image):
        global radius,drawing,drawing,value,MASK,alpha,colored_mask,repeat_flag, reset_flag,fill_flag
        radius = 5
        drawing = False
        value = 255
        w = int(image.shape[1] * self.show_scale_percentage / 100)
        h = int(image.shape[0] * self.show_scale_percentage / 100)
        IMAGE = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        MASK = cv2.resize(mask, (w, h), interpolation = cv2.INTER_AREA)
        colored_mask = cv2.applyColorMap(MASK, cv2.COLORMAP_JET)
        alpha = 0.7
        repeat_flag=False
        reset_flag =False
        fill_flag = False
        def mouse_callback_editor(event, x, y, flags, param):
            global alpha,colored_mask,IMG, MASK, radius, drawing, value,repeat_flag,reset_flag, fill_flag

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                value = 255
                MASK = cv2.circle(MASK, (x, y), radius, value, -1)
                colored_mask = cv2.applyColorMap(MASK, cv2.COLORMAP_JET)
                IMG = cv2.addWeighted(IMAGE.copy(), alpha, colored_mask, 1-alpha, 0)    

            elif event == cv2.EVENT_RBUTTONDOWN:
                drawing = True
                value = 0
                MASK = cv2.circle(MASK, (x, y), radius, value, -1)
                colored_mask = cv2.applyColorMap(MASK, cv2.COLORMAP_JET)
                IMG = cv2.addWeighted(IMAGE.copy(), alpha, colored_mask, 1-alpha, 0)  

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    MASK = cv2.circle(MASK, (x, y), radius, value, -1)
                    colored_mask = cv2.applyColorMap(MASK, cv2.COLORMAP_JET)
                    IMG = cv2.addWeighted(IMAGE.copy(), alpha, colored_mask, 1-alpha, 0)  
                if repeat_flag==True:
                    MASK = self.refine_mask_with_superpixels(image.copy(), cv2.resize(MASK.copy(),(mask.shape[1], mask.shape[0])), n_segments=700, compactness=10)
                    MASK = cv2.resize(MASK, (w, h), interpolation = cv2.INTER_AREA)
                    repeat_flag = False
                if reset_flag==True:
                    MASK = np.zeros_like(MASK).astype(np.uint8)
                    reset_flag = False
                if fill_flag==True:
                    MASK = (np.ones_like(MASK)*255).astype(np.uint8)
                    fill_flag = False
            elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                drawing = False
            cv2.imshow('Mask', IMG)
        global IMG
        while(True):
            IMG = cv2.addWeighted(IMAGE.copy(), alpha, colored_mask, 1-alpha, 0)
            cv2.imshow('Mask', IMG)
            cv2.setMouseCallback('Mask', mouse_callback_editor)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("1"):
                radius=10
            if key == ord("2"):
                radius=20
            if key == ord("3"):
                radius=30  
            if key == ord("4"):
                radius=40  
            if key == ord("5"):
                radius=50  
            if key == ord("6"):
                radius=60  

            if key == ord("f"):
                if alpha+0.1<=1:
                    alpha+=0.1
            if key == ord("d"):
                if alpha-0.1>=0:
                    alpha-=0.1
            if key == ord("a"):
                repeat_flag = True
            if key == ord("s"):
                cv2.imwrite("saved_temp_image.png", IMG)
            if key == ord("r"):
                reset_flag = True
            if key==  ord("t"):
                fill_flag = True
            if key == 27:
                break
        MASK = cv2.resize(MASK,(mask.shape[1], mask.shape[0]))
        cv2.destroyAllWindows()
        return MASK
    
    def run(self,img_path, roi=[]):
        img_org = cv2.imread(img_path)
        height, width = img_org.shape[:2]
        if len(roi)==0:
            roi = self.get_roi(img_org,show_scale = self.show_scale_percentage)
        elif len(roi)!=4:
            print("The given region of interest (roi) must have four coordinate as x1,y1,x2,y2")
            roi = self.get_roi(img_org,show_scale = self.show_scale_percentage)
        full_image = img_org.copy()
        image_embedding = self.get_image_feature(full_image)
        patch_embedding = self.get_patch_feature(roi,width,height, image_embedding)
        sim_map= self.get_similarity_map(image_embedding, patch_embedding,width,height)
        overlay = self.get_overlay_heatmap(sim_map,img_org.copy())
        if self.debug:
            self.show_image(overlay)
        # binary_mask = self.adaptive_mask(sim_map, sigma=2, k=0.5)
        binary_mask = self.adaptive_mask(sim_map)
        if self.refine_mask_auto:
            binary_mask = self.refine_mask_with_superpixels(img_org.copy(), binary_mask, n_segments=700, compactness=10)
        if self.debug:
            self.show_image(binary_mask)
            overlay_binary = self.get_overlay_heatmap(cv2.cvtColor(binary_mask//255, cv2.COLOR_GRAY2BGR),img_org.copy())
            self.show_image(overlay_binary)
        if self.refine_mask_manual:
            binary_mask = self.refine_mask_manually(binary_mask, img_org.copy())
        return binary_mask
    



class MultiMaskAnnotator():
    def __init__(self, masks_dict,class_order, show_scale_percentage=50):
        self.show_scale_percentage = show_scale_percentage
        self.masks_dict = masks_dict
        self.class_order = class_order
        self.masks_dict = {cls: masks_dict[cls] for cls in self.class_order}
        self.class_names = list(masks_dict.keys())
        self.class_colors = self._generate_colors(len(self.class_names))
        self.current_class_idx = 0
        self.radius = 10
        self.drawing = False
        self.erasing = False
        self.auto_refine = False
        self.reset = False

    def _generate_colors(self, n):
        np.random.seed(42)
        return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n)]
    
    def refine_mask_with_superpixels(self, image_bgr, binary_mask):
        image_lab = img_as_float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))
        scale_factor = 0.5
        small_image = cv2.resize(image_lab, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        segments_small = felzenszwalb(small_image, scale=3.0, sigma=0.95, min_size=10)
        segments = cv2.resize(segments_small.astype(np.uint16), (image_lab.shape[1], image_lab.shape[0]), interpolation=cv2.INTER_NEAREST)
        refined_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        fg_ratio = ndi.mean(binary_mask == 255, labels=segments, index=np.unique(segments))
        accepted = np.unique(segments)[fg_ratio > 0.5]
        mask_lookup = np.isin(segments, accepted)
        refined_mask = (mask_lookup * 255).astype(np.uint8)
        return refined_mask
    
    
    def _mouse_callback(self, event, x, y, flags, param):
        global img
        class_name = self.class_names[self.current_class_idx]
        mask = self.masks_dict[class_name]
        if self.auto_refine:
            mask = self.refine_mask_with_superpixels(img.copy(), mask)
            self.auto_refine = False
        if self.reset == True:
            mask = np.zeros_like(mask)
            self.reset=False
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        elif event == cv2.EVENT_RBUTTONUP:
            self.erasing = False

        if self.drawing:
            cv2.circle(mask, (x, y), self.radius, 255, -1)
        elif self.erasing:
            cv2.circle(mask, (x, y), self.radius, 0, -1)
        self.masks_dict[class_name]=mask

    def annotate(self, image):
        alpha = 0.9
        global img
        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", self._mouse_callback)

        while True:
            overlay = image.copy()
            img = image.copy()
            class_name = self.class_names[self.current_class_idx]            
            mask = self.masks_dict[class_name]
            color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(overlay, alpha, color_mask, 1-alpha, 0)
            cv2.putText(overlay, f"Class: {class_name} (Key: {self.current_class_idx + 1})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Annotator", overlay)

            key = cv2.waitKey(2) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("f"):
                if alpha+0.03<1:
                    alpha+=0.03
            elif key == ord("d"):
                if alpha-0.03>0:
                    alpha-=0.03
            elif key == ord("w"):
                self.radius-=5
            elif key == ord("e"):
                self.radius+=5
            elif key == ord("a"):
                self.auto_refine=True
            elif key == ord("r"):
                self.reset = True
            elif key in [ord(str(i+1)) for i in range(len(self.class_names))]:
                self.current_class_idx = key - ord('1')

        cv2.destroyAllWindows()
        return self._combine_masks()

    def _combine_masks(self):


        height, width = next(iter(self.masks_dict.values())).shape
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        index_order = {
            "Mud": 1.0,
            "Sand": 2.0,
            "Shellhash coverage": 3.0,
            "Dog Cockle Bed": 4.0,
            "PatchesWorms": 5.0,
            "Bryozoans": 6.0,
        }
        for class_name in self.class_order:
            combined_mask[self.masks_dict[class_name] == 255] = index_order[class_name]
        return combined_mask
    


class DinoSegmentorMultiClassFromBinary():
    def __init__(self, model_paths,show_scale_percentage,refine_mask_auto,refine_mask_manual,superpixel,debug, thresholds=[0.5 for i in range(6)],class_order= []):
        self.model_paths = model_paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dict = self.load_models()
        self.transform = A.Compose([A.Resize(518, 518),A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),ToTensorV2()])
        self.show_scale_percentage = show_scale_percentage
        self.refine_mask_auto = refine_mask_auto
        self.refine_mask_manual = refine_mask_manual
        self.superpixel = superpixel
        self.debug = debug
        self.thresholds = thresholds
        if len(class_order)==0:
            raise ValueError ("class order should be provided as a list of class names")
        self.class_order = class_order

    def load_models(self):
        class_names = list(self.model_paths.keys())
        models_dict = dict()
        for class_name in class_names:
            # temp_model = DINOv2SegHead()
            temp_model = DINOv2SegHeadV2()
            temp_model.load_state_dict(torch.load(self.model_paths[class_name], map_location=self.device))
            temp_model.eval()
            temp_model.to(self.device)
            models_dict.update({class_name:temp_model})
            print(f"Model for {class_name} has been loaded")
        return models_dict
    
    def show_image(self, image):
        w = int(image.shape[1] * self.show_scale_percentage / 100)
        h = int(image.shape[0] * self.show_scale_percentage / 100)
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def preprocess_single(self, image):
        image = self.transform(image=image)
        image_tensor = image['image'].unsqueeze(0).to(self.device)  
        return image_tensor
    
    def predict(self, path):
        class_names = list(self.model_paths.keys())
        image = cv2.imread(path)
        self.height,self.width, = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess_single(image)
        masks_dict = dict()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for index,class_name in enumerate(class_names):
                output = self.models_dict[class_name](input_tensor)
                prob = torch.sigmoid(output)
                mask = (prob > self.thresholds[index]).float().cpu().squeeze().numpy() 
                mask = (mask*255).astype(np.uint8)
                mask = cv2.resize(mask, (self.width, self.height))
                masks_dict.update({class_name:mask})
        return masks_dict  
    
    def refine_mask_with_superpixels(self, image_bgr, binary_mask, n_segments=700, compactness=10):
        image_lab = img_as_float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))
        scale_factor = 0.5
        small_image = cv2.resize(image_lab, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        if self.superpixel == "quickshift":
            segments_small = quickshift(small_image, kernel_size=3, max_dist=6, ratio=0.5)
        elif self.superpixel == "slic":
            segments_small = slic(small_image, n_segments=n_segments, compactness=10, start_label=1)
        elif self.superpixel == "felzenszwalb":
            segments_small = felzenszwalb(small_image, scale=3.0, sigma=0.95, min_size=10)
        else:
            raise Exception("superpixel method must be one of the following:  felzenszwalb, quickshift,slic")

        segments = cv2.resize(segments_small.astype(np.uint16), (image_lab.shape[1], image_lab.shape[0]), interpolation=cv2.INTER_NEAREST)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.debug:
            boundary_img = mark_boundaries(image_rgb, segments, color=(1, 0, 0)) 
            self.show_image(boundary_img)
        refined_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        fg_ratio = ndi.mean(binary_mask == 255, labels=segments, index=np.unique(segments))
        accepted = np.unique(segments)[fg_ratio > 0.5]
        mask_lookup = np.isin(segments, accepted)
        refined_mask = (mask_lookup * 255).astype(np.uint8)
        return refined_mask
    
    def run(self, image_path):
        image = cv2.imread(image_path)
        masks_dict = self.predict(image_path)
        width = image.shape[1]
        height = image.shape[0]
        w = int(width * self.show_scale_percentage / 100)
        h = int(height * self.show_scale_percentage / 100)
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        for key in masks_dict.keys():
            msk = cv2.resize(masks_dict[key], (w, h), interpolation = cv2.INTER_AREA)
            if self.refine_mask_auto:
                masks_dict[key] = self.refine_mask_with_superpixels(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB),msk)
            else:
                masks_dict[key] = msk

        annotator = MultiMaskAnnotator(masks_dict,class_order=self.class_order, show_scale_percentage=self.show_scale_percentage)
        if self.refine_mask_manual:
            mask = annotator.annotate(image.copy())
        mask = annotator._combine_masks()
        mask = cv2.resize(mask, (width, height), interpolation = cv2.INTER_AREA)
        return mask



class DinoSegmentorMultiClassRefiner():
    def __init__(self, label_map,model_path,input_size,predict_mode,show_scale_percentage,refine_mask_auto,superpixel,refine_mask_manual,debug,thresholds):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = label_map
        self.NUM_CLASSES = len(label_map)
        self.input_size =input_size
        self.model = self.load_model()
        self.transform = A.Compose([A.Resize(self.input_size, self.input_size),A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),ToTensorV2()])
        self.predict_mode = predict_mode
        self.show_scale_percentage = show_scale_percentage
        self.refine_mask_auto = refine_mask_auto
        self.superpixel = superpixel
        self.refine_mask_manual = refine_mask_manual
        self.debug = debug
        self.thresholds = thresholds
        


    def load_model(self):
        model = DINOv2SegHeadV2(num_classes=self.NUM_CLASSES).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        return model
    
    
    def show_image(self, image):
        w = int(image.shape[1] * self.show_scale_percentage / 100)
        h = int(image.shape[0] * self.show_scale_percentage / 100)
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def preprocess_single(self, image):
        image = self.transform(image=image)
        image_tensor = image['image'].unsqueeze(0).to(self.device)  
        return image_tensor
    
    def predict(self, path):
        image = cv2.imread(path)
        self.height,self.width, = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess_single(image)
        masks_dict = dict()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            output = self.model(input_tensor)
            preds = torch.argmax(output, dim=1)
            preds = preds[0].cpu().squeeze().numpy() 

        for index, class_name in enumerate(self.label_map):
            msk = (preds==index)
            msk = (msk*255).astype(np.uint8)
            msk = cv2.resize(msk, (self.width, self.height))
            masks_dict.update({class_name:msk})
        return masks_dict  


    def predict_(self, path, threshold=0.5):
        image = cv2.imread(path)
        self.height, self.width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess_single(image)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            output = self.model(input_tensor)

        probs = torch.sigmoid(output)  # shape: [B, C, H, W]
        masks = probs.cpu().numpy()[0]
        output_mask = np.zeros((masks.shape[1], masks.shape[2])).astype(np.uint8)
        thresholds = self.thresholds
        for index in range(len(masks)):
            output_mask[masks[index]>=thresholds[index]]=index
        
        masks_dict = dict()
        for index, class_name in enumerate(self.label_map):
            msk = (output_mask==index).astype(np.uint8) * 255
            msk = cv2.resize(msk, (self.width, self.height))
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel)
            # msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
            masks_dict[class_name] = msk.copy()

        return masks_dict
    
    
    def refine_mask_with_superpixels(self, image_bgr, binary_mask, n_segments=700, compactness=10):
        image_lab = img_as_float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))
        scale_factor = 0.5
        small_image = cv2.resize(image_lab, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        if self.superpixel == "quickshift":
            segments_small = quickshift(small_image, kernel_size=3, max_dist=6, ratio=0.5)
        elif self.superpixel == "slic":
            segments_small = slic(small_image, n_segments=n_segments, compactness=10, start_label=1)
        elif self.superpixel == "felzenszwalb":
            segments_small = felzenszwalb(small_image, scale=3.0, sigma=0.95, min_size=10)
        else:
            raise Exception("superpixel method must be one of the following:  felzenszwalb, quickshift,slic")

        segments = cv2.resize(segments_small.astype(np.uint16), (image_lab.shape[1], image_lab.shape[0]), interpolation=cv2.INTER_NEAREST)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.debug:
            boundary_img = mark_boundaries(image_rgb, segments, color=(1, 0, 0)) 
            self.show_image(boundary_img)
        refined_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        fg_ratio = ndi.mean(binary_mask == 255, labels=segments, index=np.unique(segments))
        accepted = np.unique(segments)[fg_ratio > 0.5]
        mask_lookup = np.isin(segments, accepted)
        refined_mask = (mask_lookup * 255).astype(np.uint8)
        return refined_mask
    
    def run(self, image_path):
        image = cv2.imread(image_path)
        if self.predict_mode == "sigmoid":
            masks_dict = self.predict_(image_path)
        else:
            masks_dict = self.predict(image_path)
        width = image.shape[1]
        height = image.shape[0]
        w = int(width * self.show_scale_percentage / 100)
        h = int(height * self.show_scale_percentage / 100)
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        for key in masks_dict.keys():
            msk = cv2.resize(masks_dict[key], (w, h))
            if self.refine_mask_auto:
                masks_dict[key] = self.refine_mask_with_superpixels(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB),msk)
            else:
                masks_dict[key] = msk
        masks_dict.pop('Background', None)
    
        annotator = MultiMaskAnnotator(masks_dict, class_order=list(masks_dict.keys()), show_scale_percentage=self.show_scale_percentage)
        if self.refine_mask_manual:
            mask = annotator.annotate(image.copy())
        mask = annotator._combine_masks()
        mask = cv2.resize(mask, (width, height), interpolation = cv2.INTER_AREA)
        return mask


