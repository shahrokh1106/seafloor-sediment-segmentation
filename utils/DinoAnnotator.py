import random
import numpy as np
import torch
import os
import cv2
import timm
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.ndimage as ndimage


SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   

class DinoFeatureMatching():
    def __init__(self, refine_mask_auto = True,refine_mask_manual= True,superpixel="felzenszwalb", debug=True,show_scale_percentage=30):
        self.refine_mask_auto = refine_mask_auto
        self.refine_mask_manual = refine_mask_manual
        self.superpixel = superpixel
        self.debug = debug
        self.show_scale_percentage = show_scale_percentage
        self.inputsize = 518
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.transform = T.Compose([T.ToPILImage(),T.Resize((self.inputsize, self.inputsize)),T.ToTensor(),T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
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
    
    def get_patch_feature(self,patch):
        patch_image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            patch_tensor = self.transform(patch_image).unsqueeze(0).to(self.device)  # [1, 3, 518, 518]
            patch_embedding = self.model.forward_features(patch_tensor)  # base:[1, 1370, 768] large: [1, 1370, 1024]
            patch_embedding = patch_embedding[:, 1:, :].mean(dim=1) # [1, dim]-->base: [1, 768] large: [1, 1024]
            patch_embedding = patch_embedding.squeeze().cpu().numpy()
            patch_embedding = patch_embedding / np.linalg.norm(patch_embedding)
        return patch_embedding
    
    def get_image_feature(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            full_tensor = self.transform(image).unsqueeze(0).to(self.device) # [1, 3, 518, 518]
            full_embedding = self.model.forward_features(full_tensor)  # base:[1, 1370, 768] large: [1, 1370, 1024]
            full_embedding = full_embedding[:, 1:, :]  # remove CLS → base: [1, 1369, 768] large:[1, 1369, 1024]
            full_embedding = full_embedding[0].reshape(37, 37, 768)
        full_embedding = full_embedding.reshape(-1, full_embedding.shape[-1]).cpu().numpy()  # [1369, 768] 
        full_embedding = full_embedding / np.linalg.norm(full_embedding, axis=1, keepdims=True)
        return full_embedding
    
    def get_patch_feature_(self, patch):
        patch_image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            patch_tensor = self.transform(patch_image).unsqueeze(0).to(self.device)  # [1, 3, 518, 518]
            patch_embedding = self.model.forward_features(patch_tensor)  # [1, 1370, 768]
            cls_token = patch_embedding[:, 0]                          # [1, 768]
            spatial_tokens = patch_embedding[:, 1:, :].mean(dim=1)    # [1, 768]
            fused_vector = torch.cat([cls_token, spatial_tokens], dim=-1)  # [1, 1536]
            fused_vector = F.normalize(fused_vector, dim=-1)
            return fused_vector.squeeze().cpu().numpy()  # [1536,]
    
    def get_image_feature_(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            full_tensor = self.transform(image).unsqueeze(0).to(self.device)
            full_embedding = self.model.forward_features(full_tensor)  # [1, 1370, 768]
            spatial_tokens = full_embedding[:, 1:, :]  # [1, 1369, 768]
            cls_token = full_embedding[:, 0].repeat(1, 1369).reshape(1, 1369, 768)  # repeat CLS to align
            fused_tokens = torch.cat([cls_token, spatial_tokens], dim=-1)  # [1, 1369, 1536]
            fused_tokens = fused_tokens.squeeze().cpu().numpy()
            fused_tokens = fused_tokens / np.linalg.norm(fused_tokens, axis=1, keepdims=True)
        return fused_tokens  # [1369, 1536]
   
    def get_similarity_map(self,full_embedding,patch_embedding,width, height):
        similarity = full_embedding @ patch_embedding.T  # [1369,]
        sim_map = similarity.reshape(37, 37)
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

    def adaptive_mask(self, sim_map, sigma=2, k=0.5):
        blurred = ndimage.gaussian_filter(sim_map, sigma=sigma)
        mean_val = blurred.mean()
        std_val = blurred.std()
        threshold = mean_val + k * std_val
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
        global radius,drawing,drawing,value,MASK,alpha,colored_mask,repeat_flag, reset_flag
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
        def mouse_callback_editor(event, x, y, flags, param):
            global alpha,colored_mask,IMG, MASK, radius, drawing, value,repeat_flag,reset_flag

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
        resized_image = cv2.resize(img_org, (self.inputsize, self.inputsize), interpolation=cv2.INTER_AREA)
        patch_image = self.get_patch_image(resized_image,roi,width=width,height=height)
        full_image = img_org.copy()
        patch_embedding = self.get_patch_feature_(patch_image)
        image_embedding = self.get_image_feature_(full_image)
        sim_map= self.get_similarity_map(image_embedding, patch_embedding,width,height)
        overlay = self.get_overlay_heatmap(sim_map,img_org.copy())
        if self.debug:
            self.show_image(overlay)
        binary_mask = self.adaptive_mask(sim_map, sigma=2, k=0.5)
        if self.refine_mask_auto:
            binary_mask = self.refine_mask_with_superpixels(img_org.copy(), binary_mask, n_segments=700, compactness=10)
        if self.debug:
            self.show_image(binary_mask)
            overlay_binary = self.get_overlay_heatmap(cv2.cvtColor(binary_mask//255, cv2.COLOR_GRAY2BGR),img_org.copy())
            self.show_image(overlay_binary)
        if self.refine_mask_manual:
            binary_mask = self.refine_mask_manually(binary_mask, img_org.copy())
        return binary_mask
    



