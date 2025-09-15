import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import cv2
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2_repo'))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   

def get_smooth_sim_map(sim_map):
    # sim_map = sim_map.astype(np.float32)
    # mu  = cv2.GaussianBlur(sim_map, (0,0), sigmaX=1)
    # var = cv2.GaussianBlur((sim_map - mu)**2, (0,0), sigmaX=1)
    # sim_map = (sim_map - mu) / (np.sqrt(var) + 1e-6)
    # # to [0,1]
    # sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    return sim_map

class TorchPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components
    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self
    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected

def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device
    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()
    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)
    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)
    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))
    return reduced_feats, fit_pca


def resize_transform(mask_image: Image,image_size,patch_size):
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    # w_patches = int((w * image_size) / (h * patch_size))
    w_patches =  int(image_size / patch_size)
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

def get_points(image, show_scale=300):
    global drawing, prev_pt
    drawing = False  
    prev_pt = None
    def draw_freehand(event, x, y, flags, params):
        global drawing, prev_pt, points, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            prev_pt = (x, y)
            points.append(prev_pt)
            cv2.circle(img, prev_pt, 4, (0, 0, 255), -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cur_pt = (x, y)
            cv2.line(img, prev_pt, cur_pt, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            num_steps = max(abs(cur_pt[0]-prev_pt[0]), abs(cur_pt[1]-prev_pt[1])) + 1
            for t in range(num_steps):
                xi = int(prev_pt[0] + (cur_pt[0]-prev_pt[0]) * t / num_steps)
                yi = int(prev_pt[1] + (cur_pt[1]-prev_pt[1]) * t / num_steps)
                points.append((xi, yi))
            prev_pt = cur_pt
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            prev_pt = None
    global img, img_orig, points
    points = []
    w = int(image.shape[1] * show_scale / 100)
    h = int(image.shape[0] * show_scale / 100)
    image_small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    img = image_small.copy()
    img_orig = img.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_freehand)
    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"): 
            img = img_orig.copy()
            points = []
            prev_pt = None
        elif key == ord("s"):  # save/return points
            break
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    if len(points) != 0:
        rescaled_points = []
        for (px, py) in points:
            rx = int(px * 100 / show_scale)
            ry = int(py * 100 / show_scale)
            rescaled_points.append((rx, ry))
        return rescaled_points
    else:
        raise Exception("No roi has selected")

def get_rgb_feature_map(x_grid,input_size,show_scale, debug):
    [feats_pca, _], _ = pca([x_grid, x_grid])
    feat = feats_pca[0]
    feat_np = feat.permute(1, 2, 0).detach().cpu().numpy()
    feat_norm = cv2.normalize(feat_np, None, 0, 255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    feat_norm = cv2.resize(feat_norm, (input_size,input_size))
    w = int(feat_norm.shape[1] * show_scale / 100)
    h = int(feat_norm.shape[0] * show_scale / 100)
    feat_norm_show = cv2.resize(feat_norm, (w, h), interpolation = cv2.INTER_AREA)
    if debug:
        cv2.imshow("Feature map", feat_norm_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return feat_norm

def kmeans_on_feature_map(x_grid, k=5, fit_samples=100_000, batch_size=8192, standardize=True, seed=42):
    """
    x_grid: torch.Tensor [C,H,W] or [1,C,H,W] (float)
    returns: labels_map [H,W] (np.uint8), kmeans, (optional) scaler
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Ensure shape [C,H,W]
    if x_grid.dim() == 4:
        x_grid = x_grid[0]
    assert x_grid.dim() == 3, f"Expected [C,H,W], got {tuple(x_grid.shape)}"

    C, H, W = x_grid.shape
    feats = x_grid.detach().permute(1, 2, 0).reshape(-1, C).cpu().numpy().astype(np.float32)  # [N, C], N=H*W

    # Optional: standardize each feature dimension
    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        # fit scaler on a subset (or all if small)
        if feats.shape[0] > fit_samples:
            idx = rng.choice(feats.shape[0], size=fit_samples, replace=False)
            scaler.fit(feats[idx])
        else:
            scaler.fit(feats)
        feats_scaled = scaler.transform(feats)
    else:
        feats_scaled = feats

    # Fit MiniBatchKMeans on a subset for speed/memory; then predict full image
    if feats.shape[0] > fit_samples:
        idx_fit = rng.choice(feats.shape[0], size=fit_samples, replace=False)
        X_fit = feats_scaled[idx_fit]
    else:
        X_fit = feats_scaled

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=seed,
        n_init='auto',  # sklearn >=1.4
        max_no_improvement=10,
        verbose=0
    )
    kmeans.fit(X_fit)

    # Predict labels for all pixels
    labels = kmeans.predict(feats_scaled).astype(np.int32)         # [N]
    labels_map = labels.reshape(H, W).astype(np.uint8)             # [H,W]

    return labels_map, kmeans, scaler


def get_kmeans_clusters_rgb (x_grid, k, debug=True,seed=42):
    labels_map, kmeans, scaler = kmeans_on_feature_map(x_grid, k=k, fit_samples=120_000, batch_size=8192, standardize=True, seed=seed)
    vis = (labels_map.astype(np.float32) * (255.0 / max(1, k-1))).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    if debug:
        cv2.imshow("KMeans clusters on ORIGINAL features", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return vis

def get_sim_map_points(path,x, inputsize,patchsize,show_scale,debug, cmap=cv2.COLORMAP_INFERNO):
    image_bgr = cv2.imread(path)
    image_bgr = cv2.resize(image_bgr, (inputsize, inputsize))
    points_org = get_points (image_bgr ,show_scale = show_scale)
    points = [(point[0]//patchsize,point[1]//patchsize) for point in points_org]
    feats = x[0].detach().cpu().numpy()  
    point_features = []
    for point in points:
        point_features.append(feats[:,point[1],point[0]])
    point_features = np.vstack(point_features)
    point_features = point_features.mean(0)
    C, H, W = feats.shape
    feats = feats.reshape(C,H*W)

    #L2
    feats /= (np.linalg.norm(feats, axis=0, keepdims=True) + 1e-8)
    point_features /= (np.linalg.norm(point_features) + 1e-8)


    sim = point_features.T@feats
    sim_map = sim.reshape(H,W)
    sim_map = get_smooth_sim_map(sim_map)
    mean_vis = cv2.normalize(sim_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean_vis = cv2.applyColorMap(mean_vis, cmap) 
    mean_vis = cv2.resize(mean_vis, (inputsize, inputsize))
    w = int(mean_vis.shape[1] * show_scale / 100)
    h = int(mean_vis.shape[0] * show_scale / 100)
    mean_vis_show = cv2.resize(mean_vis, (w, h), interpolation = cv2.INTER_AREA)
    if debug:
        cv2.imshow("DINO FEATURES", mean_vis_show); cv2.waitKey(0)
        cv2.destroyAllWindows()
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
    return mean_vis,cv2.resize(sim_map, (inputsize,inputsize)),points_org

def get_sim_map_input_points(path,x,points_org, inputsize,patchsize,show_scale,debug, cmap=cv2.COLORMAP_INFERNO):
    image_bgr = cv2.imread(path)
    image_bgr = cv2.resize(image_bgr, (inputsize, inputsize))
    points = [(point[1]//patchsize,point[0]//patchsize) for point in points_org]
    feats = x[0].detach().cpu().numpy()  
    point_features = []
    for point in points:
        point_features.append(feats[:,point[1],point[0]])
    point_features = np.vstack(point_features)
    point_features = point_features.mean(0)
    C, H, W = feats.shape
    feats = feats.reshape(C,H*W)
    
    #L2
    feats /= (np.linalg.norm(feats, axis=0, keepdims=True) + 1e-8)
    point_features /= (np.linalg.norm(point_features) + 1e-8)

    sim = point_features.T@feats
    sim_map = sim.reshape(H,W)
    sim_map = get_smooth_sim_map(sim_map)
    mean_vis = cv2.normalize(sim_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean_vis = cv2.applyColorMap(mean_vis, cmap) 
    mean_vis = cv2.resize(mean_vis, (inputsize, inputsize))
    w = int(mean_vis.shape[1] * show_scale / 100)
    h = int(mean_vis.shape[0] * show_scale / 100)
    mean_vis_show = cv2.resize(mean_vis, (w, h), interpolation = cv2.INTER_AREA)
    if debug:
        cv2.imshow("DINO FEATURES", mean_vis_show); cv2.waitKey(0)
        cv2.destroyAllWindows()
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
    return mean_vis,cv2.resize(sim_map, (inputsize,inputsize)),points_org



def get_roi(image,show_scale = 300):
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
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"): # If 'r' is pressed, reset the cropping region
            img = img_orig.copy()
        elif key == ord("s"): # If 's' is pressed, break from the loop and do the cropping
            break
        elif key == ord("q"):
            break    
    cv2.destroyAllWindows()
    if len(rectangle_points)!=0:
        roi =  [(int(rectangle_points[0][0]*100/show_scale),int(rectangle_points[0][1]*100/show_scale)),(int(rectangle_points[1][0]*100/show_scale),int(rectangle_points[1][1]*100/show_scale))]
        return [roi[0][0],roi[0][1], roi[1][0],roi[1][1]]
    else:
        raise Exception("No roi has selected")
    


def get_sim_map_box(path,x, inputsize,patchsize,show_scale,debug, cmap=cv2.COLORMAP_INFERNO):
    image_bgr = cv2.imread(path)
    image_bgr = cv2.resize(image_bgr, (inputsize, inputsize))
    box_org = get_roi(image_bgr,show_scale = show_scale)
    box = [b//patchsize for b in box_org]
    x1,y1,x2,y2 = box
    feats = x[0].detach().cpu().numpy()  
    box_features = feats[:,y1:y2, x1:x2]
    box_features = box_features.mean(axis=(1, 2)) 
    C, H, W = feats.shape
    feats = feats.reshape(C,H*W)
    #L2
    feats /= (np.linalg.norm(feats, axis=0, keepdims=True) + 1e-8)
    box_features /= (np.linalg.norm(box_features) + 1e-8)
    sim = box_features.T@feats
    sim_map = sim.reshape(H,W)
    sim_map = get_smooth_sim_map(sim_map)
    mean_vis = cv2.normalize(sim_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean_vis = cv2.applyColorMap(mean_vis, cmap) 
    mean_vis = cv2.resize(mean_vis, (inputsize, inputsize))
    w = int(mean_vis.shape[1] * show_scale / 100)
    h = int(mean_vis.shape[0] * show_scale / 100)
    mean_vis_show = cv2.resize(mean_vis, (w, h), interpolation = cv2.INTER_AREA)
    if debug:
        cv2.imshow("DINO FEATURES", mean_vis_show); cv2.waitKey(0)
        cv2.destroyAllWindows()
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
    return mean_vis,cv2.resize(sim_map, (inputsize,inputsize)),box_org


def random_walk_refine(Y0, affinity, alpha=0.5, num_iter=10):
    DIRECTIONS = [
        (-1,  0),  # up
        ( 1,  0),  # down
        ( 0, -1),  # left
        ( 0,  1),  # right
        (-1, -1),  # up-left
        (-1,  1),  # up-right
        ( 1, -1),  # down-left
        ( 1,  1),  # down-right
    ]
    """
    Fast GPU-based random walk refinement.
    Args:
        Y0: [B, C, H, W] initial soft predictions (e.g., softmax outputs)
        affinity: [B, H, W, 8] affinity map
        alpha: propagation strength (0 < alpha < 1)
        num_iter: number of propagation iterations
    Returns:
        Refined soft predictions [B, C, H, W]
    """
    B, C, H, W = Y0.shape
    Y = Y0.clone()
    for _ in range(num_iter):
        Y_new = torch.zeros_like(Y)
        for i, (dx, dy) in enumerate(DIRECTIONS):
            affinity_weight = affinity[..., i]  # [B, H, W]
            affinity_weight = affinity_weight.unsqueeze(1)  # [B, 1, H, W]
            # Shift prediction
            shifted = F.pad(Y, (1, 1, 1, 1), mode='replicate')  # pad to handle borders
            shifted = shifted[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]  # shifted prediction
            Y_new += affinity_weight * shifted
        Y = alpha * Y_new + (1 - alpha) * Y0
    
    return Y


def compute_affinity_from_features(feat_map):
    """
    Compute 8-directional affinity map from feature embeddings.
    feat_map: Tensor of shape (B, D, H, W)
    Returns: Tensor of shape (B, H, W, 8)
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    affinities = []
    for dx, dy in directions:
        shifted = torch.roll(feat_map, shifts=(dx, dy), dims=(2, 3))  # shift H, W
        dist = torch.norm(feat_map - shifted, dim=1, keepdim=True)    # L2 over D

        affinity = torch.exp(-dist*10)  # similarity from distance
        affinities.append(affinity)  # (B, 1, H, W)
    # Stack and return as (B, H, W, 8)
    return torch.cat(affinities, dim=1).permute(0, 2, 3, 1)


def get_affinity_mask(sim_map,image_embedding_high,device,show_sclae,debug,cmap):
    sim_map_tensor = torch.tensor(sim_map, dtype=torch.float32, device=device)  # (H, W)
    foreground = sim_map_tensor  # similarity as foreground
    background = 1.0 - foreground  # inverse similarity as background
    sim_map_tensor = torch.stack([background, foreground], dim=0)  # (2, H, W)
    sim_map_tensor = sim_map_tensor.unsqueeze(0) # (1, 2, H, W)
    
    image_embedding_high = image_embedding_high.permute(0,3,1,2)
    
    affinity_map = compute_affinity_from_features(image_embedding_high).to(device)
    # affinity_map = compute_affinity_from_features(sim_map_tensor).to(device)
    affinity_map = F.interpolate(affinity_map.permute(0, 3, 1, 2), size=(sim_map_tensor.shape[2], sim_map_tensor.shape[3]), mode='bilinear', align_corners=False) 
    affinity_map = affinity_map.permute(0, 2, 3, 1)
    refined_mask = random_walk_refine(sim_map_tensor, affinity_map)
    refined_mask = refined_mask.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)*255


    refined_mask = (sim_map>0.5).astype(np.uint8)*255
    
    if debug:
        w = int(refined_mask.shape[1] * show_sclae / 100)
        h = int(refined_mask.shape[0] * show_sclae / 100)
        refined_mask_vis = cv2.resize(refined_mask, (w, h), interpolation = cv2.INTER_AREA)
        cv2.imshow("DINO FEATURES", refined_mask_vis); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cv2.applyColorMap(cv2.normalize(refined_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cmap) ,refined_mask



def get_overlay_heatmap(sim_map,img_org,debug,show_scale):
    heatmap = cv2.applyColorMap(np.uint8(sim_map), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img_org.shape[1], img_org.shape[0]))
    overlay = cv2.addWeighted(img_org, 0.5, heatmap, 0.5, 0)
    if debug:
        w = int(overlay.shape[1] * show_scale / 100)
        h = int(overlay.shape[0] * show_scale / 100)
        overlay_vis = cv2.resize(overlay, (w, h), interpolation = cv2.INTER_AREA)
        cv2.imshow("DINO FEATURES", overlay_vis); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return overlay


def get_mask_sam_points (path,points,inputsize,device,debug,cmap):
    sam_checkpoint = os.path.join("sam2_repo","checkpoints","sam2.1_hiera_large.pt")
    sam_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_predictor = SAM2ImagePredictor(build_sam2(sam_model_cfg, sam_checkpoint, device=device))
    sam_img = Image.open(path)
    sam_img = np.array(sam_img.convert("RGB"))
    sam_img = cv2.resize(sam_img, (inputsize, inputsize))
    # sam_predictor.reset_predictor()
    sam_predictor.set_image(sam_img)
    sam_masks, sam_scores, sam_logits = sam_predictor.predict(
                    point_coords=np.asarray(points),
                    point_labels=np.asarray([1.0 for _ in points]).reshape(len(points,)),
                    multimask_output=True)
    sorted_ind = np.argsort(sam_scores)[::-1]
    sam_masks = sam_masks[sorted_ind]
    sam_scores = sam_scores[sorted_ind]
    sam_logits = sam_logits[sorted_ind]
    sam_mask = sam_masks[0]
    sam_mask_vis = cv2.applyColorMap(cv2.normalize(sam_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cmap) 
    if debug:
        cv2.imshow("SAM2 MASK", sam_mask); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return sam_mask_vis,sam_mask


def get_mask_sam_box (path,box,inputsize,device,debug,cmap):
    sam_checkpoint = os.path.join("sam2_repo","checkpoints","sam2.1_hiera_large.pt")
    sam_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_predictor = SAM2ImagePredictor(build_sam2(sam_model_cfg, sam_checkpoint, device=device))
    sam_img = Image.open(path)
    sam_img = np.array(sam_img.convert("RGB"))
    sam_img = cv2.resize(sam_img, (inputsize, inputsize))
    # sam_predictor.reset_predictor()
    sam_predictor.set_image(sam_img)


    sam_masks, sam_scores, sam_logits = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.asarray(box)[None, :],
        multimask_output=False,
        )
    sorted_ind = np.argsort(sam_scores)[::-1]
    sam_masks = sam_masks[sorted_ind]
    sam_scores = sam_scores[sorted_ind]
    sam_logits = sam_logits[sorted_ind]
    sam_mask = sam_masks[0]
    sam_mask_vis = cv2.applyColorMap(cv2.normalize(sam_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cmap) 
    if debug:
        cv2.imshow("SAM2 MASK", sam_mask); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return sam_mask_vis,sam_mask
