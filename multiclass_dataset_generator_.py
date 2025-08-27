import os
import glob
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
import random
from FeatUp.featup.util import norm
import torchvision.transforms as T

DEBUG = False
MASK_SCALE = 30
SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   

COLOR_MAP = {
        'Mud':  (49, 46, 166),
        'Sand': (255, 117, 10),
        'Shellhash coverage': (0, 117, 239),      
        'Dog Cockle Bed': (67, 225, 65),      
        'PatchesWorms': (80, 212, 255),      
        'Bryozoans': (164, 1, 236),     
        }
COLOR_MAP_INDEX = {
        0: (0,0,0),
        1:  (49, 46, 166),
        2: (255, 117, 10),
        3: (0, 117, 239),      
        4: (67, 225, 65),      
        5: (80, 212, 255),      
        6: (164, 1, 236),    
        255: (255,255,255), 
        }

@torch.no_grad()
def build_class_prototypes_from_seeds(
    feats: torch.Tensor,    
    seeds: torch.Tensor,   
    num_classes: int,
    ):
    """
    Returns:
      protos: (B,C,D) unit-norm prototypes
      valid_mask: (B,C) bool, True if class has at least one seed pixel
    """
    B, D, H, W = feats.shape
    device = feats.device
    protos = torch.zeros((B, num_classes, D), device=device, dtype=feats.dtype)
    valid_mask = torch.zeros((B, num_classes), device=device, dtype=torch.bool)
    # normalize features along channel just in case
    feats_n = F.normalize(feats, p=2, dim=1)
    for b in range(B):
        sb = seeds[b]  # (H,W)
        for c in range(num_classes):
            m = (sb == c)
            if m.any():
                vecs = feats_n[b, :, m]          # (D, N)
                proto = vecs.mean(dim=1)         # (D,)
                proto = F.normalize(proto, p=2, dim=0)
                protos[b, c] = proto
                valid_mask[b, c] = True
    return protos, valid_mask

@torch.no_grad()
def cosine_similarity_maps(
    feats: torch.Tensor,        # (B,D,H,W)
    protos: torch.Tensor,       # (B,C,D)
    valid_mask: torch.Tensor,   # (B,C) bool
    ):
    """
    Returns sim: (B,C,H,W) in [-1,1]; invalid classes filled with -inf
    """
    B, D, H, W = feats.shape
    _, C, _ = protos.shape
    device = feats.device
    feats_n = F.normalize(feats, p=2, dim=1)
    sim = torch.empty((B, C, H, W), device=device, dtype=feats.dtype)
    # (B,D,H,W) ⋅ (B,C,D) → (B,C,H,W)
    # Expand prototypes to (B,C,D,1,1) and do tensordot-like matmul
    protos_bc = protos.view(B, C, D, 1, 1)         # (B,C,D,1,1)
    sim = (feats_n.unsqueeze(1) * protos_bc).sum(dim=2)  # (B,C,H,W)
    # Mask out invalid classes
    invalid = ~valid_mask  # (B,C)
    if invalid.any():
        sim[invalid.unsqueeze(-1).unsqueeze(-1).expand_as(sim)] = float('-inf')
    return sim

def neighbor_affinity_support(cur_labels: torch.Tensor, affinity: torch.Tensor):
    """
    cur_labels: (B,H,W) in {0..C-1,255}
    affinity:   (B,H,W,8) in [0,1]
    Return: support (B,C,H,W) in [0,1], max aff to any neighbor of that class
    """
    B, H, W, K = affinity.shape
    device = affinity.device
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    C = (cur_labels[cur_labels != 255].max().item()+1) if (cur_labels != 255).any() else 1
    support = torch.zeros((B, C, H, W), device=device)
    for k,(dy,dx) in enumerate(dirs):
        neigh = F.pad(cur_labels, (1,1,1,1), mode='replicate')[:, 1+dy:H+1+dy, 1+dx:W+1+dx]
        affk = affinity[..., k]
        for c in range(C):
            support[:, c] = torch.maximum(support[:, c], affk * (neigh == c).float())
    return support

@torch.no_grad()
def expand_with_feats(
    probs: torch.Tensor,          # (B,C,H,W) softmax from logits
    seeds: torch.Tensor,          # (B,H,W) uint8 in {0..C-1,255}
    feats: torch.Tensor,          # (B,D,H,W) features
    affinity: torch.Tensor,       # (B,H,W,8) affinity
    img_org = None):

    device = probs.device
    B, C, H, W = probs.shape
    labels = seeds.clone().to(torch.int64)
    seed_mask = (seeds != 255)
    img_org = cv2.resize(img_org, (labels.shape[2],labels.shape[1]))

    if DEBUG:
        mask_vis = labels[0].cpu().numpy().astype(np.uint8) 
        mask_vis[seed_mask[0].cpu().numpy()] = mask_vis[seed_mask[0].cpu().numpy()]
        mask_rgb = np.zeros((*mask_vis.shape,3),dtype=np.uint8)
        for k,c in COLOR_MAP_INDEX.items(): mask_rgb[mask_vis==k]=c
        overlay = cv2.addWeighted(img_org,0.6,mask_rgb,0.4,0)
        cv2.imshow("Initial Seed", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 1) Build prototypes from seeds
    protos, valid = build_class_prototypes_from_seeds(feats, seeds, num_classes=C)
    # If no seeds at all, just return model argmax (nothing to expand from)
    if (~seed_mask).all():
        return probs.argmax(dim=1)
    # 2) Cosine similarity maps to prototypes
    sim = cosine_similarity_maps(feats, protos, valid)  # (B,C,H,W)
    if DEBUG:
        sim_vis = []
        for cls in range(6):
            sim_map = sim[0, cls].detach().cpu().numpy()   # (H,W), [-1,1]
            sim_norm = cv2.normalize(sim_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sim_color = cv2.applyColorMap(sim_norm, cv2.COLORMAP_JET)
            sim_vis.append(sim_color)
        rows, cols = 2, int(np.ceil(len(sim_vis) / 2))
        h, w = sim_vis[0].shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i, img in enumerate(sim_vis):
            r, c = divmod(i, cols)
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
        cv2.imshow(f"Cosine similarity - classes", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    still = (labels == 255)
    
    if still.any():
        feat_top1v, feat_top1c = sim.max(dim=1)
        feat_top1v = (feat_top1v + 1) / 2
        fill_by_feat = still & (feat_top1v > 0.6)
        labels[fill_by_feat] = feat_top1c[fill_by_feat]

    assign = (labels==255)
    labels[assign] = 255

    if DEBUG:
        mask_vis = labels[0].cpu().numpy().astype(np.uint8) 
        mask_vis[seed_mask[0].cpu().numpy()] = mask_vis[seed_mask[0].cpu().numpy()]
        mask_rgb = np.zeros((*mask_vis.shape,3),dtype=np.uint8)
        for k,c in COLOR_MAP_INDEX.items(): mask_rgb[mask_vis==k]=c
        overlay = cv2.addWeighted(img_org,0.6,mask_rgb,0.4,0)
        cv2.imshow("Initial Seed", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return labels

def make_seeds(probs, tau_high=0.5, min_margin=0.15):
    top1p, top1c = probs.max(dim=1)  # (B, H, W)
    sortedp, _ = probs.sort(dim=1, descending=True)  # (B, C, H, W)
    margin = (sortedp[:, 0] - sortedp[:, 1])         # (B, H, W)
    seeds = torch.full_like(top1c, 255, dtype=torch.uint8)
    # Keep only confident, unambiguous pixels
    keep = (top1p >= tau_high) & (margin >= min_margin)
    seeds[keep] = top1c[keep].byte()
    return seeds  # (B, H, W)

def random_walk_refine(Y0, affinity, alpha=0.9, num_iter=20):
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
        for i, (dy, dx) in enumerate(DIRECTIONS):
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


def get_logits_and_features(path, input_size,device,upsampler_encoder_feats,model):
    def preprocess_single(image, device):
        transform = A.Compose([A.Resize(input_size, input_size),A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),ToTensorV2()])
        image = transform(image=image)
        image_tensor = image['image'].unsqueeze(0).to(device)  
        return image_tensor
    image = cv2.imread(path)
    height,width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_single(image,device=device)
    transform = T.Compose([T.Resize((input_size,input_size)),T.ToTensor(),norm])
    image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        outputs = model(input_tensor, output_hidden_states=True)
        logits = outputs.logits
        logits = nn.functional.interpolate(logits, size=(input_size,input_size),mode="bilinear",align_corners=False)
        encoder_feats = upsampler_encoder_feats(image_tensor)
        encoder_feats = F.normalize(encoder_feats, p=2, dim=1) 
    with torch.no_grad():
        outputs = model.segformer(image_tensor, output_hidden_states=True,return_dict=True)
        seg_features = F.normalize(outputs.hidden_states[0], p=2, dim=1) 
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR),logits, encoder_feats,seg_features



if __name__ == "__main__":
    model_name = "SegformerDatasetV00.pth"
    current_dataset_version = "V00"
    new_dataset_version = "V0"




    image_paths = glob.glob(os.path.join("datasets", "segmented", "multiclass", "train2", current_dataset_version, "images","*"))
    # image_paths = glob.glob(os.path.join("datasets", "feature_matching_test","*"))

    
    output_image_path = os.path.join("datasets", "segmented", "multiclass", "train2", new_dataset_version, "images")
    output_mask_path = os.path.join("datasets", "segmented", "multiclass", "train2", new_dataset_version, "labels")
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_mask_path, exist_ok=True)
    input_size = 518
    label_map =["Background","Mud","Sand","Shellhash coverage","Dog Cockle Bed","PatchesWorms","Bryozoans"]
    NUM_CLASSES = len(label_map)
    id2label = {i: label_map[i] for i in range(NUM_CLASSES)}
    label2id = {v: k for k, v in id2label.items()}
    batch_size = 4
    model_path = os.path.join("trained_models", "multiclass", model_name)
    SEGFORMER_MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME,
                                                    reduce_labels=False,
                                                    do_resize=False, 
                                                    do_normalize=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(device)
    upsampler.eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    existing_results = glob.glob(os.path.join(output_mask_path, "*"))
    existing_results = [os.path.basename(c) for c in existing_results]
    for indexx, path in enumerate(image_paths):
        image_name =  os.path.basename(path)
        if image_name not in existing_results:
            image ,logits, encoder_feats,seg_features = get_logits_and_features(path, input_size,device,upsampler,model)
            height,width = image.shape[:2]

            affinity_map = compute_affinity_from_features(encoder_feats)
            affinity_map = F.interpolate(affinity_map.permute(0, 3, 1, 2), size=(518, 518), mode='nearest')  # [1, 8, 518, 518]
            affinity_map = affinity_map.permute(0, 2, 3, 1)  # back to [1, 518, 518, 8]
            encoder_feats = F.interpolate(encoder_feats, size=(518, 518), mode='nearest')  
            seg_features = F.interpolate(seg_features, size=(518, 518), mode='nearest')  
            probs = torch.softmax(logits, dim=1)
            seeds = make_seeds(probs, tau_high=0.90, min_margin=0.15)   # (1,H,W)
            refined_mask = expand_with_feats(
                probs=probs, seeds=seeds, feats=seg_features, affinity=affinity_map,img_org=image,
            )
            refined_mask = refined_mask[0].detach().cpu().numpy().astype(np.uint8)
            ignore_mask = (refined_mask==255)
            refined_mask[ignore_mask] = 0
            C = NUM_CLASSES
            H, W = refined_mask.shape
            device = affinity_map.device if isinstance(affinity_map, torch.Tensor) else "cuda"
            mask_t = torch.from_numpy(refined_mask).long().to(device)          # (H,W)
            Y0 = F.one_hot(mask_t, num_classes=C).permute(2,0,1).unsqueeze(0)  # (1,C,H,W)
            Y0 = Y0.float()
            refined_mask = random_walk_refine(Y0, affinity_map)
            refined_mask = refined_mask.argmax(dim=1)[0].to(torch.uint8).detach().cpu().numpy()  # (H,W)
            mask_uint8 = ignore_mask.astype(np.uint8) * 255
            blurred = cv2.GaussianBlur(mask_uint8, (5,5), 0)  # or cv2.blur
            ignore_mask = (blurred > 127)  
            refined_mask[ignore_mask] = 255

            # probs = torch.softmax(logits, dim=1)
            # refined_mask = logits.argmax(dim=1)[0].to(torch.uint8).detach().cpu().numpy()  # (H,W)
            if DEBUG:
                img_org = cv2.resize(image, (refined_mask.shape[0],refined_mask.shape[1]))
                mask_vis = refined_mask.copy()
                mask_rgb = np.zeros((*mask_vis.shape,3),dtype=np.uint8)
                for k,c in COLOR_MAP_INDEX.items(): mask_rgb[mask_vis==k]=c
                overlay = cv2.addWeighted(img_org,0.6,mask_rgb,0.4,0)
                cv2.imshow("Initial Seed", overlay)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            

            refined_mask = cv2.resize(refined_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            # # cv2.imwrite(os.path.join(output_image_path, os.path.basename(path)), image)
            cv2.imwrite(os.path.join(output_mask_path, os.path.basename(path)), (refined_mask*30).astype(np.uint8))

        
        
        

         


