from dinov3_utils import *
SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   

DATASET_PATH = "workshop"
os.makedirs(DATASET_PATH, exist_ok=True)
PATCH_SIZE = 16
IMAGE_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_str = "cuda" if torch.cuda.is_available() else "cpu"
REPO_DIR = "dinov3"
IMG_PATH = os.path.join(DATASET_PATH,"inputs","13rgb.png")
DEBUG = True
OUTPUT = {}
OUTPUTSIZE = IMAGE_SIZE//2
SHOWSCALE = 50
OUTPUT_PATH = os.path.join(DATASET_PATH, "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

if __name__ == "__main__":
    model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights="dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    model.to(DEVICE)
    image = Image.open(IMG_PATH).convert("RGB")
    image_resized = resize_transform(image,IMAGE_SIZE,PATCH_SIZE)
    image_resized_norm = TF.normalize(image_resized, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    with torch.inference_mode():
        with torch.autocast(device_type=DEVICE_str, dtype=torch.float32):
            feats = model.get_intermediate_layers(image_resized_norm.unsqueeze(0).to(DEVICE), n=range(24), reshape=True, norm=True)
            x = feats[-1].squeeze().detach().cpu()
            x = x.view(x.shape[0], -1)
            x = x.permute(1, 0)
    h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]
    x_grid = x.view(h_patches, w_patches, x.shape[1]).permute(2,0,1).unsqueeze(0)  # [1,C,Hp,Wp]

    
    xgrid_show = x_grid[0].detach().cpu().numpy()
    xgrid_show = np.mean(xgrid_show,axis=0)
    xgrid_show = cv2.normalize(xgrid_show, None, 0, 255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    xgrid_show = cv2.resize(xgrid_show, (IMAGE_SIZE,IMAGE_SIZE))
    xgrid_show = cv2.applyColorMap(xgrid_show, cv2.COLORMAP_INFERNO)
    OUTPUT.update({"mean_features": xgrid_show})


    # PCA FEATURE RGB 
    feat_rgb = get_rgb_feature_map(x_grid,input_size =IMAGE_SIZE,show_scale = SHOWSCALE, debug=DEBUG)
    OUTPUT.update({"feat_rgb": feat_rgb})

    ##################### MASK GENERATION USING POINTS #####################
    # DINOv3
    sim_map_points_rgb,sim_map_points,points = get_sim_map_points(IMG_PATH,x_grid, IMAGE_SIZE,PATCH_SIZE,SHOWSCALE,DEBUG,cmap=cv2.COLORMAP_INFERNO)
    overlay_heatmap_points = get_overlay_heatmap(sim_map_points_rgb,cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)),DEBUG,SHOWSCALE)
    OUTPUT.update({"sim_map_points_rgb": sim_map_points_rgb})
    OUTPUT.update({"overlay_heatmap_points": overlay_heatmap_points})
    mask_points_rgb,mask_points = get_affinity_mask(sim_map_points,x_grid,DEVICE,SHOWSCALE, DEBUG,cv2.COLORMAP_INFERNO)
    overlay_mask_points = get_overlay_heatmap(mask_points_rgb,cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)),DEBUG,SHOWSCALE)
    print(mask_points_rgb[0][0])
    OUTPUT.update({"mask_points_rgb": mask_points_rgb})
    print(mask_points.max())
    OUTPUT.update({"mask_points_binary": (mask_points).astype(np.uint8)})
    OUTPUT.update({"overlay_mask_points": overlay_mask_points})
    # SAM2
    sam_mask_points_rgb,sam_mask_points = get_mask_sam_points (IMG_PATH,points,IMAGE_SIZE,DEVICE,DEBUG,cv2.COLORMAP_INFERNO)
    sam_overlay_mask_points = get_overlay_heatmap(sam_mask_points_rgb,cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)),DEBUG,SHOWSCALE)
    OUTPUT.update({"sam_mask_points_rgb": sam_mask_points_rgb})
    OUTPUT.update({"sam_overlay_mask_points": sam_overlay_mask_points})

    ##################### MASK GENERATION USING BOX #####################
    # DINOv3
    sim_map_box_rgb, sim_map_box,box = get_sim_map_box(IMG_PATH,x_grid, IMAGE_SIZE,PATCH_SIZE,SHOWSCALE,DEBUG,cmap=cv2.COLORMAP_INFERNO)
    overlay_heatmap_box = get_overlay_heatmap(sim_map_box_rgb,cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)),DEBUG,SHOWSCALE)
    OUTPUT.update({"sim_map_box_rgb": sim_map_box_rgb})
    OUTPUT.update({"overlay_heatmap_box": overlay_heatmap_box})
    mask_box_rgb,mask_box = get_affinity_mask(sim_map_box,x_grid,DEVICE,SHOWSCALE,DEBUG,cv2.COLORMAP_INFERNO)
    overlay_mask_box = get_overlay_heatmap(mask_box_rgb,cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)),DEBUG,SHOWSCALE)
    OUTPUT.update({"mask_box_rgb": mask_box_rgb})
    OUTPUT.update({"mask_box_binary": (mask_box).astype(np.uint8)})
    OUTPUT.update({"overlay_mask_box": overlay_mask_box})
    # SAM2
    sam_mask_box_rgb,sam_mask_box = get_mask_sam_box(IMG_PATH,box,IMAGE_SIZE,DEVICE,DEBUG,cv2.COLORMAP_INFERNO)
    sam_overlay_mask_box = get_overlay_heatmap(sam_mask_box_rgb,cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)),DEBUG,SHOWSCALE)
    OUTPUT.update({"sam_mask_box_rgb": sam_mask_box_rgb})
    OUTPUT.update({"sam_overlay_mask_box": sam_overlay_mask_box})

    # SAM2-DINOv3
    sam_points_seeds = np.argwhere(sam_mask_box == 1)
    sim_map_input_points_rgb,sim_map_input_points,_ = get_sim_map_input_points(IMG_PATH,x_grid,sam_points_seeds, IMAGE_SIZE,PATCH_SIZE,SHOWSCALE,DEBUG,cmap=cv2.COLORMAP_INFERNO)
    overlay_heatmap_input_points = get_overlay_heatmap(sim_map_input_points_rgb,cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)),DEBUG,SHOWSCALE)
    OUTPUT.update({"sim_map_input_points_rgb": sim_map_input_points_rgb})
    OUTPUT.update({"overlay_heatmap_input_points": overlay_heatmap_input_points})
    mask_input_points_rgb,mask_input_points = get_affinity_mask(sim_map_input_points,x_grid,DEVICE,SHOWSCALE, DEBUG,cv2.COLORMAP_INFERNO)
    overlay_mask_input_points = get_overlay_heatmap(mask_input_points_rgb,cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)),DEBUG,SHOWSCALE)
    OUTPUT.update({"mask_input_points_rgb": mask_input_points_rgb})
    OUTPUT.update({"overlay_mask_input_points": overlay_mask_input_points})

    for key in OUTPUT.keys():
        if key!= "feat_rgb" and key!="mean_features":
            OUTPUT[key] = cv2.resize(OUTPUT[key], (OUTPUTSIZE,OUTPUTSIZE))
    
    image_bgr = cv2.imread(IMG_PATH)
    OUTPUT.update({"image": cv2.resize(image_bgr, (OUTPUTSIZE,OUTPUTSIZE))})

    image_bgr = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    for point in points:
        cv2.circle(image_bgr, point, 8, (0, 0, 255), -1)
    OUTPUT.update({"image_points": cv2.resize(image_bgr, (OUTPUTSIZE,OUTPUTSIZE))})
    image_bgr = cv2.imread(IMG_PATH)
    image_bgr = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    cv2.rectangle(image_bgr, (box[0],box[1]), (box[2],box[3]), (0, 0, 255), 4)
    OUTPUT.update({"image_box": cv2.resize(image_bgr, (OUTPUTSIZE,OUTPUTSIZE))})
    
    
    

    OUTPUT_PATH_ = os.path.join(OUTPUT_PATH,str(IMAGE_SIZE),os.path.basename(IMG_PATH[:-4]))
    print(OUTPUT_PATH_)
    os.makedirs(OUTPUT_PATH_, exist_ok=True)
    for key in OUTPUT.keys():
        cv2.imwrite(os.path.join(OUTPUT_PATH_, key+".png"),OUTPUT[key])


    print("ALL GOOD")
