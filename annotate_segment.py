from utils.DinoAnnotator import *
import argparse
import glob
import random
from tqdm import tqdm
from pathlib import Path
SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   

def feature_matching (img_paths, image_output_path,label_output_path):
    MatchingModule = DinoFeatureMatching(refine_mask_auto = True,
                                         refine_mask_manual = True,
                                         superpixel="felzenszwalb",
                                         debug = False,
                                         show_scale_percentage=50)
    for img_path in img_paths:
        existing_annotations = glob.glob(os.path.join(label_output_path, "*"))
        existing_annotations = [os.path.basename(path) for path in existing_annotations]
        print("total number of annotated images: ", len(existing_annotations))
        name = os.path.basename(img_path)
        img_org = cv2.imread(img_path)
        if name not in existing_annotations:
            mask = MatchingModule.run(img_path, roi=[])
            cv2.imwrite(os.path.join(image_output_path, name),img_org)
            cv2.imwrite(os.path.join(label_output_path, name),mask)
    print("all done")



def multiclass_creation(model_paths, img_paths, image_output_path,label_output_path, from_binary = True):
    if from_binary:
        MultiClassModule = DinoSegmentorMultiClassFromBinary(
            model_paths = model_paths,
            show_scale_percentage =50,
            refine_mask_auto = False,
            refine_mask_manual = False,
            superpixel="felzenszwalb",
            debug = False,
            thresholds=[0.5 for i in range(6)],
            class_order= ["Mud", "Sand", "Shellhash coverage", "Dog Cockle Bed", "PatchesWorms", "Bryozoans"]
        )
        order_dict = {
            "Mud": ["Sand", "Shellhash coverage","Mud", "Dog Cockle Bed", "PatchesWorms", "Bryozoans"],
            "Sand": ["Mud", "Shellhash coverage","Sand", "Dog Cockle Bed", "PatchesWorms", "Bryozoans"],
            "Shellhash coverage": ["Sand", "Mud","Shellhash coverage", "Dog Cockle Bed", "PatchesWorms", "Bryozoans"],
            "Dog Cockle Bed": ["Mud", "Sand", "Shellhash coverage", "Dog Cockle Bed", "PatchesWorms", "Bryozoans"],
            "PatchesWorms": ["Mud", "Sand", "Shellhash coverage", "Dog Cockle Bed", "PatchesWorms", "Bryozoans"],
            "Bryozoans": ["Sand", "Shellhash coverage","Mud", "Dog Cockle Bed", "PatchesWorms", "Bryozoans"],
        }
        for img_path in tqdm(img_paths):
            name = os.path.basename(img_path)
            folder_name = Path(img_path).parent.name
            MultiClassModule.class_order = order_dict[folder_name]
            existing_annotations = glob.glob(os.path.join(label_output_path, "*"))
            existing_annotations = [os.path.basename(path) for path in existing_annotations]
            # print("total number of annotated images: ", len(existing_annotations))
            img_org = cv2.imread(img_path)
            if name not in existing_annotations:
                mask = MultiClassModule.run(img_path)
                mask=(mask*30).astype(np.uint8)
                cv2.imwrite(os.path.join(image_output_path, name),img_org)
                cv2.imwrite(os.path.join(label_output_path, name),mask)
        print("all done")
    else:
        model_path = model_paths_dict["all"]
        label_map =["Background","Mud","Sand","Shellhash coverage","Dog Cockle Bed","PatchesWorms","Bryozoans"]
        MultiClassModule = DinoSegmentorMultiClassRefiner(
            label_map = label_map,
            model_path = model_path,
            input_size=518,
            predict_mode = "argmax", # argmax or sigmoid, when sigmoid is selected thresholds will be used
            show_scale_percentage = 50,
            refine_mask_auto = False,
            superpixel="felzenszwalb",
            refine_mask_manual = False,
            debug = False,
            thresholds = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]
            )
        for img_path in tqdm(img_paths):
            name = os.path.basename(img_path)
            existing_annotations = glob.glob(os.path.join(label_output_path, "*"))
            existing_annotations = [os.path.basename(path) for path in existing_annotations]
            # print("total number of annotated images: ", len(existing_annotations))
            img_org = cv2.imread(img_path)
            if name not in existing_annotations:
                mask = MultiClassModule.run(img_path)
                mask=(mask*30).astype(np.uint8)
                cv2.imwrite(os.path.join(image_output_path, name),img_org)
                cv2.imwrite(os.path.join(label_output_path, name),mask)
        print("all done")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate a seafloor-sediment image for phase-1 and phase-2.")
    parser.add_argument("image_paths", type=str, help="Path to the images to annotate")
    parser.add_argument("output_path", type=str, help="Path to save the annotation")
    parser.add_argument("mode", type=str, help="Select the type of the annotation")
    args = parser.parse_args()
    image_dir = args.image_paths
    output_dir = args.output_path
    mode = args.mode
    dataset_path = args.image_paths
    img_paths = glob.glob(os.path.join(dataset_path, "*"))
    output_path = args.output_path
    image_output_path =  os.path.join(output_path, "images")
    label_output_path =  os.path.join(output_path, "labels")
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)
    if args.mode == "feature_matching":
        feature_matching (img_paths, image_output_path,label_output_path)
    elif args.mode == "multiclass_generation_from_binary":
        model_folders = os.path.join("trained_models", "binary")
        model_folders = glob.glob(os.path.join(model_folders, "*"))
        class_names = [os.path.basename(c) for c in model_folders]
        model_paths_dict = {}
        for index, name in enumerate(class_names):
            model_paths_dict.update({name: os.path.join(model_folders[index], "dinov2_segmentor_v2.pth")})
        multiclass_creation(model_paths_dict,img_paths, image_output_path,label_output_path,from_binary = True)
    elif args.mode == "multiclass_refinement":
        model_folder = os.path.join("trained_models", "multiclass")
        model_paths_dict = {}
        model_paths_dict.update({"all": os.path.join(model_folder, "DINOv2SegHeadV2DatasetV0.pth")})
        multiclass_creation(model_paths_dict,img_paths, image_output_path,label_output_path,from_binary = False)

    else:
        print("mode should be either feature)matching for phase-1 or multiclass_generation_from_binary/multiclass_refinement for phase-2")

