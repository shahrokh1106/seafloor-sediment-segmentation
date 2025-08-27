from utils.DinoAnnotator import *
import argparse
import glob
import random
from tqdm import tqdm
from pathlib import Path
import json
SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   

def feature_matching (img_paths, image_output_path,label_output_path,refine_mask_auto_flag=True, refine_mask_manual_flag=True):
    MatchingModule = DinoFeatureMatching(refine_mask_auto = refine_mask_auto_flag,
                                         refine_mask_manual = refine_mask_manual_flag,
                                         superpixel="felzenszwalb",
                                         debug = True,
                                         show_scale_percentage=50)
    print(len(img_paths))
    for img_path in img_paths:
        existing_annotations = glob.glob(os.path.join(label_output_path, "*"))
        existing_annotations = [os.path.basename(path) for path in existing_annotations]
        print("total number of annotated images: ", len(existing_annotations))
        name = os.path.basename(img_path)
        img_org = cv2.imread(img_path)
        if name  not in existing_annotations:
            mask = MatchingModule.run(img_path, roi=[])
            cv2.imwrite(os.path.join(image_output_path, name),img_org)
            cv2.imwrite(os.path.join(label_output_path, name),mask)
    print("all done")



def multiclass_creation(model_paths_dict, img_paths, image_output_path,label_output_path, from_binary = True,refine_mask_auto_flag=True, refine_mask_manual_flag=True,predict_mode="argmax",thresholds= [0.5]*7):
    if from_binary:
        MultiClassModule = DinoSegmentorMultiClassFromBinary(
            model_paths = model_paths_dict,
            show_scale_percentage =50,
            refine_mask_auto = refine_mask_auto_flag,
            refine_mask_manual = refine_mask_manual_flag,
            superpixel="felzenszwalb",
            debug = False,
            thresholds=thresholds,
            class_order= ['Shellhash coverage', 'Dog Cockle Bed', 'Sand', 'PatchesWorms', 'Bryozoans', 'Mud']
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
                # cv2.imwrite(os.path.join(image_output_path, name),img_org)
                cv2.imwrite(os.path.join(label_output_path, name),mask)
        print("all done")
    else:
        model_path = model_paths_dict["all"]
        if "Dinov2Segv1" in model_path: 
            seg_model_name = "Dinov2Segv1"
        elif "Dinov2Segv2" in model_path:
            seg_model_name = "Dinov2Segv2"
        elif "Segformer" in model_path:
            seg_model_name = "Segformer"
        label_map =["Background","Mud","Sand","Shellhash coverage","Dog Cockle Bed","PatchesWorms","Bryozoans"]
        MultiClassModule = DinoSegmentorMultiClassRefiner(
            label_map = label_map,
            model_path = model_path,
            input_size=518,
            predict_mode = predict_mode, # argmax or sigmoid, when sigmoid is selected thresholds will be used
            show_scale_percentage = 50,
            refine_mask_auto = refine_mask_auto_flag,
            superpixel="felzenszwalb",
            refine_mask_manual = refine_mask_manual_flag,
            debug = False,
            thresholds = thresholds,
            seg_model_name = seg_model_name
            )
        for img_path in tqdm(img_paths):
            name = os.path.basename(img_path)
            existing_annotations = glob.glob(os.path.join(label_output_path, "*"))
            existing_annotations = [os.path.basename(path) for path in existing_annotations]
            # print("total number of annotated images: ", len(existing_annotations))
            img_org = cv2.imread(img_path)
            if name not in existing_annotations:
                mask = MultiClassModule.run(img_path)
                if np.sum(mask)!=0:
                    mask=(mask*30).astype(np.uint8)
                    cv2.imwrite(os.path.join(image_output_path, name),img_org)
                    cv2.imwrite(os.path.join(label_output_path, name),mask)
                else:
                    print(name+" was ignored")
        print("all done")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate a seafloor-sediment image for phase-1 and phase-2.")
    parser.add_argument("config_path", type=str, help="Path to the json file")
    args = parser.parse_args()
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = json.load(f)
    dataset_path = config["image_paths"]
    output_path = config["output_path"]
    annotate_mode = config["annotate_mode"]
    multiclass_model_name = config["multiclass_model_name"]
    binary_model_name =config["binary_model_name"]
    refine_mask_auto_flag = config["refine_mask_auto"]
    refine_mask_manual_flag = config["refine_mask_manual"]
    predict_mode = config["predict_mode"]
    img_paths = glob.glob(os.path.join(dataset_path, "*"))
    print(dataset_path)
    # check to not include validation images
    image_names = [os.path.basename(c) for c in img_paths]
    val_paths = glob.glob(os.path.join("datasets", "segmented", "Multiclass", "validation","images", "*"))
    val_names = [os.path.basename(c) for c in val_paths]
    img_paths = [p for p in img_paths if os.path.basename(p) not in val_names]
    # ############################################
    image_output_path =  os.path.join(output_path, "images")
    label_output_path =  os.path.join(output_path, "labels")
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)
    if annotate_mode == "feature_matching":
        feature_matching (img_paths, image_output_path,label_output_path,refine_mask_auto_flag,refine_mask_manual_flag)
    elif annotate_mode == "multiclass_generation_from_binary":
        label_map  = {0:"Background", 1:"Mud", 2:"Sand", 3:"Shellhash coverage", 4:"Dog Cockle Bed", 5:"PatchesWorms", 6:"Bryozoans"}
        class_names = [label_map[i] for i in range(1,7)]
        model_paths_dict = {}
        for index, name in enumerate(class_names):
            model_paths_dict.update({name: os.path.join("trained_models", "binary", name, binary_model_name)})
        multiclass_creation(model_paths_dict,img_paths, image_output_path,label_output_path,from_binary = True,refine_mask_auto_flag=refine_mask_auto_flag,refine_mask_manual_flag=refine_mask_manual_flag,predict_mode=predict_mode)

    elif annotate_mode == "multiclass_refinement":
        model_folder = os.path.join("trained_models", "multiclass")
        model_paths_dict = {}
        model_paths_dict.update({"all": os.path.join(model_folder, multiclass_model_name)})
        if predict_mode=="sigmoid":
            best_thresholds = [0.5]*7
        else:
            best_thresholds = [0.5]*7
        
        multiclass_creation(model_paths_dict,img_paths,
                            image_output_path,
                            label_output_path,
                            from_binary = False,
                            refine_mask_auto_flag=refine_mask_auto_flag,
                            refine_mask_manual_flag=refine_mask_manual_flag,
                            predict_mode = predict_mode,
                            thresholds = best_thresholds)
    else:
        print("mode should be either feature matching for phase-1 or multiclass_generation_from_binary/multiclass_refinement for phase-2")

