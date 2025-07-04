from utils.DinoAnnotator import *
import argparse
import glob
import random
random.seed(42)

def feature_matching (img_paths, image_output_path,label_output_path):
    MatchingModule = DinoFeatureMatching(refine_mask_auto = False,
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

