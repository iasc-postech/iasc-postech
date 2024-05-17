import os
import random
from tqdm import tqdm
import json

# shuffle train and valid and re-divide them to include 10,000 images for valid 
root_folder_name = "modified_coco_0.1_0.6"

train_data_dir = f"{root_folder_name}/train2017"
valid_data_dir = f"{root_folder_name}/val2017"

with open(os.path.join(train_data_dir, "modified_data.json"), "r") as f:
    train_json = json.load(f)
with open(os.path.join(valid_data_dir, "modified_data.json"), "r") as f:
    valid_json = json.load(f)

full_file_list = train_json + valid_json
new_valid_json = random.sample(full_file_list, 10000)
new_train_json = [i for i in full_file_list if i not in new_valid_json]


folders = ["image512", "input", "mask512", "predict_panoptic_seg", "score_obj_quries_bboxes"]

for i in tqdm(new_valid_json):
    if i in valid_json:
        continue
    else:
        for folder in folders:
            extension = "npz" if folder in ["predict_panoptic_seg", "score_obj_quries_bboxes"] else "png"
            os.rename(os.path.join(train_data_dir, folder, i["score_obj_quries_bboxes"] + "." + extension), os.path.join(valid_data_dir, folder, i["score_obj_quries_bboxes"] + "." + extension))

for i in tqdm(new_train_json):
    if i in train_json:
        continue
    else:
        for folder in folders:
            extension = "npz" if folder in ["predict_panoptic_seg", "score_obj_quries_bboxes"] else "png"
            os.rename(os.path.join(valid_data_dir, folder, i["score_obj_quries_bboxes"] + "." + extension), os.path.join(train_data_dir, folder, i["score_obj_quries_bboxes"] + "." + extension))

with open(os.path.join(valid_data_dir, "modified_data.json"), "w") as f:
    json.dump(new_valid_json, f)


with open(os.path.join(train_data_dir, "modified_data.json"), "w") as f:
    json.dump(new_train_json, f)
