import numpy as np
from PIL import Image
import cv2
from collections import Counter

def convert_image(image_input):
    """
    Converts a PIL Image to an OpenCV image or vice versa.

    :param image_input: PIL Image or OpenCV image (numpy array).
    :return: Converted image (OpenCV image if input was PIL, PIL Image if input was OpenCV).
    """
    if isinstance(image_input, Image.Image):
        # Convert PIL Image to OpenCV Image
        image_output = np.array(image_input)  # Convert PIL Image to numpy array
        image_output = cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        return image_output
    elif isinstance(image_input, np.ndarray):
        # Convert OpenCV Image to PIL Image
        image_output = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image_output = Image.fromarray(image_output)  # Convert numpy array to PIL Image
        return image_output
    else:
        raise TypeError("input is not a PIL.Image.Image or cv2 image")

def resize_for_stable_diffusion(image, use_nearest=False):
    width, height = image.size

    # Ensure at least one dimension is larger than 512
    if width <= 512 and height <= 512:
        if width > height:
            width = max(512, (width // 8) * 8)
            height = int((width / image.width) * image.height)
        else:
            height = max(512, (height // 8) * 8)
            width = int((height / image.height) * image.width)

    # Resize the image if it exceeds the maximum dimension of 1024
    if max(width, height) > 1024:
        if width > height:
            width = 1024
            height = int((width / image.width) * image.height)
        else:
            height = 1024
            width = int((height / image.height) * image.width)

    # Make both dimensions multiples of 8
    width = (width // 8) * 8
    height = (height // 8) * 8

    # Resize the image if necessary
    if width != image.width or height != image.height:
        resizer = Image.NEAREST if use_nearest else Image.LANCZOS
        resized_img = image.resize((width, height), resizer)
    else:
        resized_img = image

    return resized_img

def object_id2class_name(obj_id, objects):
    for obj in objects:
        if obj["object_id"] == obj_id:
            return obj["class_name"]

def grep_predicates(obj_id, predicates, objects):
    predicate = []
    rel_predicate = []
    for pred in predicates:
        if pred["subject_id"] == obj_id and pred["predicate_name"] not in ["N/A", "Unknown"]:
            if pred["object_id"] == -1:
                predicate.append(pred["predicate_name"])
            else:
                rel_predicate.append([pred["predicate_name"], object_id2class_name(pred["object_id"], objects)])
    return predicate, rel_predicate

def get_caption(scene_graph):
    objects = scene_graph['objects']
    predicates = scene_graph['predicates']
    for obj in objects:
        obj_id = obj["object_id"]
        predicate, rel_predicate = grep_predicates(obj_id, predicates, objects)
        obj["predicate"], obj["rel_predicate"] = predicate, rel_predicate

def binarize_mask(mask):
    mask = np.array(mask, dtype=np.uint8)
    mask = np.where(mask > 100, 255, 0)
    mask = np.array(mask, dtype=np.uint8)
    mask = Image.fromarray(mask)
    return mask

def calculate_overlap_area(mask, bbox):
    """
    Calculate the area of overlapping region given a mask and four coordinates of a bounding box.
    :param mask: PIL Image object of the mask.
    :param bbox: Dictionary containing 'x', 'y', 'width', and 'height' keys.
    :return: Area of overlapping region.
    """
    # Convert mask to numpy array
    mask = np.array(mask)

    # Get coordinates of mask within bounding box
    x1, y1, x2, y2 = bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']
    mask_x1 = max(0, x1)
    mask_y1 = max(0, y1)
    mask_x2 = min(mask.shape[1], x2)
    mask_y2 = min(mask.shape[0], y2)
    # Calculate area of overlapping region
    overlap_area = np.sum(mask[mask_y1:mask_y2, mask_x1:mask_x2] > 0)
    return overlap_area

def stringify_list(l, use_and=True):
    if len(l) == 0:
        return ""
    elif len(l) == 1:
        return l[0]
    else:
        if use_and:
            return ", ".join(l[:-1]) + " and " + l[-1]
        else:
            return ", ".join(l)

def relative_stringify_list(l, class_name):
    cap = class_name + " is "
    for l_ in l:
        cap = cap + l_[0] + " to " + l_[1] + ", "
    cap = cap[:-2] + ". "
    return cap

def refine_caption(scene_graph, mask):
    default_caption = "a natural and realistic image of "
    objects = scene_graph['objects']
    object_class_counter = Counter([obj["class_name"] for obj in objects])
    contained_objects = []
    masked_region = np.sum(np.array(mask) > 0)
    for obj in objects:
        overlap_area = calculate_overlap_area(mask, obj['object_bbox'])
        obj['overlap_area'] = overlap_area
        if overlap_area / masked_region > 0.5:
            contained_objects.append(obj)
    contained_objects.sort(key=lambda x: x['overlap_area'], reverse=True)
    if len(contained_objects) > 0:
        obj_captions = ""
        for obj in contained_objects:
            obj_caption = ""
            if object_class_counter[obj["class_name"]] > 1:
                obj_caption = f"{object_class_counter[obj['class_name']]} {obj['class_name']}s, "
            if obj["predicate"] != []:
                obj_caption = obj_caption + obj["class_name"] + " " + stringify_list(obj["predicate"]) + ", "
            if obj["rel_predicate"] != []:
                obj_caption = obj_caption + relative_stringify_list(obj["rel_predicate"], obj["class_name"]) + ", "
            obj_captions = obj_captions + obj_caption
        default_caption = default_caption + obj_captions
    return default_caption