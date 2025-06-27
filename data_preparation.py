import json
import pandas as pd
import datetime
import yaml
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import random

class InlineList(list): pass

def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def create_data_yml(path_to_output, class_list):
    yml_data = {
        'train': '../train/images',
        'val': '../val/images',
        "nc": len(class_list),
        "names": InlineList(class_list)
    }
    yaml.add_representer(InlineList, represent_inline_list)
    with open(path_to_output + 'data.yml', 'w') as outfile:
        yaml.dump(yml_data, outfile,default_flow_style=False, sort_keys=False)


def data_json_to_joined_df_and_class_list(path_prefix_to_json, percent = 100, random_sample=False, class_type = "category_only"):
    # percent: take only certain percent of the images
    # random_sample: True if you want to randomize which images to take

    category = pd.read_json(path_prefix_to_json + 'category.json').rename(columns={"name": "category_name"})
    category = category.sort_values(by='category_name')
    category['category_id'] = range(len(category))

    obj_ann = pd.read_json(path_prefix_to_json + 'object_ann.json')
    sample_data = pd.read_json(path_prefix_to_json + 'sample_data.json')
    sample_data = sample_data[sample_data['filename'].str.startswith('sample')] # take only keyframe images (annotated images) which always start with 'samples'. sweeps images are not annotated
    n_rows = int(len(sample_data) * (percent / 100))
    if random_sample:
        sample_data = sample_data.sample(n = n_rows)
    else:
        sample_data = sample_data.iloc[:n_rows]

    attribute = pd.read_json(path_prefix_to_json + 'attribute.json').rename(columns={"name": "attribute_name"})
    attribute = attribute.sort_values(by='attribute_name')

    #object annotation inner join category
    merged = obj_ann.merge(
        category,
        how = 'inner',
        left_on='category_token',
        right_on='token',
        suffixes=('_from_obj_ann', '_from_category')
    )

    attr_token_to_attr_name = dict(zip(attribute['token'], attribute['attribute_name']))

    def get_attribute_only_class_name(df_row):
        attr_names = list(map(attr_token_to_attr_name.get, df_row['attribute_tokens'])) # convert attr tokens to attr name
        if len(attr_names) == 0: # not all objects have attributes. if we don't do this then it will return an empty string
          return "no_attribute"
        return "+".join(attr_names)

    def get_category_and_attribute_class_name(df_row):
        attr_names = list(map(attr_token_to_attr_name.get, df_row['attribute_tokens'])) # convert attr tokens to attr name
        return "+".join([df_row['category_name']] + attr_names)

    # attribute_tokens type is a list of string. an object can have at most 2 attributes, can also be 0
    # sorting the list of tokens is to make sure (attr_1, attr_2) is the same with (attr_2, attr_1). we don't want those to be treated as two different things
    merged['attribute_tokens'] = merged['attribute_tokens'].apply(sorted)

    # for determining class name of the object
    if class_type == "category_only":
      merged['class_name'] = merged['category_name'] # class = category name
    elif class_type == "attribute_only":
      merged['class_name'] = merged.apply(get_attribute_only_class_name, axis = 1) # class = attribute name
    elif class_type == "category_and_attribute":
      merged['class_name'] = merged.apply(get_category_and_attribute_class_name, axis = 1)

    class_name_list = merged['class_name'].unique()
    class_name_list.sort()

    # object annotation inner join sample_data
    merged = merged.merge(
        sample_data,
        how = 'inner',
        right_on='token',
        left_on='sample_data_token',
        suffixes=('_from_first_merged', '_from_sample_data')
    )

    return merged, class_name_list

def has_content(file_path): # return True if the file exist and has content, False if file is empty or does not exist
    try:
        with open(file_path, 'r') as file:
            return bool(file.read().strip())
    except FileNotFoundError:
        return False


# create list of train/val/test image file name list in a txt file (train/val.txt), so we can use cp/rsync with it
# NOTE: make sure that you run this only once! because if you run this more than once (there is already an existing train.txt and val.txt) this code will keep appending to the txt files, leading to duplicates filenames in the txt which might show up as an error during cp like:
# cp: cannot create regular file '/root/val2017/n003-2018-01-05-15-22-31+0800__CAM_FRONT__1515137385231616.jpg': Permission denied
def create_image_filename_list_txt(data_split, merged_df):
    if data_split not in ['train', 'val', 'test']:
        return -1

    # validation to force us to start clean
    if has_content(data_split+".txt"):
        print("The text file (train/val.txt) already exists and is already filled. Remove the txt file first to make sure we start clean.")
        return

    df = merged_df[['filename']]
    filenames = set([])
    for index, row in df.iterrows():
        filenames.add(row['filename'])
    for fn in filenames:
        with open(data_split+".txt", 'a') as f:
            f.write(fn + "\n")

def df_to_coco_format(merged_df, split, class_name_list): # DETR uses COCO format. it can be used for detectron2 too
    if split not in ['val', 'train']:
        return -1

    df = merged_df[['filename', 'bbox', 'class_name', 'category_name', 'width', 'height']]

    coco_dict = {"images": [], "annotations": [], "categories": []}
    coco_dict["info"] = { # not including this will cause error during training
        "description": "Nothing",
        "url": "https://github.com/nobody/nothing",
        "version": "2.0.2",
        "year": 2025,
        "contributor": "nobody",
        "date_created": datetime.datetime.utcnow().isoformat(" "),
    }

    is_image_added = {}
    filename_to_id = {}

    id = 0

    # build images
    for i, row in df.iterrows():
        if row['filename'] not in is_image_added: # if image not added yet, then add
            coco_dict["images"].append({
                "id": id,
                "width": row['width'],
                "height": row['height'],
                "file_name": row['filename'].split("/")[-1],
            })
            filename_to_id[row['filename']] = id
            id += 1
            is_image_added[row['filename']] = True


    # build category/class (# don't confuse this 'category' with NuScene Category schema)
    class_name_list.sort()
    class_name_to_class_id = {class_name_list[i]:i for i in range(len(class_name_list))} 
    for i in range(len(class_name_list)):
      coco_dict["categories"].append({
            "id": i,
            "name": class_name_list[i],
      })


    # build annotations
    for i, row in df.iterrows():
        xmin, ymin, xmax, ymax = row['bbox'][0], row['bbox'][1], row['bbox'][2], row['bbox'][3]
        coco_dict["annotations"].append({
            "id": i,
            "image_id": filename_to_id[row['filename']],
            "category_id": class_name_to_class_id[row['class_name']], # don't confuse this category with NuScene Category schema
            "area": (xmax - xmin) * (ymax - ymin),
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin], # top left, width, height
            "segmentation": [],
            "iscrowd": 0, # not including this will cause an error during training
            "date_captured": datetime.datetime.utcnow().isoformat(" "),
            "license": 0,
            "flickr_url": "",
            "coco_url": ""
        })


    with open(split + ".json", "w") as file:
        file.write(json.dumps(coco_dict))


def display_image_from_coco_json(ann_file, img_dir):
    coco = COCO(ann_file)

    imgIds = coco.getImgIds()
    img_id = imgIds[random.randint(1, len(imgIds)-1)]
    img_info = coco.loadImgs(img_id)[0]
    print("File name: ", img_info['file_name'])
    img_path = os.path.join(img_dir, img_info['file_name'])
    print("Image path: ", img_path)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    dpi = 100
    height, width = img.shape[:2]
    figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    ax.imshow(img)

    for ann in anns:
        x, y, w, h = ann['bbox']
        category_id = ann['category_id']
        category_name = coco.loadCats(category_id)[0]['name']

        # Draw rectangle
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Draw category label
        ax.text(x, y - 5, category_name, color='red', fontsize=10,
                backgroundcolor='white', verticalalignment='bottom')

    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


def bbox_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    box_width = xmax - xmin
    box_height = ymax - ymin

    x_center = xmin + box_width / 2
    y_center = ymin + box_height / 2

    # Normalize by image size
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = box_width / image_width
    height_norm = box_height / image_height

    return x_center_norm, y_center_norm, width_norm, height_norm

# create yolo label txt and yaml
def df_to_yolo_format_txt(path_prefix, data_split, class_name_list, merged):
    if data_split not in ['train', 'val', 'test']:
        return -1
    
    df = merged[['bbox', 'class_name', 'category_name', 'filename', 'width',  'height']]
    
    class_name_list.sort()
    class_name_to_class_id = {class_name_list[i]: i for i in range(len(class_name_list))}
    
    for index, row in df.iterrows():
        filename = row['filename'].split("/")[-1] # filename column is in the format of samples/CAM_BACK/n010-2018-08-27-16-15-24+0800...
        abs_path_to_label_txt = path_prefix + data_split + "/labels/" + filename[:len(filename)-4] + '.txt' # example: /content/train/label/asdf.txt. remove the .jpg from asdf.jpg first
        x_center, y_center, width, height = bbox_to_yolo(row['bbox'][0], row['bbox'][1], row['bbox'][2], row['bbox'][3], row['width'], row['height'])
        with open(abs_path_to_label_txt, 'a') as f: # append label to the txt
            f.write(str(class_name_to_class_id[row['class_name']]) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height) + "\n")

