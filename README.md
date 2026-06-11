All of the code in this repository are tested on Google Colab. Remember that if you try to run it on a different environment with different version of libraries installed, it might not guarantee compatibility with the code and result in errors. 

Note that the data preparation part of the code requires very few dependencies, so the data preparation would most likely work just fine -- this reminder applies to the training part of the code which was adopted from the links included in the notebooks.

Refer to the original documentation of the dataset: https://www.nuscenes.org/nuimages. Also download the dataset which is stored in ```.tgz``` file (as seen in the notebooks) in the official site. Use Google Drive to store the dataset so you can use it in the Python notebooks.

You need to untar the ```tgz``` as seen in the notebooks. After you uncompress the metadata (```nuimages-v1.0-all-metadata.tgz```), it will have this directory structure:

content/
├── v1.0-test/
├── v1.0-train/
└── v1.0-val/

The code ```data_preparation.get_df_and_class_list('/content/'...``` assumes that the directory structure to be the above (since the parameter ```root_dir_path``` is set to ```/content/```).

There are 3 python notebooks: ```detectron2.ipynb```, ```detr_with_detectron2.ipynb```, ```yolo.ipynb```. All of them expect you to import the ```data_preparation.py``` which is asssumed to be in ```/content/```. All of them train an object detection model with NuImages.

```detr_category_only_as_class.ipynb``` is not used anymore, so you can ignore it. The reason it is not deleted is just to serve as an evidence that we made an effort on it.

(see the dataset docs to know about what Category and Attribute mean)
If you want to train the model with category-only (example: ```vehicle.car```) class, change the class_type in this part of the notebook:

```train_df, val_df, class_list = data_preparation.get_df_and_class_list('/content/', 100, 100, train_random_sample=False, val_random_sample=False, class_type = "category_only")```

If you want attribute-only as class (example: ```vehicle.moving```):

```train_df, val_df, class_list = data_preparation.get_df_and_class_list('/content/', 100, 100, train_random_sample=False, val_random_sample=False, class_type = "attribute_only")```

If you want Category concatenated with attribute (example: ```vehicle.car+vehicle.moving```) as class:

```train_df, val_df, class_list = data_preparation.get_df_and_class_list('/content/', 100, 100, train_random_sample=False, val_random_sample=False, class_type = "category_and_attribute")```

A part of ```multi_label_NMS_yolo_inference.ipynb``` which contains a modified code from Ultralytics is licensed with AGPL-3.0 License. Please respect the license if you wish to use it. ```multi_label_NMS_yolo_inference.ipynb``` is a demonstration of multi-label inference on YOLO. 
