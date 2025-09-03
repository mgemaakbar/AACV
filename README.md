All of the code in this repository are tested on Google Colab.

Refer to the original documentation of the dataset: https://www.nuscenes.org/nuimages. Also download the dataset in the official site.

detr_category_only_as_class.ipynb is not used anymore. The reason it is not deleted is just to serve as an evidence that we made an effort on it.

If you want to train the model with category-only (example: 'vehicle.car') class, change the class_type in this part of the notebook:

train_df, val_df, class_list = data_preparation.get_df_and_class_list('/content/', 100, 100, train_random_sample=False, val_random_sample=False, class_type = "category_only")

If you want attribute-only as class (example: 'vehicle.moving'):

train_df, val_df, class_list = data_preparation.get_df_and_class_list('/content/', 100, 100, train_random_sample=False, val_random_sample=False, class_type = "attribute_only")

If you want Category concatenated with attribute (example: 'vehicle.car+vehicle.moving') as class:

train_df, val_df, class_list = data_preparation.get_df_and_class_list('/content0', 100, 100, train_random_sample=False, val_random_sample=False, class_type = "category_and_attribute")