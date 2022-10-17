# KKStream Data

## Data Preprocessing
DataPreprocessor will filter the interaction data by its **item type**, **interaction type**, **active user threshold**, and will split the data into training and testing data by **client_upload_timestamp**. Besides, it will filter the metadata by its **branch type**, **relation type**.
```
usage: main.py [-h] [-interaction_paths INTERACTION_PATHS [INTERACTION_PATHS ...]] [-meta_dir_path META_DIR_PATH] [-item_type ITEM_TYPE]
               [-select_interaction_type SELECT_INTERACTION_TYPE [SELECT_INTERACTION_TYPE ...]] [-interaction_thre INTERACTION_THRE] [-select_branch_type SELECT_BRANCH_TYPE [SELECT_BRANCH_TYPE ...]]
               [-select_relation_type SELECT_RELATION_TYPE [SELECT_RELATION_TYPE ...]]

Parameters for the script.

optional arguments:
  -h, --help            show this help message and exit
  -interaction_paths INTERACTION_PATHS [INTERACTION_PATHS ...]
                        interaction log csv(s)
  -meta_dir_path META_DIR_PATH
                        directory to meta
  -item_type ITEM_TYPE  target item type
  -select_interaction_type SELECT_INTERACTION_TYPE [SELECT_INTERACTION_TYPE ...]
                        types to be selected for train/test
  -interaction_thre INTERACTION_THRE
                        threshold for selecting users
  -select_branch_type SELECT_BRANCH_TYPE [SELECT_BRANCH_TYPE ...]
                        types to be selected for train/test
  -select_relation_type SELECT_RELATION_TYPE [SELECT_RELATION_TYPE ...]
                        types to be selected for train/test
```
Example Command:
```
python3 main.py \
-interaction_paths data/16b9973c-3556-40be-81af-89efa792a880.csv data/36d1641e-1ee9-466a-8e10-b1b0ca2b4f98.csv
-meta_dir_path data/metadata_1656673401
```



## File Writing
FileWriter will write the output file based on the given training_df, testing_df, kg_data.  

To write out interaction data, you can choose the format to be *triple* (user, interaction, item), *tuple* (user, item), or *userwise* (user, item1, item2, ...). In addition, you can decide whether to do remapping, which is default to be **True**. If you decide to do remapping, it will also write out the classes of encoder in `.npy` format. To load the classes, use the codes below:
```
from sklearn.preprocessing import LabelEncoder
import numpy as np
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy')
```

To write out kg data, you can decide whether to do remapping, which is default to be **False**. If you decide to do remapping, it will also write out the classes of encoder in `.npy` format.

## Running
Just run the command `python main.py` to do data preprocessing and file writing. You can directly edit the path or the conditions in `main.py`.
