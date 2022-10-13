# KKStream Data

## Data Preprocessing
DataPreprocessor will filter the interaction data by its **item type**, **interaction type**, **active user threshold**, and will split the data into training and testing data by **client_upload_timestamp**. Besides, it will filter the metadata by its **branch type**, **relation type**.

To check the detail, please see the file *data_preprocessor.py*.

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
