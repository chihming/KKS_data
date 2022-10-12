# KKStream Data

DataPreprocess will filter the interaction data by its **item type**, **interaction type**, **active user threshold**, and will split the data into training and testing data by **client_upload_timestamp**. Besides, it will filter the metadata by its **branch type**, **relation type**.

To run the preprocessor, just run the command `python main.py`. You can directly edit the path or the filter conditions in *main.py*.

### TODO:
- file writing : write interaction data and metadata into txt file.