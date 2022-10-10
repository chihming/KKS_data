# KKStream Data

DataPreprocess will filter the interaction data by its **item type**, **interaction type**, **active user threshold**, and will split the data into training and testing data by **client_upload_timestamp**.

To run the preprocessor, just run the command `python main.py`. You can directly edit the path or the filter conditions in *main.py*.

### TODO:
- metadata preprocessing : convert to the triples form to fit in the RecSys