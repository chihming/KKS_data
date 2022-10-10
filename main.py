from data_preprocessor import DataPreprocessor
import os

base = os.getcwd()
interaction_data_1_path = os.path.join(base, "interaction_data/16b9973c-3556-40be-81af-89efa792a880.csv")
interaction_data_2_path = os.path.join(base, "interaction_data/36d1641e-1ee9-466a-8e10-b1b0ca2b4f98.csv")

data_preprocessor = DataPreprocessor(interaction_data_1_path, interaction_data_2_path)

item_type = 'VOD'
select_interaction_type = ['click', 'play']
interaction_thre = 10
training_df, testing_df = data_preprocessor.filter_data(item_type, select_interaction_type, interaction_thre)
print(len(training_df), len(testing_df))