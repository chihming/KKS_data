from data_preprocessor import DataPreprocessor
from file_writer import FileWriter
import os

base = os.getcwd()
interaction_data_1_path = os.path.join(base, "interaction_data/16b9973c-3556-40be-81af-89efa792a880.csv")
interaction_data_2_path = os.path.join(base, "interaction_data/36d1641e-1ee9-466a-8e10-b1b0ca2b4f98.csv")
metadata_dir_path = os.path.join(base, "metadata_1656673401")

data_preprocessor = DataPreprocessor(interaction_data_1_path, interaction_data_2_path, metadata_dir_path)

# filter interaction data
item_type = 'VOD'
select_interaction_type = ['click', 'play']
interaction_thre = 10
training_df, testing_df = data_preprocessor.filter_interaction_data(item_type, select_interaction_type, interaction_thre)

# filter metadata
select_branch_type=['movie', 'series', 'season', 'episode']
select_relation_type=['artists', 'genres']
kg_data = data_preprocessor.filter_metadata(select_branch_type, select_relation_type)

# write files
file_writer = FileWriter(training_df, testing_df, kg_data)
file_writer.write_interaction(format_="triple", remap=True)
file_writer.write_interaction(format_="tuple", remap=True)
file_writer.write_interaction(format_="userwise", remap=True)
file_writer.write_interaction(format_="triple", remap=False)
file_writer.write_interaction(format_="tuple", remap=False)
file_writer.write_interaction(format_="userwise", remap=False)
file_writer.write_kgdata(remap=True)
file_writer.write_kgdata(remap=False)