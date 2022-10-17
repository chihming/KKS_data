import argparse
from pathlib import Path
from data_preprocessor import DataPreprocessor
from file_writer import FileWriter

parser = argparse.ArgumentParser(
    description="Parameters for the script."
)
parser.add_argument(
    '-interaction_paths',
    nargs='+',
    required=True,
    help='interaction log csv(s)'
)
parser.add_argument(
    '-split_time',
    default="2022-07-02 15:00",
    help='interaction log csv(s)'
)
parser.add_argument(
    '-meta_dir_path',
    required=True,
    help='directory to meta'
)
parser.add_argument(
    '-item_type',
    default='VOD',
    help='target item type'
)
parser.add_argument(
    '-select_interaction_type',
    default=['click', 'play'],
    nargs='+',
    help='types to be selected for train/test'
)
parser.add_argument(
    '-interaction_thre',
    default=10,
    type=int,
    help='threshold for selecting users'
)
parser.add_argument(
    '-select_branch_type',
    default=['movie', 'series', 'season', 'episode'],
    nargs='+',
    help='types to be selected for train/test'
)
parser.add_argument(
    '-select_relation_type',
    default=['artists', 'genres'],
    nargs='+',
    help='types to be selected for train/test'
)
args = parser.parse_args()

data_preprocessor = DataPreprocessor(
    interaction_paths = [Path(p) for p in args.interaction_paths],
    meta_dir_path = Path(args.meta_dir_path),
)

# filter interaction data
training_df, testing_df = data_preprocessor.filter_interaction_data(
    item_type = args.item_type,
    select_interaction_type = args.select_interaction_type,
    interaction_thre = args.interaction_thre,
    split_time = args.split_time,
)

# filter metadata
kg_data = data_preprocessor.filter_metadata(
    select_branch_type = args.select_branch_type,
    select_relation_type = args.select_relation_type
)

# write files
file_writer = FileWriter()
file_writer.write_triple(
    training_df = training_df,
    testing_df = testing_df,
    output_dir_path = Path('exp/triple')
)
file_writer.write_tuple(
    training_df = training_df,
    testing_df = testing_df,
    output_dir_path = Path('exp/tuple')
)
file_writer.write_userwise(
    training_df = training_df,
    testing_df = testing_df,
    output_dir_path = Path('exp/userwise')
)
file_writer.write_kgdata(
    kg_data = kg_data,
    output_dir_path = Path('exp/kgdata')
)
