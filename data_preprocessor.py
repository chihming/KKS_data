import logging
import gzip, json
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(
        self,
        interaction_paths,
        meta_dir_path,
    ):
        # interactions
        logger.info('Load interaction data from:')
        for path in interaction_paths:
            logger.info(f'{path}')
        self.interaction_df = pd.concat(
            [pd.read_csv(path) for path in interaction_paths]
        )

        # metadata
        logger.info(f'Load metadata from {meta_dir_path}')
        self.meta_dir_path = meta_dir_path
        self.descriptor = json.load((meta_dir_path/"descriptor.json").open('r'))


    def filter_interaction_data(
        self,
        item_type='VOD',
        select_interaction_type=['play', 'click'],
        interaction_thre=10,
        split_time="2022-07-02 15:00",
    ):
        logger.info("filtering interaction data...")

        assert item_type.lower() in ['vod', 'tv', 'all'], \
            "Invalid item type! Item type should be 'VOD', 'TV' or 'All'."
        assert set(select_interaction_type).issubset({'click','download', 'favor', 'play', 'purchase', 'record', 'reserve', 'search'}), \
            "Invalid interaction type! Interaction type should be a subset of {'click','download', 'favor', 'play', 'purchase', 'record', 'reserve', 'search'}."
        assert interaction_thre > 0, \
            "Invalid interaction threshold! Interaction threshold should be a non-negative number."

        # item_type : ['VOD', 'TV', None]
        if item_type.lower() == 'vod':
            interaction_df = self.interaction_df[self.interaction_df['item_type'].isin(['movie', 'series'])]
        elif item_type.lower() == 'tv':
            interaction_df = self.interaction_df[self.interaction_df['item_type'].isin(['BSD', 'JCC', 'JCM', 'OTD', 'ip'])]
        elif item_type.lower() == 'all':
            interaction_df = self.interaction_df

        # interaction type : ['click','download', 'favor', 'play', 'purchase', 'record', 'reserve', 'search']
        interaction_df = interaction_df[interaction_df['interaction'].isin(select_interaction_type)]

        # interaction threshold : non-negative number
        user_count_df = interaction_df.groupby('user_id').size().reset_index(name='counts')
        active_user_set = set(user_count_df[user_count_df['counts'] > interaction_thre]['user_id'])
        interaction_df = interaction_df[interaction_df['user_id'].isin(active_user_set)]

        # Data splitting
        training_df = interaction_df[interaction_df['client_upload_timestamp'] < split_time].sort_values(by="client_upload_timestamp")
        testing_df = interaction_df[interaction_df['client_upload_timestamp'] >= split_time].sort_values(by="client_upload_timestamp")
        return training_df, testing_df

    def filter_metadata(
        self,
        select_branch_type=['movie', 'series', 'season', 'episode'],
        select_relation_type=['artists', 'genres']
    ):
        logger.info("filtering metadata...")

        assert set(select_branch_type).issubset({'OTD', 'BSD', 'B4K', 'JCM', 'JCC', 'ip', 'movie', 'episode', 'season', 'series'}), \
            "Invalid branch type! Item type should be 'OTD', 'BSD', 'B4K', 'JCM', 'JCC', 'ip', 'movie', 'episode', 'season', 'series'."
        assert set(select_relation_type).issubset({'artists', 'content_rating', 'genres', 'type'}), \
            "Invalid relation type! Relation type should be 'artists', 'content_rating', 'genres', 'type'."

        logger.info('Load metadata')
        metadata_dict = {}
        for branch in self.descriptor['branches']:
            if 'type' not in branch['branchname']:
                continue
            branch_type = branch['branchname']['type']
            if branch_type not in select_branch_type:
                logger.info(f'skip {branch_type}')
                continue
            logger.info(f'from {branch_type}')

            metadata_dict[branch_type] = []
            for file_name in branch['filenames']:
                file_path = self.meta_dir_path / file_name
                with gzip.open(file_path, 'rb') as f:
                    lines = f.readlines()
                    for line in lines:
                        metadata_dict[branch_type].append(json.loads(line))

        for keyword in ['series', 'season', 'episode']:
            assert keyword in metadata_dict, f"{keyword} not found in metadata"

        for series in metadata_dict['series']:
            title = series['properties#SeriesTitle']
            for season in metadata_dict['season']:
                if season['properties#SeriesTitle'] == title:
                    for rel in select_relation_type:
                        if season[rel] != None:
                            series[rel] = list(set(series[rel]) | set(season[rel]))
            for episode in metadata_dict['episode']:
                if episode['properties#SeriesTitle'] == title:
                    for rel in select_relation_type:
                        if episode[rel] != None:
                            series[rel] = list(set(series[rel]) | set(episode[rel]))
        select_branch_type = list(set(select_branch_type) - {'season', 'episode'})

        kg_data = []
        for branch in select_branch_type:
            for item in metadata_dict[branch]:
                title = item['name']
                for rel in select_relation_type:
                    if item[rel] != None:
                        for ent in item[rel]:
                            kg_data.append([title, rel, ent])
        return kg_data

