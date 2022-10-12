import pandas as pd
import json
import os
import gzip

class DataPreprocessor:
    def __init__(self, interaction_path_1=None, interaction_path_2=None, metadata_dir_path=None):
        self.interaction_data_1_path = interaction_path_1
        self.interaction_data_2_path = interaction_path_2
        self.metadata_dir_path = metadata_dir_path
        
        if interaction_path_1 == None:
            self.interaction_data_1_path = "interaction_data/16b9973c-3556-40be-81af-89efa792a880.csv"
        if interaction_path_2 == None:
            self.interaction_data_2_path = "interaction_data/36d1641e-1ee9-466a-8e10-b1b0ca2b4f98.csv"
        self._load_interaction_data()
        self._load_metadata()
        
    def _load_interaction_data(self):
        print("loading interaction data...")
        interaction_data_1_df = pd.read_csv(self.interaction_data_1_path)
        interaction_data_2_df = pd.read_csv(self.interaction_data_2_path)
        self.interaction_df = pd.concat([interaction_data_1_df, interaction_data_2_df])
        
    def _load_metadata(self):
        print("loading metadata...")
        descriptor_path = os.path.join(self.metadata_dir_path, "descriptor.json")
        with open(descriptor_path, 'r') as f:
            self.descriptor = json.load(f)
        
    def filter_interaction_data(self, item_type='VOD', select_interaction_type=['play', 'click'], interaction_thre = 10):
        print("filtering interaction data...")
        
        def error_handler(item_type, select_interaction_type, interaction_thre):
            if item_type not in ['VOD', 'vod', 'TV', 'tv', 'All', 'all']:
                raise ValueError ("Invalid item type! Item type should be 'VOD', 'TV' or 'All'.")
            if not set(select_interaction_type) < {'click','download', 'favor', 'play', 'purchase', 'record', 'reserve', 'search'}:
                raise ValueError ("Invalid interaction type! Interaction type should be a subset of {'click','download', 'favor', 'play', 'purchase', 'record', 'reserve', 'search'}.")
            if interaction_thre < 0:
                raise ValueError ("Invalid interaction threshold! Interaction threshold should be a non-negative number.")
        
        error_handler(item_type, select_interaction_type, interaction_thre)
        
        # item_type : ['VOD', 'TV', None]
        if item_type in ['VOD', 'vod']:
            interaction_df = self.interaction_df[self.interaction_df['item_type'].isin(['movie', 'series'])]
        elif item_type in ['TV', 'tv']:
            interaction_df = self.interaction_df[self.interaction_df['item_type'].isin(['BSD', 'JCC', 'JCM', 'OTD', 'ip'])]
        elif item_type in ['All', 'all']:
            interaction_df = self.interaction_df

        # interaction type : ['click','download', 'favor', 'play', 'purchase', 'record', 'reserve', 'search']
        interaction_df = interaction_df[interaction_df['interaction'].isin(select_interaction_type)]
    
        # interaction threshold : non-negative number
        user_count_df = interaction_df.groupby('user_id').size().reset_index(name='counts')
        self.active_user_set = set(user_count_df[user_count_df['counts'] > interaction_thre]['user_id'])
        interaction_df = interaction_df[interaction_df['user_id'].isin(self.active_user_set)]
        
        # Data splitting
        training_df = interaction_df[interaction_df['client_upload_timestamp'] < "2022-07-02 15:00"].sort_values(by="client_upload_timestamp")
        testing_df = interaction_df[interaction_df['client_upload_timestamp'] >= "2022-07-02 15:00"].sort_values(by="client_upload_timestamp")
        return training_df, testing_df
    
    def filter_metadata(self, select_branch_type=['movie', 'series', 'season', 'episode'], select_relation_type=['artists', 'genres']):
        print("filtering metadata...")
        
        def error_handler(select_branch_type, select_relation_type):
            if not set(select_branch_type) < {'OTD', 'BSD', 'B4K', 'JCM', 'JCC', 'ip', 'movie', 'episode', 'season', 'series'}:
                raise ValueError ("Invalid branch type! Item type should be 'OTD', 'BSD', 'B4K', 'JCM', 'JCC', 'ip', 'movie', 'episode', 'season', 'series'.")
            if not set(select_relation_type) < {'artists', 'content_rating', 'genres', 'type'}:
                raise ValueError ("Invalid relation type! Relation type should be 'artists', 'content_rating', 'genres', 'type'.")
        
        error_handler(select_branch_type, select_relation_type)
        
        file_name_dict = {}
        for branch in self.descriptor['branches']:
            if 'type' not in branch['branchname']:
                continue
            branch_type = branch['branchname']['type']
            if branch_type in select_branch_type:
                file_name_dict[branch_type] = branch['filenames']
                
        self.metadata_dict = {}
        for branch_type in file_name_dict:
            self.metadata_dict[branch_type] = []
            for file_name in file_name_dict[branch_type]:
                file_path = os.path.join(self.metadata_dir_path, file_name)
                with gzip.open(file_path, 'rb') as f:
                    lines = f.readlines()
                    for line in lines:
                        self.metadata_dict[branch_type].append(json.loads(line))
                       
        if ['series', 'season', 'episode'] in select_branch_type:
            for series in self.metadata_dict['series']:
                title = series['properties#SeriesTitle']
                for season in self.metadata_dict['season']:
                    if season['properties#SeriesTitle'] == title:
                        for rel in select_relation_type:
                            if season[rel] != None:
                                series[rel] = list(set(series[rel]) | set(season[rel]))
                for episode in self.metadata_dict['episode']:
                    if episode['properties#SeriesTitle'] == title:
                        for rel in select_relation_type:
                            if episode[rel] != None:
                                series[rel] = list(set(series[rel]) | set(episode[rel]))
        select_branch_type = list(set(select_branch_type) - {'season', 'episode'})
        
        kg_data = []
        for branch in select_branch_type:
            for item in self.metadata_dict[branch]:
                title = item['name']
                for rel in select_relation_type:
                    if item[rel] != None:
                        for ent in item[rel]:
                            kg_data.append([title, rel, ent])
        return kg_data
    
    
        
