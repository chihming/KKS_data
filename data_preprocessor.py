import pandas as pd

class DataPreprocessor:
    def __init__(self, path_1=None, path_2=None):
        self.interaction_data_1_path = path_1
        self.interaction_data_2_path = path_2
        if path_1 == None:
            self.interaction_data_1_path = "interaction_data/16b9973c-3556-40be-81af-89efa792a880.csv"
        if path_2 == None:
            self.interaction_data_2_path = "interaction_data/36d1641e-1ee9-466a-8e10-b1b0ca2b4f98.csv"
        self._load_interaction_data()
        
    def _load_interaction_data(self):
        print("loading interaction data...")
        interaction_data_1_df = pd.read_csv(self.interaction_data_1_path)
        interaction_data_2_df = pd.read_csv(self.interaction_data_2_path)
        self.interaction_df = pd.concat([interaction_data_1_df, interaction_data_2_df])
        
    def filter_data(self, item_type='VOD', select_interaction_type=['play', 'click'], interaction_thre = 10):
        print("filtering data...")
        
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
    
    
        
