import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

class FileWriter:
    def __init__(self, training_df, testing_df, kg_data):
        self.training_df = training_df
        self.testing_df = testing_df
        self.kg_data = kg_data
        
        base = os.getcwd()
        self.output_dir_path = os.path.join(base, "output_data")
    
    def write_interaction(self, format_="triple", remap=True):
        print("writing interaction data...")
        interaction_dir_path = os.path.join(self.output_dir_path, "interaction")
        training_df, testing_df = self.training_df.copy(), self.testing_df.copy()
        if remap:
            interaction_dir_path = os.path.join(interaction_dir_path, "remap_id")
            if not os.path.isdir(interaction_dir_path):
                os.makedirs(interaction_dir_path)
            usr_encoder, rel_encoder, item_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()
            usr_encoder.fit(list(set(training_df['user_id']) | set(testing_df['user_id'])))
            rel_encoder.fit(list(set(training_df['interaction']) | set(testing_df['interaction'])))
            item_encoder.fit(list(set(training_df['item_id']) | set(testing_df['item_id'])))
            np.save(interaction_dir_path + '/usr_classes.npy', usr_encoder.classes_)
            np.save(interaction_dir_path + '/rel_classes.npy', rel_encoder.classes_)
            np.save(interaction_dir_path + '/item_classes.npy', item_encoder.classes_)
            training_df['user_id'] = usr_encoder.transform(training_df['user_id'])
            training_df['interaction'] = rel_encoder.transform(training_df['interaction'])
            training_df['item_id'] = item_encoder.transform(training_df['item_id'])
            testing_df['user_id'] = usr_encoder.transform(testing_df['user_id'])
            testing_df['interaction'] = rel_encoder.transform(testing_df['interaction'])
            testing_df['item_id'] = item_encoder.transform(testing_df['item_id'])
            
        else:
            interaction_dir_path = os.path.join(interaction_dir_path, "original_id")
            
        if format_ == "triple":
            interaction_dir_path = os.path.join(interaction_dir_path, "triple")
            if not os.path.isdir(interaction_dir_path):
                os.makedirs(interaction_dir_path)
            with open(interaction_dir_path + "/train_triple.txt", "w") as f:
                for usr, rel, item in training_df[["user_id", "interaction", "item_id"]].values:
                    f.write(str(usr) + '\t' + str(rel) + '\t' + str(item) + '\n')
            with open(interaction_dir_path + "/test_triple.txt", "w") as f:
                for usr, rel, item in testing_df[["user_id", "interaction", "item_id"]].values:
                    f.write(str(usr) + '\t' + str(rel) + '\t' + str(item) + '\n')
        elif format_ == "tuple":
            interaction_dir_path = os.path.join(interaction_dir_path, "tuple")
            if not os.path.isdir(interaction_dir_path):
                os.makedirs(interaction_dir_path)
            with open(interaction_dir_path + "/train_tuple.txt", "w") as f:
                for usr, item in training_df[["user_id", "item_id"]].values:
                    f.write(str(usr) + '\t' + str(item) + '\n')
            with open(interaction_dir_path + "/test_tuple.txt", "w") as f:
                for usr, item in testing_df[["user_id", "item_id"]].values:
                    f.write(str(usr) + '\t' + str(item) + '\n')
        elif format_ == "userwise":
            train_usr_dict, test_usr_dict = {}, {}
            for usr, item in training_df[["user_id", "item_id"]].values:
                if usr not in train_usr_dict:
                    train_usr_dict[usr] = set()
                train_usr_dict[usr].add(item)
            for usr, item in testing_df[["user_id", "item_id"]].values:
                if usr not in test_usr_dict:
                    test_usr_dict[usr] = set()
                test_usr_dict[usr].add(item)
            
            interaction_dir_path = os.path.join(interaction_dir_path, "userwise")
            if not os.path.isdir(interaction_dir_path):
                os.makedirs(interaction_dir_path)
            with open(interaction_dir_path + "/train_userwise.txt", "w") as f:
                for usr in train_usr_dict:
                    items = list(sorted(train_usr_dict[usr]))
                    items = [str(item) for item in items]
                    items = " ".join(items)
                    f.write(str(usr) + " " + items + '\n')
            with open(interaction_dir_path + "/test_userwise.txt", "w") as f:
                for usr in test_usr_dict:
                    items = list(sorted(test_usr_dict[usr]))
                    items = [str(item) for item in items]
                    items = " ".join(items)
                    f.write(str(usr) + " " + items + '\n')
        else:
            raise ValueError ("Invalid output format! Format should be 'triple', 'tuple' or 'userwise'.")
            
    def write_kgdata(self, remap=False):
        print("writing kg data...")
        kg_dir_path = os.path.join(self.output_dir_path, "kg_data")
        kg_data = self.kg_data
        if remap:
            kg_dir_path = os.path.join(kg_dir_path, "remap_id")
            if not os.path.isdir(kg_dir_path):
                os.makedirs(kg_dir_path)
            kg_data_array = np.array(kg_data)
            item_encoder, rel_encoder, ent_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()
            kg_data_array[:,0] = item_encoder.fit_transform(kg_data_array[:,0])
            kg_data_array[:,1] = rel_encoder.fit_transform(kg_data_array[:,1])
            kg_data_array[:,2] = ent_encoder.fit_transform(kg_data_array[:,2])
            np.save(kg_dir_path + '/item_classes.npy', item_encoder.classes_)
            np.save(kg_dir_path + '/rel_classes.npy', rel_encoder.classes_)
            np.save(kg_dir_path + '/ent_classes.npy', ent_encoder.classes_)
            kg_data = kg_data_array.tolist()
            
        else:
            kg_dir_path = os.path.join(kg_dir_path, "original_id")
            
        if not os.path.isdir(kg_dir_path):
            os.makedirs(kg_dir_path)
        with open(kg_dir_path + '/kg.txt', 'w') as f:
            for h, r, t in kg_data:
                f.write(str(h) + '\t' + str(r) + '\t' + str(t) +'\n')
        
