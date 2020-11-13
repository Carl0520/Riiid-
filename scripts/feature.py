import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm.notebook import tqdm
import lightgbm as lgb
from datatable import dt, fread

train_path = '/home/carlchao/Riiid-/CV/csv_file/cv1_train.csv'
valid_path = '/home/carlchao/Riiid-/CV/csv_file/cv1_valid.csv'
question_file = '/home/carlchao/Riiid_data/data/questions.csv'


def add_user_feats(df, answered_correctly_sum_u_dict, count_u_dict, content_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    bool = np.zeros(len(df), dtype=np.int8)
    
    for cnt,row in enumerate(tqdm(df[['user_id','answered_correctly','content_id']].values)):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
        answered_correctly_sum_u_dict[row[0]] += row[1]
        count_u_dict[row[0]] += 1
        
        if row[2] not in content_dict[row[0]]:
            content_dict[row[0]].append(row[2])
            bool[cnt] = 1
        else:
            bool[cnt] = 0
            
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu, 'first_time': bool})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    
    return df

def add_user_feats_without_update(df, answered_correctly_sum_u_dict, count_u_dict, content_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    bool = np.zeros(len(df), dtype=np.int8)
    
    for cnt,row in enumerate(df[['user_id','content_type_id']].values):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
        
        if row[1] not in content_dict[row[0]]:
            bool[cnt] = 1
        else:
            bool[cnt] = 0
            
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu, 'first_time': bool})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def update_user_feats(df, answered_correctly_sum_u_dict, count_u_dict, content_dict):
    for row in df[['user_id','answered_correctly','content_type_id']].values:
        if row[2] == 0:
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1
            content_dict[row[0]].append(row[2])
            
            
# drop task_container_id, user_answer
feld_needed = ['row_id', 'timestamp','user_id', 'content_id', 'content_type_id', 'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation']
train = fread(train_path).to_pandas()[feld_needed]
valid = fread(valid_path).to_pandas()[feld_needed]


train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

# answered correctly average for each content
content_df = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).reset_index()
content_df.columns = ['content_id', 'answered_correctly_avg_c']
train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")



answered_correctly_sum_u_dict = defaultdict(int)
count_u_dict = defaultdict(int)
content_dict = defaultdict(list)
train = add_user_feats(train, answered_correctly_sum_u_dict, count_u_dict, content_dict)
train.to_pickle('preprocess_fea/small_train.pickle')
del train
_ = gc.collect()
valid = add_user_feats(valid, answered_correctly_sum_u_dict, count_u_dict, content_dict)
valid.to_pickle('preprocess_fea/small_valid.pickle')



import pickle

with open("preprocess_fea/answered_correctly_sum_u_dict.pickle", "wb") as filename:  
    pickle.dump(answered_correctly_sum_u_dict, filename)
    filename.close()

with open("preprocess_fea/count_u_dict.pickle", "wb") as filename:  
    pickle.dump(count_u_dict, filename)
    filename.close()

with open("preprocess_fea/content_dict.pickle", "wb") as filename:  
    pickle.dump(content_dict, filename)
    filename.close()