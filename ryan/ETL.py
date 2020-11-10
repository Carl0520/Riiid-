# %%
import pandas as pd
from datatable import dt, fread
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
import resource
# env = riiideducation.make_env()
pd.options.display.max_rows = 150
pd.options.display.max_columns = 50
# %% 
%%time
train = fread("/home/carlchao/Riiid_data/data/train.csv")

questions = fread('/home/carlchao/Riiid_data/data/questions.csv')
lectures = fread('/home/carlchao/Riiid_data/data/lectures.csv')
# %% 
%%time
train = train.to_pandas()
questions = questions.to_pandas()
lectures = lectures.to_pandas()
# %%
train = train.astype({
      'timestamp': 'int64',
      'user_id': 'int32',
      'content_id': 'int16',
      'content_type_id': 'bool',
      'task_container_id': 'int16',
      'user_answer': 'int8',
      'answered_correctly':'int8',
      'prior_question_elapsed_time': 'float32',
      'prior_question_had_explanation': 'bool'
})
questions = questions.astype({
      'question_id': 'int16',
      'bundle_id': 'int16',
      'correct_answer': 'int8',
      'part':'int8',
})
questions.info()
train.info()
# %% 

# q_target = train.loc[train.content_type_id == 0, ['content_id','user_id','answered_correctly']]
# q_target['question_count'] = q_target.groupby(['user_id', 'content_id']).answered_correctly.transform('count')
# # q_target.question_count.quantile([0.9, 0.95, 0.98, 0.995, 0.999])  # NOTE 99.5% percentile = 5 
# q_target.question_count.where(q_target.question_count < 5, 5, inplace=True) # NOTE clip to avoid column exploding
# q_target.drop('user_id',axis=1,inplace=True)
# q_target = q_target.groupby(['content_id','question_count']).agg(['mean','std','count'])
# q_target.columns = ['mean_acc', 'std_acc', 'count']

# # q_target[q_target.mean_acc>.9]['count'].sum() # NOTE =7261656 

# q_target = q_target.merge(questions,how='left',left_on='content_id',right_on='question_id')
# %% 
q_target = train.loc[train.content_type_id == 0, ['content_id','answered_correctly']]
q_target = q_target.groupby('content_id').agg(['mean','std','count'])
q_target.columns = ['question_mean_acc', 'question_std_acc', 'question_count']
q_target = q_target.merge(questions,how='left',left_on='content_id',right_on='question_id')
# %% 
task_target = train.loc[train.content_type_id == 0,['task_container_id','answered_correctly']].groupby('task_container_id').agg(['mean','std','count'])
task_target.columns = ['task_mean_acc', 'task_std_acc','task_count']

# %% 
# tag_dummy = questions.tags.str.get_dummies(sep=' ')
# q_target = pd.concat([q_target,tag_dummy],axis=1)
# %% 
train = train.iloc[-2000000:]
# %% 
train['time_diff'] = train.timestamp.diff()
train['user_diff'] = train.user_id.diff()
train.time_diff.min() # <0
train.loc[(train.user_diff!=0),'time_diff']=0
train.time_diff.min() # =0
# NOTE time_diff 最小值為0，對同一user來說，時間為簡單遞增，不用額外排序
# todo drop time_diff
train['user_change'] = 1
train.loc[train.user_diff==0,'user_change']=0
train.drop(['time_diff','user_diff'],axis=1,inplace=True)
train.user_change = train.user_change.astype('int8')
# %% 


def match_tags(lecture_tags, question_tags):
    matches = 0
    for l_tag in lecture_tags:
        if l_tag in question_tags:
            matches += 1
    return matches


def part_match(lecture_parts, question_part):
    if question_part in lecture_parts:
        return 1
    else:
        return 0
# %% 
lectures = lectures.set_index('lecture_id')
questions=questions.set_index('question_id')
# %% 
# %%time
# NOTE 將某user在T時間點所看過的lecture ID 接在一起變成一個column

tag_matches_ls = []
part_matches_ls = []
num_lecture_seen_ls = []
question_part_seen_ls = []
num_question_seen_ls = []

lecture_seen = []
question_part_seen = [0] * 7
num_question_seen = 0 
num_lecture_seen = 0 

for row in tqdm(train.itertuples(),total=train.shape[0]):
    if row.user_change !=0:
        lecture_seen = []
        num_lecture_seen = 0 
        num_question_seen = 0 
        question_part_seen = [0] *7

    if row.content_type_id ==1:
        lecture_seen.append(row.content_id)
        num_lecture_seen += 1
        num_tags_match = 0
        part_match_bool = 0
    else:
        num_question_seen += 1
        question_tags = questions.at[row.content_id, 'tags'].split(' ')
        question_tags = [np.int32(i) for i in question_tags]
        lecture_tags = [lectures.at[i, 'tag'] for i in lecture_seen]
        # TODO 考慮對 lecture_tags 取 set
        num_tags_match = match_tags(lecture_tags, question_tags)
        
        question_part = questions.at[row.content_id, 'part']
        question_part_seen[question_part-1] += 1
        lecture_parts = [lectures.at[i, 'part'] for i in lecture_seen]
        part_match_bool = part_match(lecture_parts, question_part)
        
    num_lecture_seen = len(lecture_seen)
    num_lecture_seen_ls.append(num_lecture_seen)
    num_question_seen_ls.append(num_question_seen)
    tag_matches_ls.append(num_tags_match)
    part_matches_ls.append(part_match_bool)
    question_part_seen_ls.append(question_part_seen.copy())

# %% 

# 將做好的list 塞回去train 然後只保留question row
train['num_lecture_seen'] =num_lecture_seen_ls
train['num_question_seen'] =num_question_seen_ls
train['num_tags_match'] =tag_matches_ls 
train['part_matches'] =part_matches_ls
part_seen_col = ['part1_seen','part2_seen','part3_seen','part4_seen','part5_seen','part6_seen','part7_seen']
part_seen = pd.DataFrame(question_part_seen_ls,columns=part_seen_col)
train = train.reset_index(drop=True)
train =pd.concat([train,part_seen],axis=1) 


train = train[train.content_type_id==0].reset_index(drop=True)
del num_lecture_seen_ls 
del tag_matches_ls 
del part_matches_ls 
del question_part_seen_ls
del num_question_seen_ls
# %% 
# %% 
train['time_min'] = train[['timestamp','user_id']].groupby('user_id').transform('min')
train.timestamp = train.timestamp -train.time_min
train.timestamp.clip(lower = 1,inplace=True)
train.timestamp = np.log(train.timestamp)

train.prior_question_elapsed_time.fillna(train.prior_question_elapsed_time.mean(),inplace=True)
train.prior_question_elapsed_time.clip(lower = 1,inplace=True)
train.prior_question_elapsed_time = np.log(train.prior_question_elapsed_time)
# %% 
final_train = train.merge(q_target[['question_mean_acc', 'question_std_acc', 'question_count', 'part', 'question_id']],
how='left',left_on='content_id',right_on='question_id')
final_train = final_train.merge(task_target,how='left',on='task_container_id')
final_train.drop(['user_id','row_id','content_id','content_type_id','task_container_id','question_id','time_min'],axis=1,inplace=True)


# %%time
# train.to_feather('train_with_lecture.ftr')

# %%
