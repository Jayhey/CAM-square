import time
import json
import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

with open('original_data.json', encoding='utf-8') as f:
    data = json.load(f)

random.seed(1234)
random.shuffle(data)

# data 전체 전처리
df = pd.DataFrame(data)
cat = pd.factorize(df.cuisine)
df.cuisine = pd.factorize(df.cuisine)[0]

ingredient_list = list(df['ingredients'])
unique_ingredient = list(set(x for l in ingredient_list for x in l))

mlb = MultiLabelBinarizer()
one_hot_df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('ingredients')),
                                  columns=mlb.classes_,
                                  index=df.index))


idx = int(len(data) * 0.8)
train = pd.DataFrame(one_hot_df[:idx])
test = pd.DataFrame(one_hot_df[idx:])

dtrain = xgb.DMatrix(train.iloc[:,2:].values, label=train.iloc[:,0].values)
dtest = xgb.DMatrix(test.iloc[:,2:].values, label=test.iloc[:,0].values)

# hyper parameter
num_round = 700
param = {'objective': 'multi:softmax', 
         'num_class': 20, 
         'tree_method': 'gpu_hist', 
        }

gpu_res = {} # Store accuracy result
start = time.time()
# Train model
bst = xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
print("Training Time: {} seconds".format(time.time() - start))

true = test.iloc[:,0].values
predict = bst.predict(dtest)

acc = accuracy_score(true, predict)

print("Test accuracy: {}".format(acc))