import numpy as np
import pandas as pd
import torch
import os

def min_max_normalization(data):

    # Calculate the minimum and maximum values for each feature
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Min-max normalization formula: (X - min) / (max - min)
    normalized_data = (data - min_vals) / (max_vals - min_vals)

    return normalized_data

file_path_train = '/work/sghoshstat/ywu39393/Research/Code/train_data_deepcdr.csv'
file_path_test ='/work/sghoshstat/ywu39393/Research/Code/test_data_deepcdr.csv'

train = pd.read_csv(file_path_train)
test = pd.read_csv(file_path_test)

# # random select n rows
# random_train = train.sample(n= 1000, random_state=66)
# random_test = test.sample(n= 1000, random_state=66)
# train = random_train
# test = random_test
name = 'full'

# max_column=train.shape[1] #maxmum column
# gene_id = 691 # 691 all gene expression
# select_id = gene_id

#create train dataset x
train_x_a = train.iloc[:,3: ]
train_y = train.iloc[:,2]
#create test dataset x
test_x_a = test.iloc[:,3: ]
test_y = test.iloc[:,2]
print(train_x_a, test_x_a)
#standization
#standization
train_x_s = min_max_normalization(train_x_a)
test_x_s = min_max_normalization(test_x_a)
train_y_s = min_max_normalization(train_y)
test_y_s = min_max_normalization(test_y)

#transfer input data
train_x_n = train_x_s.values
test_x_n = test_x_s.values
train_x_t = torch.tensor(train_x_n, dtype=torch.float)
test_x_t = torch.tensor(test_x_n, dtype=torch.float)
#transfer y data to torch
train_y_t = train_y_s.values
test_y_t = test_y_s.values
train_y_t = torch.tensor(train_y_t, dtype=torch.float).view(-1,1)
test_y_t = torch.tensor(test_y_t, dtype=torch.float).view(-1,1)



print('train_x:' + str(train_x_t.shape) + ', train_y:' + str(train_y_t.shape)+ ', test_x:' + str(test_x_t.shape) + ', test_y:' + str(test_y_t.shape)   )
torch.save(train_x_t, 'train_x_'+ name)
torch.save(train_y_t, 'train_y_'+ name)
torch.save(test_x_t, 'test_x_'+ name)
torch.save(test_y_t, 'test_y_'+ name)

