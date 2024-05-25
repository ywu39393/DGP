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

###############Calculate mean IC50 value by drug/cell

train['IC50_drugmean'] = train.groupby('drug_id')['IC50'].transform('mean')
train['IC50_cellmean'] = train.groupby('Cancer_Cell_Line')['IC50'].transform('mean')
test['IC50_drugmean'] = test.groupby('drug_id')['IC50'].transform('mean')
test['IC50_cellmean'] = test.groupby('Cancer_Cell_Line')['IC50'].transform('mean')


name = 'margin'

max_column=train.shape[1] #maxmum column
gene_id = 691 # 691 all gene expression 
drug_id = 100     # 100 drug embedding


#create train dataset x
train_x_gene = train.iloc[:, 3:gene_id+3]
train_x_drug = train.iloc[:,gene_id+3: gene_id+3+drug_id]
train_y_drug = train.iloc[:,max_column-2]
train_y_cell = train.iloc[:,max_column-1]
#create test dataset x
test_x_gene = test.iloc[:, 3:gene_id+3]
test_x_drug = test.iloc[:,gene_id+3: gene_id+3+drug_id]
test_y_drug = test.iloc[:,max_column-2]
test_y_cell = test.iloc[:,max_column-1]
#standization
#standization
train_x_gene_s = min_max_normalization(train_x_gene)
train_x_drug_s = min_max_normalization(train_x_drug)
train_y_drug_s = min_max_normalization(train_y_drug)
train_y_cell_s = min_max_normalization(train_y_cell)

test_x_gene_s = min_max_normalization(test_x_gene)
test_x_drug_s = min_max_normalization(test_x_drug)
test_y_drug_s = min_max_normalization(test_y_drug)
test_y_cell_s = min_max_normalization(test_y_cell)

#transfer input data
train_x_gene_n = train_x_gene_s.values
train_x_drug_n = train_x_drug_s.values
test_x_gene_n = test_x_gene_s.values
test_x_drug_n = test_x_drug_s.values

train_x_gene_t = torch.tensor(train_x_gene_n, dtype=torch.float)
train_x_drug_t = torch.tensor(train_x_drug_n, dtype=torch.float)
test_x_gene_t = torch.tensor(test_x_gene_n, dtype=torch.float)
test_x_drug_t = torch.tensor(test_x_drug_n, dtype=torch.float)

#transfer y data to torch
train_y_drug_t = train_y_drug_s.values
train_y_cell_t = train_y_cell_s.values
test_y_drug_t = test_y_drug_s.values
test_y_cell_t = test_y_cell_s.values

train_y_drug_t = torch.tensor(train_y_drug_t, dtype=torch.float).view(-1,1)
train_y_cell_t = torch.tensor(train_y_cell_t, dtype=torch.float).view(-1,1)
test_y_drug_t = torch.tensor(test_y_drug_t, dtype=torch.float).view(-1,1)
test_y_cell_t = torch.tensor(test_y_cell_t, dtype=torch.float).view(-1,1)


###04/10
print('train_x_gene:' + str(train_x_gene_t.shape) + ', train_x_drug:'+ str(train_x_drug_t.shape) 
      + ', test_x_gene:' + str(test_x_gene_t.shape)+ ', test_x_drug:' + str(test_x_drug_t.shape) 
      + ', train_y_drug:' + str(train_y_drug_t.shape) + ', train_y_cell:' + str(train_y_cell_t.shape) 
      + ', test_y_drug:' + str(test_y_drug_t.shape) + ', test_y_cell:' + str(test_y_cell_t.shape))
torch.save(train_x_gene_t, 'train_x_gene_'+ name)
torch.save(train_x_drug_t, 'train_x_drug_'+ name)
torch.save(test_x_gene_t, 'test_x_gene_'+ name)
torch.save(test_x_drug_t, 'test_x_drug_'+ name)
torch.save(train_y_drug_t, 'train_y_drug_'+ name)
torch.save(train_y_cell_t, 'train_y_gene_'+ name)
torch.save(test_y_drug_t, 'test_y_drug_'+ name)
torch.save(test_y_cell_t, 'test_y_gene_'+ name)

