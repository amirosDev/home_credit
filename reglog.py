#Import libraries
from sklearn.linear_model import LogisticRegression
import pandas as pd
from ml_models_results import reglog_model_results
from ml_models_results import resmpling_data


# Import datasets
data_train = pd.read_csv("data_output/data_train.csv",index_col=0)
data_val = pd.read_csv("data_output/data_val.csv",index_col=0)
data_test = pd.read_csv("data_output/data_test.csv",index_col=0)

# Get target variable from each data
y_train = data_train['TARGET']
y_val = data_val['TARGET']
y_test = data_test['TARGET']

# drop target variable from each dataset
data_train_model = data_train.drop(['TARGET'], axis=1)
data_val_model = data_val.drop(['TARGET'], axis=1)
data_test_model = data_test.drop(['TARGET'], axis=1)


##########  logistic regression model with unbalanced classes  #############
model = LogisticRegression(random_state=0)
model1 = model.fit (data_train_model,y_train)

### model results
results_model1 = reglog_model_results(model1,data_val_model,y_val,y_train)

##########  logistic regression model with balanced classes  ##############
model_balanced = LogisticRegression(random_state=0,class_weight='balanced')
model2 = model_balanced.fit (data_train_model,y_train)

### model results
results_model2 = reglog_model_results(model2,data_val_model,y_val,y_train)

# get resampled data with perfect balanced classes (the same number of elements in each class)
percentage_to_sample = y_train.value_counts()[1]/y_train.value_counts()[0]*100
data_resampled = resmpling_data(data_train,percentage_to_sample)

######### logistic regression model with resampled data  ##############
model3 = model.fit (data_resampled['predictors'],data_resampled['target'])

### model results
results_model3 = reglog_model_results(model3,data_val_model,y_val,y_train)