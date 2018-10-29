from sklearn.ensemble import GradientBoostingClassifier
from ml_models_results import reglog_model_results, resmpling_data
from data_processing import prepare_data
from sklearn.model_selection import RandomizedSearchCV
from time import time
import pandas as pd


# Import datasets
dict_data = prepare_data(path_to="data/application_train.csv")
data_train = dict_data['data_train']

#resample data
data_resampled = resmpling_data(data_train,9,string="percentage")


# datasets :
data_val =  dict_data['data_val']
data_test = dict_data['data_test']

# Get target variable from each data
y_train = data_resampled['TARGET']
y_val = data_val['TARGET']
y_test = data_test['TARGET']

# drop target variable from each dataset
data_train_model = data_resampled.drop(['TARGET'], axis=1)
data_val_model = data_val.drop(['TARGET'], axis=1)
data_test_model = data_test.drop(['TARGET'], axis=1)


##########  GBM model   #############
model = GradientBoostingClassifier()

### Hyperparameters tunning
random_grid = {'max_depth': [10, 20, 30, 40, 50, 60],
			   'n_estimators': [200, 300, 400, 500, 600],
			   'learning_rate': [0.001, 0.01, 0.1, 0.2]}


### Using Random search
gbm_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 20, verbose=2, n_jobs = -1, return_train_score=True)

# start time
start = time()

# Fit the random search model
best_model = gbm_random.fit(data_train_model, y_train)

# Calculate time
print("RandomizedSearchCV took %.2f seconds"
      " parameter settings." % (time() - start))

# export results
data_output = pd.DataFrame(best_model.cv_results_)

data_output.to_csv("data_output/output_random_search.csv")


