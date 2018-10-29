from data_processing import prepare_data_brut, train_val_size, prepare_data
from ml_models_results import resmpling_data, results_model_dict, confusion_matrix, reglog_model_results
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.utils import resample
import pandas as pd
from prettytable import PrettyTable
from statistics import mean, median, stdev


#prepare data
data = prepare_data_brut(path_to="/home/user/Kaggle/data/application_train.csv")
dict_data_input = prepare_data(path_to="/home/user/Kaggle/data/application_train.csv")

# datasets :
data_val =  dict_data_input['data_val']
data_test = dict_data_input['data_test']

# Get target variable from each data
y_val = data_val['TARGET']
y_test = data_test['TARGET']

# drop target variable from each dataset
data_val_model = data_val.drop(['TARGET'], axis=1)
data_test_model = data_test.drop(['TARGET'], axis=1)

# the model
model = GradientBoostingClassifier()

#get data train
data_train = dict_data_input['data_train']

# performance :
auc_results = list()
accuracy_results = list()

for i in range(0,10):
    # resample data
    data_resampled = resmpling_data(data_train, 9, string="percentage")
    y_train = data_resampled['TARGET']
    data_train_model = data_resampled.drop(['TARGET'], axis=1)

    # built model
    model1 = model.fit(data_train_model, y_train)

    ### model results
    results_model1 = reglog_model_results(model1, data_test_model, y_test)

    ## append results
    auc = round(results_model1['AUC'],4)
    accuracy = round(results_model1['Accuracy'],4)
    auc_results.append(auc)
    accuracy_results.append(accuracy)

t = PrettyTable(['Model', 'accuracy','AUC'])

for i in range(0,10):
    m = "model"+" "+str(i+1)
    t.add_row([m,accuracy_results[i],auc_results[i]])

t.add_row(["Mean",mean(accuracy_results),mean(auc_results)])
t.add_row(["Standard deviation",stdev(accuracy_results),stdev(auc_results)])
t.add_row(["Median",median(accuracy_results),median(auc_results)])
print(t)

#print("Accuracy = ",round(results_model1['Accuracy'],4),"AUC = ",round(results_model1['AUC'],4))
cm = results_model1['Confusion_Matrix']
confusion_matrix(cm)


# performance :
auc_results = list()
accuracy_results = list()

# Resampling with stratification
data_input = dict_data_input['data_train']
y = data_input['TARGET']
data_input_0 = data_input[y == 0]

for i in range(0,10):
    # kmeans
    kmeans = KMeans(n_clusters=6).fit(data_input_0)
    d = pd.Series(kmeans.labels_)
    #print(d.value_counts())
    #print(len(d), data_input_0.shape[0])
    #print("percentage :", round(data_input_0.shape[0] * 9 / 100))
    dict = {}
    for i in range(0, 6):
        rows = list(d[d == i].index)
        data = data_input_0.iloc[rows, :]
        number_to_sample = data.shape[0] * 9 / 100
        number_to_sample = round(number_to_sample)
        data_sample = resample(data, replace=False, n_samples=number_to_sample)
        key = str(i)
        dict.update({key: data_sample})

    data_output = pd.concat([dict['0'], dict['1'], dict['2'], dict['3'], dict['4'], dict['5']])
    #print(data_output.shape[0])

    # reconstruct data
    data_input_1 = data_input[y == 1]
    data_train_output = data_output.append(data_input_1)
    y_train = data_train_output['TARGET']
    data_train_model = data_train_output.drop(['TARGET'], axis=1)

    # built model
    model2 = model.fit(data_train_model, y_train)

    ### model results
    results_model2 = reglog_model_results(model2, data_test_model, y_test)

    ## append results
    auc = round(results_model1['AUC'], 4)
    accuracy = round(results_model1['Accuracy'], 4)
    auc_results.append(auc)
    accuracy_results.append(accuracy)

t = PrettyTable(['Model', 'accuracy','AUC'])

for i in range(0,10):
    m = "model_Kmeans"+" "+str(i+1)
    t.add_row([m,accuracy_results[i],auc_results[i]])

t.add_row(["Mean",mean(accuracy_results),mean(auc_results)])
t.add_row(["Standard deviation",stdev(accuracy_results),stdev(auc_results)])
t.add_row(["Median",median(accuracy_results),median(auc_results)])
print(t)

#print("Accuracy = ",round(results_model2['Accuracy'],4),"AUC = ",round(results_model2['AUC'],4))
cm = results_model2['Confusion_Matrix']
confusion_matrix(cm)


