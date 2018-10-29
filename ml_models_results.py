import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.utils import resample
import matplotlib.pyplot as plt
from data_processing import prepare_data_brut
from data_processing import train_val_size
from sklearn.ensemble import GradientBoostingClassifier
from prettytable import PrettyTable
plt.rc("font", size=14)


## submission kaggle function
def submission (model,data_test):

    # get predicted target from model and the id from input data
    y_predict = model.predict(data_test)
    ID = data_test["SK_ID_CURR"]

    # construct data kaggle submission
    kaggle_submission = pd.DataFrame(ID,y_predict,columns=["TARGET","SK_ID_CURR"])

    return kaggle_submission


## Performance indicators for logistic regression function
def reglog_model_results(model,data_test,y_test):

    # Calculate Class Probabilities
    probability = model.predict_proba(data_test)

    # Predicted Class Labels
    y_predicted = model.predict(data_test)

    # Evaluate The Model

    ### Confusion Matrix
    Confusion_Matrix = metrics.confusion_matrix(y_test, y_predicted)

    ### Classification Report
    Classification_Report = metrics.classification_report(y_test, y_predicted)

    ### Model Accuracy
    Accuracy = model.score(data_test, y_test)

    ### AUC
    y_pred_proba = probability[:, 1]
    [fpr, tpr, thr] = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.auc(fpr, tpr)

    return {'Class_Probabilities':probability,'Predicted_Class_Labels':y_predicted,'Confusion_Matrix':Confusion_Matrix,'Classification_Report':Classification_Report,'Accuracy':Accuracy, 'AUC':auc}


# Show Confusion Matrix
def confusion_matrix(cm):
    tab = PrettyTable([' ', 'Predicted 0', 'Predicted 1'])
    tab.add_row(["Actual 0", cm[0][0], cm[0][1]])
    tab.add_row(["Actual 1", cm[1][0], cm[1][1]])
    print(tab)



# Show the ROC_CURVE
def roc_curve_show(model,data_test,y_test):

    result_model = reglog_model_results(model, data_test, y_test)
    y_pred_proba = result_model['Class_Probabilities'][:, 1]
    [fpr, tpr, thr] = metrics.roc_curve(y_test, y_pred_proba)
    idx = np.min(np.where(tpr > 0.95))  # index of the first threshold for which the sensibility > 0.95
    plt.figure()
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, fpr[idx]], [tpr[idx], tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx], fpr[idx]], [0, tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()


def resmpling_data(data,val,string="percentage"):
    '''
    Resampling data with given a value of percentage or number for output rows number

    :param data: pandas data_train to resample
    :param val: value of percentage or rows number
    :param string: percentage or number
    :return: dataset resampled
    '''
    y = data['TARGET']
    data_0 = data[y == 0]
    data_1 = data[y == 1]
    nrows = data_0.shape[0]
    if string == "percentage":
        sample = int(round(val*nrows/100))
        boot = resample(data_0, replace=False, n_samples=sample)
    else:
        boot = resample(data_0, replace=False, n_samples=val)

    data_boot = boot.append(data_1)
    return data_boot


def results_model_data(model,data_model,val_size=0.2,test_size=0.2):
    '''
    Split data-model into train, val and test datasets and then apply the model and finally get results
    :param model: model to apply
    :param data: data_model
    :param val_size:
    :param test_size:
    :return:  model results and random data split result
    '''

    dict_data = train_val_size(data_model, val_size, test_size)
    # datasets :
    data_train = dict_data['data_train']
    data_val = dict_data['data_val']
    data_test = dict_data['data_test']

    # Get target variable from each data
    y_train = data_train['TARGET']
    y_val = data_val['TARGET']
    y_test = data_test['TARGET']

    # drop target variable from each dataset
    data_train_model = data_train.drop(['TARGET'], axis=1)
    data_val_model = data_val.drop(['TARGET'], axis=1)
    data_test_model = data_test.drop(['TARGET'], axis=1)

    ##########  fit model   #############
    model1 = model.fit(data_train_model, y_train)

    ### model results
    results_model1 = reglog_model_results(model1, data_val_model, y_val)
    results_model1.update(dict_data)
    return results_model1

def results_model_dict(model,dict_data):
    '''
        apply the model and get results
        :param model: model to apply
        :param dict_data: dictionary including model datasets
        :return: dict of model results
        '''

    # datasets :
    data_train = dict_data['data_train']
    data_val = dict_data['data_val']
    data_test = dict_data['data_test']

    # Get target variable from each data
    y_train = data_train['TARGET']
    y_val = data_val['TARGET']
    y_test = data_test['TARGET']

    # drop target variable from each dataset
    data_train_model = data_train.drop(['TARGET'], axis=1)
    data_val_model = data_val.drop(['TARGET'], axis=1)
    data_test_model = data_test.drop(['TARGET'], axis=1)

    ##########  fit model   #############
    model1 = model.fit(data_train_model, y_train)

    ### model results
    results_model1 = reglog_model_results(model1, data_val_model, y_val)
    return results_model1
