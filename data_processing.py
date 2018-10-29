import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

threshold = 0
statistic_list = ["mean","median"]

def nb_col_mis_val(var):
    return len(var[var != 0])


def affiche_mis_val(var):
    print(var[var != 0])


def col_mis_val_threshold(percentage,data,var):
    global threshold
    threshold = round(len(data.index) / percentage)
    print(threshold)
    print("\n Columns containing missing values (less than ",percentage,"%)")
    print(var[var < threshold])


def nb_col_under_threshold(var):
    return len(var[var != 0][var < threshold])


def mis_val_treatment(data,string,percentage):
    """Treat the missing values.

    A detailed explanation of what the function does....

    :param data:
    :param string:
    :param percentage:
    :return:
    """
    var = data.isnull().sum()
    threshold = round(len(data.index) / percentage)
    data_mis_val = data[var[var!=0][var<threshold].index].copy()
    names_data = list(data_mis_val.columns.values)
    names_data_numeric = list(data_mis_val.describe().columns.values)
    for name in names_data:
        s = data_mis_val[name]
        if name in names_data_numeric:
            if string == statistic_list[0]:
                s = s.fillna(s.mean())
            if string == statistic_list[1]:
                s = s.fillna(s.median())
        else:
            s = s.fillna("missing_value")
        data_mis_val[name] = s
    return data_mis_val


def data_model_constuct(data,data2):
    var = data.isnull().sum()
    data1 = data[var[var==0].index].copy()
    data_model = pd.concat([data1,data2],axis=1)
    data_model = pd.get_dummies(data_model)
    return data_model


def train_val_size(data,val_size,test_size):
    y=data['TARGET']
    data_train, data_valtest, y_train, y_valtest = train_test_split(data, y, test_size=val_size+test_size)
    data_val, data_test, y_val, y_test = train_test_split(data_valtest, y_valtest, test_size=val_size/(val_size+test_size))
    return {'data_train':data_train,'data_val':data_val,'data_test':data_test}


def data_export(dict):
    names_dict = list(dict.keys())
    for i in range(0,3):
        path_to= "data_output/"+names_dict[i]+".csv"
        dict[names_dict[i]].to_csv(path_to)


def prepare_data(path_to,string="mean",percentage=10,val_size=0.2,test_size=0.2):
    data = pd.read_csv(path_to)
    data_mis_val = mis_val_treatment(data,string,percentage)
    data_model = data_model_constuct(data, data_mis_val)
    dict = train_val_size(data_model,val_size,test_size)
    return dict


def prepare_data_brut(path_to,string="mean",percentage=10):
    data = pd.read_csv(path_to)
    data_mis_val = mis_val_treatment(data, string, percentage)
    data_model = data_model_constuct(data, data_mis_val)
    return data_model