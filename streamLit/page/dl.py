from sklearn.metrics import log_loss,accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import precision_score,recall_score
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

from deepctr_torch.inputs import SparseFeat, get_feature_names,DenseFeat
from deepctr_torch.models import DeepFM,DCN
from deepctr_torch.models import WDL

from deepctr_torch.models import xDeepFM
import torch


def datapreprocessor(train,cat_features,continue_var,label,train_test_frac):
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=train[feat].max() + 1, embedding_dim=15)
                              for feat in cat_features] + [DenseFeat(feat, 1, )
                                                              for feat in continue_var]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    


    return feature_names,linear_feature_columns,dnn_feature_columns

def split_train_test(train,feature_names):

    train, test = train_test_split(train, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    return train, test,train_model_input,test_model_input
def runDeepFM(train,test,target,linear_feature_columns,dnn_feature_columns,train_model_input,
               test_model_input,device,  dnn_hidden_units, dnn_act,
               batch_size, epochs):
    dnn_hidden_units= dnn_hidden_units. split(',') 
    dnn_hidden_units=[int(i.strip()) for i in dnn_hidden_units]
    dnn_hidden_units= tuple(dnn_hidden_units)
    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                       task='binary',dnn_hidden_units=dnn_hidden_units,use_fm=True,  l2_reg_linear=0.1,
                       l2_reg_embedding=1e-1,  dnn_use_bn=True,device=device,dnn_activation=dnn_act)
    model.compile("adagrad", "binary_crossentropy",
                      metrics=["binary_crossentropy", "auc"], )
    history = model.fit(train_model_input, train[target].values, batch_size=batch_size, epochs=epochs, verbose=2,
                            validation_split=0.15)

    pred_ans = model.predict(test_model_input, 256)
    ypred=[]
    for i in pred_ans:
        ypred.append(round(i[0]))


    res = {'Metrics': ['Model','LogLoss', 'roc_auc_score', 'f1 score', 'precision score','recall score','accuracy score'],
        'Value': ['DeepFM',round(log_loss(test[target].values, pred_ans), 4)\
                   , round(roc_auc_score(test[target].values,ypred), 4),\
                   round(f1_score(test[target].values,ypred), 4),\
                      round(precision_score(test[target].values,ypred), 4)\
                            , round(recall_score(test[target].values,ypred), 4),\
                                 round(accuracy_score(test[target].values,ypred), 4)]}
  
        # Create DataFrame
    df = pd.DataFrame(res)
    return df
        



def runXDeepFM(train,test,target,linear_feature_columns,dnn_feature_columns,train_model_input,
               test_model_input,device,  dnn_hidden_units, dnn_act,
               batch_size, epochs):
    dnn_hidden_units= dnn_hidden_units. split(',') 
    dnn_hidden_units=[int(i.strip()) for i in dnn_hidden_units]
    dnn_hidden_units= tuple(dnn_hidden_units)
    print(dnn_hidden_units)
    model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',dnn_hidden_units=dnn_hidden_units,  l2_reg_linear=0.1,
                   l2_reg_embedding=1e-1,  dnn_use_bn=True,device=device,dnn_activation=dnn_act)
    model.compile("adagrad", "binary_crossentropy",
                      metrics=["binary_crossentropy", "auc"], )
    history = model.fit(train_model_input, train[target].values, batch_size=batch_size, epochs=epochs, verbose=2,
                            validation_split=0.15)

    pred_ans = model.predict(test_model_input, 256)
    ypred=[]
    for i in pred_ans:
        ypred.append(round(i[0]))


    res = {'Metrics': ['Model','LogLoss', 'roc_auc_score', 'f1 score', 'precision score','recall score','accuracy score'],
        'Value': ['xDeepFM',round(log_loss(test[target].values, pred_ans), 4)\
                   , round(roc_auc_score(test[target].values,ypred), 4),\
                   round(f1_score(test[target].values,ypred), 4),\
                      round(precision_score(test[target].values,ypred), 4)\
                            , round(recall_score(test[target].values,ypred), 4),\
                                 round(accuracy_score(test[target].values,ypred), 4)]}
  
        # Create DataFrame
    df = pd.DataFrame(res)
    return df
        


def runWDL(train,test,target,linear_feature_columns,dnn_feature_columns,train_model_input,
               test_model_input,device,  dnn_hidden_units, dnn_act,
               batch_size, epochs):
    dnn_hidden_units= dnn_hidden_units. split(',') 
    dnn_hidden_units=[int(i.strip()) for i in dnn_hidden_units]
    dnn_hidden_units= tuple(dnn_hidden_units)
    model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',dnn_hidden_units=dnn_hidden_units,  l2_reg_linear=0.1,
                   l2_reg_embedding=1e-1,  dnn_use_bn=True,device=device,dnn_activation=dnn_act)
    model.compile("adagrad", "binary_crossentropy",
                      metrics=["binary_crossentropy", "auc"], )
    history = model.fit(train_model_input, train[target].values, batch_size=batch_size, epochs=epochs, verbose=2,
                            validation_split=0.15)

    pred_ans = model.predict(test_model_input, 256)
    ypred=[]
    for i in pred_ans:
        ypred.append(round(i[0]))


    res = {'Metrics': ['Model','LogLoss', 'roc_auc_score', 'f1 score', 'precision score','recall score','accuracy score'],
        'Value': ['Wide and Deep',round(log_loss(test[target].values, pred_ans), 4)\
                   , round(roc_auc_score(test[target].values,ypred), 4),\
                   round(f1_score(test[target].values,ypred), 4),\
                      round(precision_score(test[target].values,ypred), 4)\
                            , round(recall_score(test[target].values,ypred), 4),\
                                 round(accuracy_score(test[target].values,ypred), 4)]}
        # Create DataFrame
    df = pd.DataFrame(res)
    return df


def runDCN(train,test,target,linear_feature_columns,dnn_feature_columns,train_model_input,
               test_model_input,device,  dnn_hidden_units, dnn_act,
               batch_size, epochs):
    dnn_hidden_units= dnn_hidden_units. split(',') 
    dnn_hidden_units=[int(i.strip()) for i in dnn_hidden_units]
    dnn_hidden_units= tuple(dnn_hidden_units)
    model = DCN(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',dnn_hidden_units=dnn_hidden_units,  l2_reg_linear=0.0001,
                   l2_reg_embedding=1e-4,  dnn_use_bn=True,device=device,dnn_activation=dnn_act)
    model.compile("adagrad", "binary_crossentropy",
                      metrics=["binary_crossentropy", "auc"], )
    history = model.fit(train_model_input, train[target].values,batch_size=batch_size, epochs=epochs, verbose=2,
                            validation_split=0.15)

    pred_ans = model.predict(test_model_input, 256)
    ypred=[]
    for i in pred_ans:
        ypred.append(round(i[0]))


    res = {'Metrics': ['Model','LogLoss', 'roc_auc_score', 'f1 score', 'precision score','recall score','accuracy score'],
        'Value': ['Deep Cross',round(log_loss(test[target].values, pred_ans), 4)\
                   , round(roc_auc_score(test[target].values,ypred), 4),\
                   round(f1_score(test[target].values,ypred), 4),\
                      round(precision_score(test[target].values,ypred), 4)\
                            , round(recall_score(test[target].values,ypred), 4),\
                                 round(accuracy_score(test[target].values,ypred), 4)]}
  
        # Create DataFrame
    df = pd.DataFrame(res)
    return df
        
        


def run(modelname,train,cat_features,continue_var,label,train_test_frac,  dnn_hidden_units
               ,dnn_act,batch_size, epochs):
    feature_names,linear_feature_columns,dnn_feature_columns=datapreprocessor(train,cat_features,continue_var,label,train_test_frac)
    train, test,train_model_input,test_model_input=split_train_test(train,feature_names)

    device = 'cpu'
    use_cuda = True


    # if use_cuda and torch.cuda.is_available():
    #         print('cuda ready...')
    #         device = 'cuda:0'

    if modelname=='DeepFM':
         return runDeepFM(train,test,label,linear_feature_columns,dnn_feature_columns,train_model_input,test_model_input,device,dnn_hidden_units,dnn_act,batch_size, epochs)

    if modelname=='xDeepFM':
         return runXDeepFM(train,test,label,linear_feature_columns,dnn_feature_columns,train_model_input,test_model_input,device,dnn_hidden_units,dnn_act,batch_size, epochs)

    if modelname=='Wide & Deep':
         return runWDL(train,test,label,linear_feature_columns,dnn_feature_columns,train_model_input,test_model_input,device,dnn_hidden_units,dnn_act,batch_size, epochs)
    if modelname=='Deep Cross':
         return runDCN(train,test,label,linear_feature_columns,dnn_feature_columns,train_model_input,test_model_input,device,dnn_hidden_units,dnn_act,batch_size, epochs)



    



    