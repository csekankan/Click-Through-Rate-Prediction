from sklearn.metrics import log_loss,accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import precision_score,recall_score
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def trainmodel(name,model,train,label,train_test_frac):
    X = train.drop([label], axis=1)
    y = train[label]
    x_train,x_cv,y_train,y_cv=train_test_split(X,y,test_size=train_test_frac)
    model.fit(x_train,y_train)
    pred_ans  = model.predict(x_cv)
    res = {'Metrics': ['Model','LogLoss', 'roc_auc_score', 'f1 score', 'precision score','recall score','accuracy score'],
        'Value': [name,round(log_loss(pred_ans, y_cv), 4)\
                   , round(roc_auc_score(pred_ans, y_cv), 4),\
                   round(f1_score(pred_ans, y_cv), 4),\
                      round(precision_score(pred_ans, y_cv), 4)\
                            , round(recall_score(pred_ans, y_cv), 4),\
                                 round(accuracy_score(pred_ans, y_cv), 4)]}
  
    # Create DataFrame
    df = pd.DataFrame(res)
    return df
        
def traintest_split(train,label,train_test_frac):
    X = train.drop([label], axis=1)
    y = train[label]
    x_train,x_cv,y_train,y_cv=train_test_split(X,y,test_size=train_test_frac,random_state=1)     

def run(modelname,train,cat_features,continue_var,label,train_test_frac,lg_max_depth=9,
               lg_pos_bagging_fraction=1,
               lg_neg_bagging_fraction=1,
               lg_feature_fraction=1,
               lg_numestimator=100,
               lg_learning_rate=0.1,
               lg_num_leaves=50,
               lg_is_unbalanced=False,
               xg_learning_rate=.1,
               xg_max_depth=9,
               xg_numestimator=100,
               ds_max_depth=9 ,
               rm_penalty='l2',
               rf_n_estimators=1000,
               rf_max_depth=9
               ):
    # all_feat=[]
    # for i in cat_features:
    #     try:
    #       all_feat.append(i)
    #     except:
    #         pass
    # for i in continue_var:
    #     try:
    #       all_feat.append(i)
    #     except:
    #         pass
          
    if modelname=='Logistic Regression':
           model = LogisticRegression(random_state=1)
           return trainmodel(modelname,model,train,label,train_test_frac)
    
    if modelname=='LightGbm':
          
           model = LGBMClassifier(boosting_type='gbdt', 
                        learning_rate=lg_learning_rate, max_depth=lg_max_depth, 
                        n_estimators=lg_numestimator,
                         num_leaves=lg_num_leaves, objective='binary', 
                        silent=True,is_unbalanced=lg_is_unbalanced,
                        )
           return trainmodel(modelname,model,train,label,train_test_frac)



    if modelname=='XGBoost':
           model =  XGBClassifier( booster='gbtree', 
                 gamma=0, learning_rate=xg_learning_rate,
                    max_depth = xg_max_depth, min_child_weight=1,  
                    n_jobs=1,  objective='binary:logistic', 
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1,n_estimators=xg_numestimator)
           return trainmodel(modelname,model,train,label,train_test_frac)
    

            
           


    if modelname=='Random Forest':
           
           model = RandomForestClassifier(n_estimators = rf_n_estimators,
                             max_depth = rf_max_depth,
                             criterion = "gini")
           return trainmodel(modelname,model,train,label,train_test_frac)
     
    if modelname=='Decision Tree':
           model =  DecisionTreeClassifier(max_depth = ds_max_depth,
                                       random_state = 123,
                                       splitter  = "best",
                                       criterion = "gini",
                                      )
           return trainmodel(modelname,model,train,label,train_test_frac)
    
   
           
           

   



    