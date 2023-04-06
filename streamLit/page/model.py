
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from page import dl,ml
def run(modelname,train,label,cat_features,continue_var,train_test_frac,lg_max_depth=9,
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
               rf_max_depth=9,dnn_hidden_units="512,128, 512,256", dnn_act='sigmoid',
              batch_size=2024, epochs=5):
    if modelname =='':
        return None
    if  modelname  in ['xDeepFM','DeepFM','Wide & Deep','Deep Cross']:
        res= dl.run(modelname,train,cat_features,continue_var,label,train_test_frac,
                    dnn_hidden_units =dnn_hidden_units , dnn_act =dnn_act,
              batch_size=batch_size, epochs=epochs)
        st.write(res)
        return res
   
    if  modelname  in ['Logistic Regression','Decision Tree','Random Forest','LightGbm','XGBoost']:
        res= ml.run(modelname,train,cat_features,continue_var,label,train_test_frac,lg_max_depth=lg_max_depth,
               lg_pos_bagging_fraction=lg_pos_bagging_fraction,
               lg_neg_bagging_fraction=lg_neg_bagging_fraction,
               lg_feature_fraction=lg_feature_fraction,
               lg_numestimator=lg_numestimator,
               lg_learning_rate=lg_learning_rate,
               lg_num_leaves=lg_num_leaves,
               lg_is_unbalanced=lg_is_unbalanced,
               xg_learning_rate=xg_learning_rate,
               xg_max_depth=xg_max_depth,
               xg_numestimator=xg_numestimator,
               ds_max_depth=ds_max_depth ,
               rm_penalty=rm_penalty,
               rf_n_estimators=rf_n_estimators,
               rf_max_depth=rf_max_depth)
        


        if(res is not None ):
                tab1, tab2 = st.tabs(["Data set","Results"] )
                
                with tab1:
                  st.pyplot(train[label].value_counts().plot(kind='bar', title='Target Classes').figure)
                
                with tab2:
                   st.write(res)
                
            
      
        
        return res
    
       
    