
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder ,MinMaxScaler
from page import model
def construct_sidebar(dfip,cat_cols,nu_col,label):
        df=dfip.copy()
        st.sidebar.markdown(
            '<p class="header-style"><h2>Model Training</h2></p>',
            unsafe_allow_html=True
        )
        if df is None or df.shape[0]==0:
             return 
        cols = [col for col in df.columns]
        fill_mean = lambda x: x.fillna(x.mean())
      
        for col in nu_col:
           
            if(len(col)>0):
             
               print(df[[col]].head(1))
               df[col] = df[col].fillna(df[col].mean())
               df[col] = df[col].astype('float64')
        df = df.fillna('-1')
        
              
        mms = MinMaxScaler(feature_range=(0, 1))
        if(df is not None and len(nu_col)>0):
                    df[nu_col] = mms.fit_transform(df[nu_col])
        #st.write(df.head(5))
        train_test = st.sidebar.text_input("Train and Test Split", value="0.2")
        train_test=float(train_test)
        model_selected = st.sidebar.selectbox(
            f"Select {cols[0]}",
           ['DeepFM','xDeepFM','Wide & Deep','Deep Cross','LightGbm','XGBoost','Logistic Regression','Decision Tree','Random Forest']
        )
        if(model_selected in ['DeepFM','xDeepFM','Wide & Deep','Deep Cross'] ):
               dnn_hidden_units=st.sidebar.text_input("dnn_hidden_units", value="512,128, 512,256")
               dnn_act=st.sidebar.selectbox(
                f"Select Activation",
                 ['sigmoid','relu']
                 )
               batch_size=int(st.sidebar.text_input("batch_size", value="512"))
               epochs=int(st.sidebar.text_input("epochs", value="5"))
               if st.sidebar.button('Train',key='dl'):
                    stats=model.run(model_selected,df,label,cat_cols,nu_col,train_test_frac=train_test,
                                    dnn_hidden_units=dnn_hidden_units,dnn_act=dnn_act,batch_size=batch_size,epochs=epochs)
        if(model_selected in ['Logistic Regression'] ):
             
               if st.sidebar.button('Train',key='lr'):
                    stats=model.run(model_selected,df,label,cat_cols,nu_col,train_test_frac=train_test)
               
        if(model_selected in ['Random Forest'] ):
               rf_n_estimators=int(st.sidebar.text_input("n_estimators", value="1000"))
               rf_max_depth=int(st.sidebar.text_input("max_depth", value="9"))
               if st.sidebar.button('Train',key='rm'):
                    stats=model.run(model_selected,df,label,cat_cols,nu_col,
                                    train_test_frac=train_test,rf_n_estimators=rf_n_estimators,rf_max_depth=rf_max_depth)

        if(model_selected in ['Decision Tree'] ):  
               ds_max_depth=int(st.sidebar.text_input("max_depth", value="9") )
               if st.sidebar.button('Train',key='dt'):
                    stats=model.run(model_selected,df,label,cat_cols,nu_col,train_test_frac=train_test)

        if(model_selected in ['LightGbm'] ):  
               #st.write(df)
               lg_max_depth=int(st.sidebar.text_input("max_depth", value="9"))
               lg_learning_rate=float(st.sidebar.text_input("learning_rate", value=".1"))
               lg_num_leaves=int(st.sidebar.text_input("num_leaves", value="50"))
               lg_numestimator=int(st.sidebar.text_input("n_estimators", value="100"))

               lg_is_unbalanced=st.sidebar.selectbox(
                f"is_unbalanced",
                 ['False','True']
                 )
               lg_is_unbalanced=(lg_is_unbalanced=='True')
               if st.sidebar.button('Train',key='lgb'):
                    stats=model.run(model_selected,df,label,cat_cols,nu_col,train_test_frac=train_test,
                                    lg_max_depth=lg_max_depth,
                                   
                                    lg_learning_rate=lg_learning_rate,lg_num_leaves=lg_num_leaves,lg_numestimator=lg_numestimator,
                                    lg_is_unbalanced=lg_is_unbalanced
                                    )
        if(model_selected in ['XGBoost'] ):  
               xg_learning_rate=float(st.sidebar.text_input("learning_rate", value=".1"))
               xg_max_depth=int(st.sidebar.text_input("max_depth", value="9"))
               xg_numestimator=int(st.sidebar.text_input("n_estimators", value="100"))
               if st.sidebar.button('Train',key='xgb'):
                    stats=model.run(model_selected,df,label,cat_cols,nu_col,train_test_frac=train_test)
               
           
                 
        return df
        
        