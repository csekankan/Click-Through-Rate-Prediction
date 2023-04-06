
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def file_up(col,separator):
        print(separator)
        df=pd.DataFrame()
        w = st.file_uploader("Upload a data")
        
        if w is not None:
                 if(separator=='\\t'):
                        df = pd.read_csv(w,sep='\t' ,names=col)
                 else:
                      df = pd.read_csv(w,sep=separator ,names=col)
     
                
        return df
def run():
     allcols = st.text_input("Enter all column names separated by comma(,)", value="")   
     
     labl = st.text_input("Enter Label Column name", value="")   
     nu_columns = st.text_input("Enter Numerical Column names separated by comma(,)", value="")        
     cat_columns = st.text_input("Enter Categorical Column names separated by comma(,)", value="")
     fraction = st.text_input("fraction of data you want to process", value="0.01")
     separator = st.selectbox(
                f"Data separator",
                 [',','|' , '\\t']
                 )  
     df= pd.DataFrame()
     allcols=allcols.strip().split(",")
     cat_columns=cat_columns.strip().split(",")
     nu_columns=nu_columns.strip().split(",")
     fraction=float(fraction)
     allcols=[x.strip() for x in allcols]
     cat_columns=[x.strip() for x in cat_columns]
     nu_columns=[x.strip() for x in nu_columns]
     labl=labl.strip()
     cols=[]
     for i in  allcols:
           cols.append(i)
     # for i in  nu_columns:
     #      cols.append(i)

     df=file_up(cols,separator)

     # Remove id columns
     drop_unique_ids=[]

     for col in (list(df.columns)):
          if len(list(df[col].unique())) == df.shape[0]:
              drop_unique_ids.append(col)
     if(len(drop_unique_ids)>0):
            df.drop(drop_unique_ids, axis=1, inplace=True)
            for i in drop_unique_ids:
                  try:
                        cat_columns.remove(i)
                  except:
                       pass
            for i in drop_unique_ids:
                  try:
                        nu_columns.remove(i)
                  except:
                       pass
            for i in drop_unique_ids:
                  try:
                        allcols.remove(i)
                  except:
                       pass


     if df is not None and df.shape[0]>0:
        df=df.sample(frac=fraction)
        if(cat_columns is not None and len(cat_columns)>0):
          for c in cat_columns:
              if(len(c)>0):
               le = LabelEncoder()
               le.fit(df[c])
               df[c] = le.transform(df[c])
        

    
     return df,cat_columns,nu_columns,labl
         

    	
   
    	



