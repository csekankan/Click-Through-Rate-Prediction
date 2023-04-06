import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


import page
from page import dataLoader,preprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import os
import pathlib
from os import listdir
from os.path import isfile, join



"""
# Click Through Rate Prediction
"""


     
if __name__ == "__main__":
    
    df,cat_columns,nu_columns,labl=dataLoader.run()
    if 'df' not in st.session_state:
        st.session_state.df = df
    preprocess.construct_sidebar(df,cat_columns,nu_columns,labl)
