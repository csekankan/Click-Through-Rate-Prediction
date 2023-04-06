import shutil
import os
from datetime import datetime

import streamlit as st

def run():
    now = datetime.now()

    src = os.getcwd()
    print(src)
    src1=os.path.join(src,"template","rf.py")
    file=now.strftime("%Y-%m-%d %H:%M:%S")+".py"
    dst1 =  os.path.join(src,"template","tmp","rf"+file)
    shutil.copyfile(src1, dst1)


    with open(dst1, 'r') as file:

        data = file.read()
    
        data = data.replace('<filename>',st.session_state['filename'] )

    with open(dst1, 'w') as file:

        file.write(data)
    
    # Printing Text replaced
    print("Text replaced")