import os
from datetime import datetime
import pandas as pd

class ComputeFile: 

    def __init__(self, input_path=''):
        self.input_path = input_path
        self.input_files = []
        self.extensions = ['.xml']
    
    def accept_extension(self, file='') :
        for ext in self.extensions :
            if file.endswith(ext) :
                return True
        return False

    def build_list_files(self):
        """
            building the list of input and output files
        """
        output = []
        for current_path, folders, files in os.walk(self.input_path):
            for file in files:
                if self.accept_extension(file=file):
                    tmp_current_path = os.path.join(current_path, file)
                    output.append(tmp_current_path)
        return output

    def get_data(self, filename=''):
        df = pd.read_csv(filename)
        columns = df.columns.tolist()
        datas = []
        length = df.shape[0]
        for i in range(0, length):
            tmp = []
            tmp_data = df.iloc[i]
            for column in columns :
                tmp.append(tmp_data[column])
            datas.append(tmp)
        return (datas, df)
        
        
