import pandas as pd
import os

class Dataset:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.normalize()
    
    def normalize(self):
        to_replace = {}
        self.attr = {}
        for class_code,class_name in enumerate(self.data.iloc[:,-1].unique()):
            to_replace[class_name] = class_code
            self.attr[class_code] = class_name
        self.data = self.data.replace(to_replace)
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def debug(self):
        print(self.data.iloc[:,:-1])