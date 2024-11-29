import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from stasmodel.api import sm
import os 

class MakeDataSet:
    def __init__(self, json_file_path: str, label: int,SexDataset:pd.DataFrame):
        self.SexRates = pd.DataFrame(json.loads(json_file_path))
        self.label = label # =0: Male , = 1:Female
        self.SexDataset = SexDataset
        self.bath_dir = "/Users/uemuraminato/Desktop/book_script/vec/vec_gen/"

    def make_data(self):
        dataset = pd.DataFrame()
        for _ , row in self.SexRates.iterrows():
            target_id = row["expert_id"]  # "expert_id"を取得
            sr = row["sr"]  
            # データを探してくる
            # word2vecの結果でもいいかな
            np_vec_path = self.search_ExpertPath(search_id=str(target_id))
            np_vec = np.load(np_vec_path)
            cos_mean = self.calc_cos_mean(dataset=np_vec)
            target_data = pd.DataFrame([[self.label,sr,cos_mean]],columns=["BSex","SSex","CosMean"])
            self.SexDataset = pd.concat([self.SexDataset,target_data],ignore_index=True)
        return self.SexDataset
    def calc_cos_mean(self, dataset):
        similality_matrix = cosine_similarity(dataset)
        triu_indices = np.tril_indices_from(similality_matrix, k=-1)#k = -1 は対角上の要素以外という意味
        similalities = similality_matrix[triu_indices]
        return similalities.mean()
    
    def search_ExpertPath(self, search_id):
        abs_path = None
        for root, dir, files in os.walk(self.bath_dir):
            for file in files:
                if search_id in file:
                    relative_path = os.path.join(root, file)
                    abs_path = os.path.abspath(relative_path)
                    return abs_path
        if abs_path is None:
            raise ValueError(f"No {search_id} files")

class Verify:
    def __init__(self,json_files:list):
        self.json_files = json_files
    def Load_Dataset(self):
        Sdataset = 
    def LinearReg(self):
