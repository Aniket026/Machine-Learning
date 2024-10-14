# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:36:53 2024

@author: aniket

"""





import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


file_path="C:/10-recommenation_Engine/game.csv"

data=pd.read_csv(file_path)


user_item_matrix=data.pivot_table(index='userId', columns="game",values='rating')

user_item_matrix_filled=user_item_matrix.fillna(0)

user_similarity=cosine_similarity(user_item_matrix_filled)

user_similarity_df=pd.DataFrame(user_similarity,index=user_item_matrix.index,columns=user_item_matrix.index)

def get_collabratives_recommendation_for_user(user_id,num_recommendations=5):
    similar_users=user_similarity_df[user_id].sort_value(accending=False)
    similar_users=similar_users.drop(user_id)
    top_similar_user=similar_users.head(50)
    weight_rating=np.dot(top_similar_users.values,user_item_matrix_filled.loc[top_similar_users.index])







