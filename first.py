# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 08:56:52 2024

@author: aniket
"""






import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#load the CSV file
file_path=("C:\10-recommenation_Engine\Entertainment.csv")
data = pd.read_csv(file_path)

#step 1 : Create a user-item matrix
user_item_matrix = data.pivot_table(index='userId' , columns='game' , values='rating')

#pivot table : this function reshapes the dataframe into a matrix where : 
#each row represents a user
#each column represents a game
#the values in the matrix represent the rating that users gave to the games.

#step 2 : fill the NaN values with 0 
user_item_matrix_filled = user_item_matrix.fillna(0)

'''this line replaces only missing values
in the user item matrix with 0 indicating that the
user did not rate that particular game.'''

#step 3 : 
user_similarity = cosine_similarity(user_item_matrix_filled)

#convert similarity matrix to a dataframe for easy reference
user_similarity_df = pd.DataFrame(user_similarity,index=user_item_matrix.index,columns=user_item_matrix.index)

#step 4:
def get_collaborative_recommendations_for_user(user_id,num_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    
    similar_users = similar_users.drop(user_id)
    
    #select the top N similar users to limit noise
    top_similar_users = similar_users.head(50)

    weighted_ratings = np.dot(top_similar_users.values,user_item_matrix_filled.loc[top_similar_users.index])
    
    #normalize by the sum of similarities
    sum_of_similarities = top_similar_users.sum()
    
    if sum_of_similarities > 0:
        weighted_ratings /= sum_of_similarities
        
        
#recommend games that the user hasn't rated yet
     user_ratings = user_item_matrix_filled.loc[user_id]
     unrated_games = user_ratings[user_ratings == 0]

#get the weighted score for unrated games
game_recommendations = pd.Series(weighted_ratings,index=user_item_matrix_filled.columns).loc[unrated_games.index]



print('recommendation game for user 3 :')
print(recommendation_games)












