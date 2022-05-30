import uvicorn
from fastapi import FastAPI


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

df_links = pd.read_csv("links.csv")
df_movies = pd.read_csv("movies.csv")
df_ratings = pd.read_csv("ratings.csv")
df_tags = pd.read_csv("tags.csv")


df_movies['genres'] = df_movies.genres.str.split('|')
df_movies['released'] =df_movies['title'].str.extract('.*\((.*)\).*',expand = False)
df_movies['title']=df_movies.title.str.slice(0,-7)
movies_ratings=pd.merge(left=df_ratings,right=df_movies,on='movieId')
movies_ratings = movies_ratings[['userId', 'movieId', 'title', 'genres', 'rating', 'timestamp']]



# First Model: Item CF

data = pd.pivot(index = 'movieId',columns = 'userId', data = df_ratings, values ='rating')
numberOf_user_voted_for_movie = pd.DataFrame(df_ratings.groupby('movieId')['rating'].agg('count'))
numberOf_user_voted_for_movie.reset_index(level = 0,inplace = True)
numberOf_movies_voted_by_user = pd.DataFrame(df_ratings.groupby('userId')['rating'].agg('count'))     
numberOf_movies_voted_by_user.reset_index(level = 0,inplace = True)
data.fillna(0,inplace = True)
data_final = data.loc[numberOf_user_voted_for_movie[numberOf_user_voted_for_movie['rating'] > 10]['movieId'],:]
data_final = data_final.loc[:,numberOf_movies_voted_by_user[numberOf_movies_voted_by_user['rating'] > 60]['userId']]
from scipy.sparse import csr_matrix
csr_data = csr_matrix(data_final.values)
data_final.reset_index(inplace=True)

#using knn and writing a function to get list of recommended movies
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn.fit(csr_data)
def get_cf_recommendation(movie_name):
    n= 10
    movie_list = df_movies[df_movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId'] #movieId
        movie_idx = data_final[data_final['movieId'] == movie_idx].index[0] #userId acc to movieId
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze(),distances.squeeze())),key=lambda x: x[1])[1::1]
        recommend = []
        recommend2 = []
        for val in rec_movie_indices:
            movie_idx = data_final.iloc[val[0]]['movieId']
            idx = df_movies[df_movies['movieId'] == movie_idx].index
            recommend.append(df_movies.iloc[idx]['title'].values[0])
            recommend2.append(val[1])         
        df1 = pd.DataFrame(recommend)
        df2 = pd.DataFrame(recommend2)
        df = pd.concat([df1,df2],axis = 'columns')
        df.columns = ['Title','Distance']
        df.set_index('Distance',inplace = True)
        return df1
    else:
        return "No movies found. Please check your input"



# Second Model: Content based TFIDF

import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

content_data = df_movies[['title','genres']]
content_data = content_data.astype(str)
content_data['content'] = content_data['title'] + ' ' + content_data['genres']
content_data = content_data.reset_index()
indices = pd.Series(content_data.index,  index=content_data['title'])
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_data['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# #Function to return a list of movies to be recommended
# def get_content_based_recommendations(title, similarity=cosine_sim, n_sim=10):
#     idx = indices[title]
    
#     # Get the pairwsie similarity scores of all movies with given movie
#     sim_scores = list(enumerate(similarity[idx]))

#     # Sort the movies based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the scores of the n_sim most similar movies
#     if n_sim > 0:
#         n = n_sim + 1
#         sim_scores = sim_scores[1:n]
#     else:    
#         sim_scores = sim_scores[1:11]
    
#     movie_indices = [i[0] for i in sim_scores]

#     # Return the n_sim most similar movies
#     recomm_movies_content_based=list(content_data['title'].iloc[movie_indices])
#     recs2=[]
#     for mov in recomm_movies_content_based:
#         recs2.append(mov)
#     recommendations_content_based=pd.DataFrame(recs2)
#     return recommendations_content_based


#Third Model: User based similarity

#finding count and mean of ratings for all movies

movies_ratings['count'] = 1
data_pivot_for_count_rating = movies_ratings.drop(['userId'], axis = 'columns').pivot_table(index = ['title'], aggfunc = {'count': 'sum', 'rating': 'mean'})
data_pivot_for_count_rating = data_pivot_for_count_rating.sort_values(by = ['count'], ascending = False)
movie_data = movies_ratings.drop(['count'], axis = 'columns').pivot_table(index = 'userId', columns = 'movieId', values = 'rating')
def calculate_cosine_similarity(x1, x2):
    numerator = np.dot(np.where(np.isnan(x1), 0, x1), np.where(np.isnan(x2), 0, x2))
    x1_squared = np.dot(np.where(np.isnan(x1), 0, x1), np.where(np.isnan(x1), 0, x1))
    x2_squared = np.dot(np.where(np.isnan(x2), 0, x2), np.where(np.isnan(x2), 0, x2))
    denominator = np.sqrt(x1_squared * x2_squared)
    return numerator / denominator
function_dict = {}
function_dict['cosine'] = calculate_cosine_similarity
list_userId = movie_data.index

def recommend_movie_by_similar_user(user):
    method = 'cosine'
    n_recommend = 10
    accept_rating = 4.5
    movies_data = df_movies
    
    # initial rating correlation dictionary and recommend movieId list
    rating_simility = {}
    recommend_movieId = []
    
    # calculate rating correlation (all users)
    for userId in list_userId:
        rating_simility[userId] = function_dict[method](movie_data.loc[user, :], movie_data.loc[userId, :])
        
    # pick most similar userId
    i = 0
    while len(recommend_movieId) < n_recommend:
        i += 1
        most_similar_userId = sorted(rating_simility, key = rating_simility.get, reverse  = True)[i]
        most_similar_userId_rating = movie_data.loc[most_similar_userId,:]
        rating_list = sorted(pd.Series(np.where(np.isnan(most_similar_userId_rating), 0, most_similar_userId_rating)).unique(), reverse = True)
        for rate_score in rating_list:
            if rate_score >= accept_rating:
                max_rating_movieId = list(most_similar_userId_rating[most_similar_userId_rating == rate_score].index)
                recommend_movieId.extend([movieid for movieid in max_rating_movieId if np.isnan(movie_data.loc[user, movieid])])
                if len(recommend_movieId) >= n_recommend:
                    break
            else:
                break

    movies_data_pivot = movies_data.set_index(['movieId'])
    recommend_movies = list(movies_data_pivot.loc[recommend_movieId[:n_recommend], 'title'])
#     print('recommend movies for user %d:\n' %(user))
    recommend_movies_for_user=pd.DataFrame(recommend_movies)
    return recommend_movies_for_user



app=FastAPI()


@app.get("/title-item-based-cf/{movie}")
def Enter_movie_name_to_get_similar_movies(movie):
    return get_cf_recommendation(movie)


# @app.get("/title-content-based-tfidf/{mov}")
# def Enter_movie_name_to_get_similar_movies(mov):
#     return get_content_based_recommendations(mov)


@app.get("/user-similarity-based-/{user}")
def Enter_user_id_to_get__movies_suggested_by_similar_users(user: int):
    return recommend_movie_by_similar_user(user)

