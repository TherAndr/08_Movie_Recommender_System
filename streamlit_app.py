#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#!pip install scikit-learn --upgrade
import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


### Preprocessing ######################################################

movies = pd.read_csv("C:/Users/there/Documents/WBS_Coding_School/2.Bootcamp/Project_8_/data/movies.csv")
ratings = pd.read_csv("C:/Users/there/Documents/WBS_Coding_School/2.Bootcamp/Project_8_/data/ratings.csv")

movies.rename(columns={'movieId': 'movieId', 'title':'title', 'genres':'genres'})
ratings.rename(columns={'userId': 'userId', 'movieId': 'movieId', 'rating':'rating', 'timestamp':'timestamp'})
movies["movieId"] = movies["movieId"].astype(str).str.strip()
ratings["movieId"] = ratings["movieId"].astype(str).str.strip()
ratings["userId"] = ratings["userId"].astype(str).str.strip()

# Create big users-items table
users_items = pd.pivot_table(data=ratings,
                            values="rating",
                            index="userId",
                            columns="movieId")

# Replace NaN with zeros
users_items.fillna(0, inplace=True)

# Compute cosine similarities
user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)

### Functions ######################################################
@st.cache_data
def bayes_top_n_movies(n = 5):
  ratings_count = ratings.groupby("movieId").agg({"rating":"mean", "userId":"count"})
  mean_rating = ratings_count["rating"].mean()  # overall average rating
  rating_75 = np.percentile(ratings_count["userId"], 75) # customized bayes - usually 25% here 75% perc.
  ratings_count["new_rating_bayes"] = ((ratings_count["rating"] + (mean_rating * rating_75))
                                        /(ratings_count["userId"] + rating_75))

  movies_ratings = ratings_count.merge(movies, on='movieId', how='inner') # merge so you get the movie names
  
  return (movies_ratings.sort_values(["userId", "new_rating_bayes"], 
                                    ascending=[False, False])
                                    .head(n)) # return sorted movie by number of votes and by bayes ratings

@st.cache_data
def similar_movies(id, n = 5):
  user_item_df = pd.pivot_table(data=ratings, values="rating", index="userId", columns="movieId")
  
  movie_ratings = user_item_df.loc[:,id]
  #movie_ratings = movie_ratings[movie_ratings >= 0] # exclude NaNs # from tania
  similar_to_movie = user_item_df.corrwith(movie_ratings)
  
  corr_movie = pd.DataFrame(similar_to_movie, columns=["PearsonR"])
  corr_movie.dropna(inplace=True)
  
  rating = pd.DataFrame(ratings.groupby("movieId")["rating"].count()).rename(columns={"rating":"rating_count"})
  
  movie_corr_summary = corr_movie.join(rating["rating_count"])
  #movie_corr_summary.drop(id, inplace = True)

  top_n = movie_corr_summary[movie_corr_summary["rating_count"]>=50].sort_values(["PearsonR", "rating_count"], ascending=[False, False]).head(n)

  return top_n.merge(movies, on="movieId", how="inner")

@st.cache_data
def user_recommendation(user_id=123, n = 5):
  weights = (
          user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id])
          )
  # Find restaurants the user has not rated
  not_watched_movies = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]
  # Compute the rating the user would give to unrated movies
  weighted_averages = pd.DataFrame(not_watched_movies.T.dot(weights), columns=["predicted_rating"])
  return weighted_averages.merge(movies, left_index=True, right_on="movieId").sort_values("predicted_rating", ascending=False).head(n)

@st.cache_data
def top_movies_by_genre(selected_genre, n = 5):
  movie_genre = movies[movies["genres"].str.contains(selected_genre)]
  movie_genre_ratings = movie_genre.merge(ratings, on="movieId", how="inner")  
    
  ratings_count = movie_genre_ratings.groupby("movieId").agg({"rating":"mean", "userId":"count"})
  mean_rating = ratings_count["rating"].mean()  # overall average rating
  rating_75 = np.percentile(ratings_count["userId"], 75) # customized bayes - usually 25% here 75% perc.
  ratings_count["new_rating_bayes"] = ((ratings_count["rating"] + (mean_rating * rating_75))
                                        /(ratings_count["userId"] + rating_75))

  movies_ratings = ratings_count.merge(movies, on='movieId', how='inner') # merge so you get the movie names
  
  return (movies_ratings.sort_values(["userId", "new_rating_bayes"], 
                                    ascending=[False, False])
                                    .head(n)) # return sorted movie by number of votes and by bayes ratings

#############################################################################
### Adjusting App ###########################################################

st.title("Get the best Movie Recommendations")
st.write(""" ### """)  
    
### Main

#### build genre drop down menu - choose genre
list = movies["genres"].values.astype(str) 
list = list.tolist()

genre_list = []
for gen in range(len(list)):
    list[gen] = list[gen].replace("|", ",")
    
genre_list = " ".join(list)
genre_list = genre_list.replace(" ", ",")
genre_list = genre_list.split(",")
genre_list = set(genre_list)
genre_list.remove("(no")
genre_list.remove("genres") 
genre_list.remove("listed)")

### Sidebar Slider & Checkboxes
st.sidebar.markdown("## Here you can change values to adjust movie recommendations.")
# set number of recommended movies -> Slider
n = st.sidebar.slider('Number of recommended movies to be displayed: ', min_value=5, max_value=10, step=1)
# set movie
#movie_id = st.sidebar.number_input("Choose movie based on movie_id", min_value=1, step =1)
# set user
user_id = st.sidebar.number_input("Select your user ID", min_value=1, step =1)
    
### choose movie
movie_list = movies.sort_values(by="title", ascending=True)["title"].values #movies["title"].values

selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

movie_id_2 = movies.loc[movies["title"] == selected_movie, "movieId"].values[0]

if st.button("Show Recommendation"):
    st.write((f"### Because you selected {selected_movie}"))
    recommended_movie_names = similar_movies(movie_id_2, n)
    st.write(recommended_movie_names[["title", "genres"]])
    

### sidebar

selected_genre = st.sidebar.selectbox("Type or select a genre from the dropdown", genre_list)

if st.sidebar.button('Show Genre'):
    st.markdown((f"### Because you selected {selected_genre}"))
    genre_movie = top_movies_by_genre(selected_genre, n)
    st.write(genre_movie[["title", "genres"]])
    
st.sidebar.markdown("""### Check the boxes to display:""")  

if st.sidebar.checkbox("Popular Movies"): 
    st.write(f"### Top {n} recommended movies")
    top_movie_recommendations = bayes_top_n_movies(n)
    st.write(top_movie_recommendations[["title", "rating", "genres"]])

if st.sidebar.checkbox("Recommended for you"): 
    st.write(f"### Movie Recommendations for you")
    user_recommendations = user_recommendation(user_id, n)
    st.write(user_recommendations[["title", "genres"]])
