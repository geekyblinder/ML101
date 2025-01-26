import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('/content/tmdb_5000_movies.csv')
credits = pd.read_csv('/content/tmdb_5000_credits.csv')
movies = movies.merge(credits,on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

movies.isnull().sum()

import ast
def convert(obj):
  L = []
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['genres']=movies['genres'].apply(convert)

movies['keywords']=movies['keywords'].apply(convert)

def convert3(obj):
  L = []
  ctr = 0
  for i in ast.literal_eval(obj):
    if ctr!=3:
      ctr+=1
      L.append(i['name'])
    else:
      break
  return L

movies['cast']=movies['cast'].apply(convert3)

movies.head()

def fetch_director(obj):
    L = []
    ctr = 0
    for i in ast.literal_eval(obj):
      if i['job'] == 'Director':
        L.append(i['name'])
        break
    return L

movies['crew']=movies['crew'].apply(fetch_director)

movies.head()

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres']= movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])

movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']= movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['overview']= movies['overview'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']= movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df = movies[['movie_id','title','tags']]

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'].apply(lambda x:x.lower())

ps = PorterStemmer()

def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors[0]

cv.get_feature_names_out()

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
  movie_index = new_df[new_df['title'] == movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

  for i in movies_list:
    print(new_df.iloc[i[0]].title)

def __main__():
  try:
    recommend('Superman')
  except:
    print("Movie not found")

if __name__ == "__main__":
    __main__()