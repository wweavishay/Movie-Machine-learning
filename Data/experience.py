
import pandas as pd

dfcomment = pd.read_csv(
    "C:/Users/מיכאל/Desktop/projects/MOVIE/IMDB-Movie-Database-Django-master/IMDB-Movie-Database-Django-master/Data/reviews.csv")
dfcomment = dfcomment[dfcomment['idofmovie'] == 2]
list1 = []
for row in dfcomment['type']:
    list1.append(int(row) *"a")

dfcomment['type'] = list1
print(dfcomment['type'] )