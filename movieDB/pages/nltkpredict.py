import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string

from nltk.corpus import stopwords
# for creating word cloud using the list of words
from wordcloud import WordCloud
# for feature extraction
from sklearn.feature_extraction.text import CountVectorizer
# for splitting training and testing dataset
from sklearn.model_selection import train_test_split
# Multinomial Naïve Bayes classifier
from sklearn.naive_bayes import MultinomialNB

# for checking the model accuracy, classification report, and confusion matrix
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score

# for ignoring any warnings

df = pd.read_csv('C:/Users/מיכאל/Desktop/projects/MOVIE/IMDB-Movie-Database-Django-master/IMDB-Movie-Database-Django-master/Data/moviefeedback.csv')
df.drop_duplicates(inplace=True)
df = df.dropna()

df['text'] = df['text'].astype(str)
df['type'] = df['type'].astype(str)
df.rename(columns = {'type':'label', 'text':'message'}, inplace=True)

Le = LabelEncoder()
df['label']=Le.fit_transform(df['label'])

label_count = df.label.value_counts()



X = df['message'] #text data
y = df['label'] #target label

cv = CountVectorizer()
X =  cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
classifier = MultinomialNB().fit(X_train, y_train)
pred_train = classifier.predict(X_train)
pred_test = classifier.predict(X_test)

def sms(text):
    text = [text]
    # creating a list of labels
    lab = ['negative','positive']
    # perform tokenization
    X = cv.transform(text).toarray()
    # predict the text
    p = classifier.predict(X)
    # convert the words in string with the help of list
    s = [str(i) for i in p]
    a = int("".join(s))
    # show out the final result
    res = str("This message is "+ lab[a])
    return res

if __name__ == '__main__':
  sms("")

