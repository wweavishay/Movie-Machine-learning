import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# for label encoding
from sklearn.preprocessing import LabelEncoder
# for working with text data
import string
import re
# natural language toolkit
import nltk


# for removing the stopwords from text data
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


df = pd.read_csv('C:/Users/מיכאל/Desktop/projects/MOVIE/IMDB-Movie-Database-Django-master/IMDB-Movie-Database-Django-master/Data/tripadvisor_hotel_reviews.csv')
df1 = pd.read_csv('C:/Users/מיכאל/Desktop/projects/MOVIE/IMDB-Movie-Database-Django-master/IMDB-Movie-Database-Django-master/Data/reviews.csv')

df = pd.concat([df,df1])
#remove duplicates
df.drop_duplicates(inplace=True)
df = df.dropna()

df['text'] = df['text'].astype(str)
df['type'] = df['type'].astype(str)
df.rename(columns = {'type':'label', 'text':'message'}, inplace=True)

#print("Shape of dataset is {}".format(df.shape))
#print("Duplicates present {}".format(df.duplicated().sum()))
#print('-'*50)


#print("Shape of dataset after removing duplicates is {}".format(df.shape))
#print('-'*50)

#label encoding
Le = LabelEncoder()
df['label']=Le.fit_transform(df['label'])
#print('0 means.... {}'.format(Le.inverse_transform([0])))
#print('1 means.... {}'.format(Le.inverse_transform([1])))
label_count = df.label.value_counts()


from numpy.ma.core import size
#plt.figure(figsize=(4,4))
#ax = sns.countplot('label',data = df)
#plt.xticks(size = 12)
#plt.xlabel('Labels')
#plt.yticks(size = 12)
#plt.ylabel('Count Of Labels')
#plt.show()
#print('-'*50)
#print('Count of 0 and 1 is {0} and {1} respectively.'.format(label_count[0],label_count[1]))
#print('-'*50)


def preprocess_text(message):
    without_punc = [char for char in message if char not in string.punctuation]
    without_punc = ''.join(without_punc)
  # Now just remove any stopwords and return the list of the cleaned text
    return [word for word in without_punc.split() if word.lower() not in stopwords.words('english')]


# apply the function to message column
#df['message'].apply(preprocess_text).head(2)


#spam_words = ' '.join(list(df[df['label'] == 1]['message']))
#spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
#plt.figure(figsize = (10, 8), facecolor = 'k')
#plt.imshow(spam_wc)
#plt.show()

X = df['message'] #text data
y = df['label'] #target label

# CountVectorizer is used to transform a given text into a vector on the
# basis of the frequency (count) of each word that occurs in the entire text

cv = CountVectorizer()
X =  cv.fit_transform(X)

# split the data into train and test sample.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifier = MultinomialNB().fit(X_train, y_train)
# Check the Classification Report, Accuracy, and Confusion Matrix for training data
pred_train = classifier.predict(X_train)

#print(classification_report(y_train, pred_train))
#print('-'*50)
#print('Accuracy : ',accuracy_score(y_train, pred_train))
#print('-'*50)
#print('Confusion Matrix:\n')
#plot_confusion_matrix(classifier, X_train, y_train,cmap=plt.cm.Blues);


pred_test = classifier.predict(X_test)
#print(classification_report(y_test, pred_test))
#print('-'*50)
#print('Accuracy : ',accuracy_score(y_test, pred_test))
#print('-'*50)
#print('Confusion Matrix:\n')
#plot_confusion_matrix(classifier, X_test, y_test,cmap=plt.cm.Blues)

def sms(text):
    text = [text]
    # creating a list of labels
    lab = ['1','2','3','4','5']
    # perform tokenization
    X = cv.transform(text).toarray()
    # predict the text
    p = classifier.predict(X)
    # convert the words in string with the help of list
    s = [str(i) for i in p]
    a = int("".join(s))
    # show out the final result
    res = str("the model predict that the feedback is  "+ lab[a]) +"  stars"
    return res

if __name__ == '__main__':
    sms("")