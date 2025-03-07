
# Importing necessary libraries

import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

df=pd.read_csv('datasets_6660_9643_Restaurant_Reviews.tsv', delimiter='\t')

df.head()

df.isnull().sum()

df['Liked'].value_counts()

# Data Cleaning

def clean_data(text):
  text=text.lower()
  pattern= r'[^\w+,!?/:;\"\'\s]'
  re.sub(pattern,'',text)
  text=text.strip()
  dig= r'[\d]'
  re.sub(dig,'',text)
  re.sub(r'\s+',' ',text)
  tokens= word_tokenize(text)
  stops=set(stopwords.words('english'))
  text=[word for word in tokens if word not in stops]
  text=' '.join(text)
  token=RegexpTokenizer(r'\w+')
  text = token.tokenize(text)
  lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
  text=[lemmatizer.lemmatize(word) for word in tokens]
  text=' '.join(text)
  return text

# Sample text

text= """
Thor: Love and Thunder is a 2022 American superhero film based on Marvel Comics featuring the character Thor, 
produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures. 
It is the sequel to Thor: Ragnarok (2017) and the 29th film in the Marvel Cinematic Universe (MCU). 
The film is directed by Taika Waititi, who co-wrote the script with Jennifer Kaytin Robinson, 
and stars Chris Hemsworth as Thor alongside Christian Bale, Tessa Thompson, Jaimie Alexander, 
Waititi, Russell Crowe, and Natalie Portman. In the film, Thor attempts to find inner peace, 
but must return to action and recruit Valkyrie (Thompson), Korg (Waititi), and Jane Foster 
(Portman)—who is now the Mighty Thor—to stop Gorr the God Butcher (Bale) from eliminating all gods.
"""

val2=clean_data(text)
val2

X=df['Review']
y=df['Liked']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)

X_train=X_train.reset_index(drop=True)
X_test=X_test.reset_index(drop=True)

X_train

clean_X_train=X_train.apply(clean_data)

clean_X_train

clean_X_test=X_test.apply(clean_data)

clean_X_test

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt

vectorizer= TfidfVectorizer()
X_train_clean=vectorizer.fit_transform(clean_X_train)
X_test_clean=vectorizer.transform(clean_X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

model=MultinomialNB()

alpha_values={'alpha': [10**-2, 10**-1, 1, 10**1, 10**2]}
grid_1=GridSearchCV(model,param_grid=alpha_values,cv=5, scoring='accuracy',return_train_score=True)
grid_1.fit(X_train_clean,y_train)
print(grid_1.best_params_)

alpha=[10**-2, 10**-1, 1, 10**1, 10**2]
train_acc=grid_1.cv_results_['mean_train_score']
val_acc=grid_1.cv_results_['mean_test_score']

plt.plot(alpha,train_acc, label='Training Score', color='g')
plt.plot(alpha,val_acc, label='Validation Score', color='r')

plt.title('Validation curve with Multinomial Naive Bayes model')
plt.xlabel('alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model1=MultinomialNB(alpha=1)
model1.fit(X_train_clean,y_train)

y_pred=model1.predict(X_test_clean)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",cm)

from sklearn import linear_model
model2=linear_model.LogisticRegression()

C_values={'C':[10**-2, 10**-1, 1, 10**1, 10**2]}
grid_2=GridSearchCV(model2,param_grid=C_values,cv=5, scoring='accuracy',return_train_score=True)
grid_2.fit(X_train_clean,y_train)
print(grid_2.best_params_)

C=[10**-2, 10**-1, 1, 10**1, 10**2]
train_acc=grid_2.cv_results_['mean_train_score']
val_acc=grid_2.cv_results_['mean_test_score']

plt.plot(alpha,train_acc, label='Training Score', color='g')
plt.plot(alpha,val_acc, label='Validation Score', color='r')

plt.title('Validation curve with Logistic Regression model')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model2=linear_model.LogisticRegression(C=10)
model2.fit(X_train_clean,y_train)

y_pred=model2.predict(X_test_clean)
print(accuracy_score(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",cm)

text=['the food was delicious']
text_vec=vectorizer.transform(text)
print(model2.predict(text_vec))

text2=['I did not like the ambience']
text_vec=vectorizer.transform(text2)
print(model2.predict(text_vec))


from sklearn.pipeline import Pipeline
import pickle

pipeline=Pipeline([('vector',vectorizer),('model',model2)])

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)



# df['cleaned']=df['Review'].apply(clean_data)
# words= ''.join(word for word in df['cleaned'][0:len(df)])
# wordcloud=WordCloud(width=500,height=300,random_state=42,max_font_size=110).generate(words)
# plt.figure(figsize=(15,8))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.show()
# plt.axis('off')
# plt.title('Words frequently observed in the reviews')