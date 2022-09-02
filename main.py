from this import d
from matplotlib.pyplot import axis
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem.snowball import  SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

#preprocessing data
fake = pd.read_csv('Project2\Fake.csv')
true  = pd.read_csv('Project2\True.csv')
fake['target'] =0
true['target']=1

#merging the datasets
data =pd.concat([fake,true],axis=0)
data =data.reset_index(drop=True)
data =data.drop(['subject','date','title'],axis=1)
print(data.columns)
data['text']= data['text'].apply(word_tokenize)
print(data.head(10))

# Stemming
porter =SnowballStemmer('english')

def stem_it(text):
    return [porter.stem(word) for word in text]

data['text'] = data['text'].apply(stem_it)
print(data.head(10))

#StopWord removal

def stop_it(t):
    dt= [word for word in t if len(word)>2]
    return dt

print(data.head(10))
data['text'] = data['text'].apply(stop_it)
print(data['text'].head(10))
data['text'] = data['text'].apply(' '.join)

#Splitting data

X_train,X_test,y_train,y_test = train_test_split(data['text'],data['target'],test_size = 0.25)
print(X_train.head())
print('\n')
print(y_train.head())

#vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
my_tfid = TfidfVectorizer(max_df=0.7)

tfid_train = my_tfid.fit_transform(X_train)
tfid_test = my_tfid.transform(X_test)

print(tfid_train)

# Logist Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model1 = LogisticRegression(max_iter=900)
model1.fit(tfid_train,y_train)
pred1 = model1.predict(tfid_test)
cr1 = accuracy_score(y_test,pred1)
print("Accuracy of Logistic Regression model: ",cr1*100)

# Passive Aggressive Classifier

from sklearn.linear_model import PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfid_train,y_train)

PassiveAggressiveClassifier(max_iter=50)
y_pred = model.predict(tfid_test)
accsore = accuracy_score(y_test,y_pred)
print('The accuracy of the passive agrressicve prediction is: ',accsore*100)