import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

job=[]
# Step 1 Import text file per row

for index in range(1, 14):
    name = "security engineer{index}.txt".format(index=index)
    f = open(name,'r')
    lines = f.readlines()
    description = '\t'.join([line.strip() for line in lines])
    job.append(description)

jobs = pd.DataFrame(job) 
#print(job)

# Step 2 convert job description rows to feature space
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(job).toarray()
print(X.shape)
print(X)

y = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0])
#print(vectorizer.get_feature_names()[:25])

# Step 3 Naive Bayes Classifier
x_train,x_test, y_train,y_test = train_test_split(X, y, test_size=0.2 )
clf = MultinomialNB()
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
print(y_test)
print(clf.predict(x_test))


