import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV

#list containing the prediction of various models
prediction = dict()

#loading of the data  
data = pd.read_csv("spam.csv",encoding='latin-1')

#Drop the unwanted columns and rename the existing columns 
data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis = 1)
data = data.rename(columns={"v1":"label","v2":"message"})
# print data.tail()

#convert the label to its numeric value for classification
data['label_num'] = data.label.map({'ham':0,'spam':1})

#Test-Train-Split 
Xtrain,Xtest,ytrain,ytest =train_test_split(data['message'],data['label'],test_size=0.2,random_state=10)

#Text transformation : Tf-idf text frequency-inverse document frequency
cv = TfidfVectorizer(min_df=1,stop_words='english')
#transform text to a relatable sparse matrix
cv.fit(Xtrain)
Xtrain_df=cv.transform(Xtrain)
Xtest_df =cv.transform(Xtest)

#The Machine Learning model_selection
#......................................
scores = dict() 
#1. Multinomial Naive Bayes Classification
mnb = MultinomialNB()
mnb.fit(Xtrain_df,ytrain)
prediction["Naive_Bayes"] = mnb.predict(Xtest_df)
scores["Naive_Bayes"]=accuracy_score(ytest,prediction["Naive_Bayes"])


#2. Logistic Regression
lr = LogisticRegression()
lr.fit(Xtrain_df,ytrain)
prediction["LogisticRegression"]=lr.predict(Xtest_df)
scores["LogisticRegression"]=accuracy_score(ytest,prediction["LogisticRegression"])

#3. k-NN classifier with Grid-Search for tuning n_neighbors
knn = KNeighborsClassifier()
k_range = np.arange(1,30)
param_grid = dict(n_neighbors = k_range)
grid = GridSearchCV(knn,param_grid)
grid.fit(Xtrain_df,ytrain)
prediction["KNN"]=grid.best_estimator_
scores["KNN"]=grid.best_score_

#4. Ensemble classifier
rf = RandomForestClassifier()
rf.fit(Xtrain_df,ytrain)
prediction["Random_Forest"]=rf.predict(Xtest_df)
scores["Random_Forest"]=accuracy_score(ytest,prediction["Random_Forest"])

#5. Linear Support Vector Machines
svc=LinearSVC()
svc.fit(Xtrain_df,ytrain)
prediction["Linear_SVC"]=svc.predict(Xtest_df)
scores["Linear_SVC"]=accuracy_score(ytest,prediction["Linear_SVC"])



print "The accuracy of various models are as follows:"
iterator=0
max = 0
best = 0
for i in scores:
	iterator=iterator+1	
	if scores[i] > max:
		max=scores[i]
		best = i
	print iterator,i,": ",float(int(scores[i]*100000))/1000.0,"%"

best_model=best
print "The best model is : ",best_model

print "Comparing various models"

x=[]
y=[]
for i in prediction:
	x.append(i)
	y.append(scores[i]*100)

plt.plot(x,y)
plt.title("Spam Filtering using various models")
plt.xlabel("--------------Models--------------->")
plt.ylabel("---------Accuracy in %---------->")
plt.savefig("compare_models.png")

print "Printing the confusion_matrix"
conf_mat = confusion_matrix(ytest, prediction[best_model])
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

sns.heatmap(conf_mat_normalized)
bmodel = "Best model is "+best
plt.title(bmodel)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("confusion_matrix.png")
