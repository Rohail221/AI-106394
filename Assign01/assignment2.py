import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import CategoricalNB

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC


# Read the training file with pandas 

df=pd.read_csv("train.csv")


# Remove label column using drop
X = df.drop(["label"], axis=1) 
y = df["label"]
X = X / 255


# Split data into random training and testing subsets
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.15, random_state=76)

# Training data by using linear-regression provided by scikit learn
lRegr = LinearRegression()
lRegr.fit(xTrain, yTrain)

#testing our data by giving it a score
yTestPredReg = lRegr.predict(xTest)
metrics.r2_score(yTest, yTestPredReg)


# Using gaussian Naive Bayes 
gauss = GaussianNB()
yPredGNB = gauss.fit(xTrain, yTrain).predict(xTest)
accuracy_gnb = metrics.accuracy_score(yTest, yPredGNB)
print("Accuracy by Gaussian: ",accuracy_gnb)

# Using bernoulli Naive Bayes 
bernoulli = BernoulliNB()
yPredBNB = bernoulli.fit(xTrain, yTrain).predict(xTest)
accuracy_bnb = metrics.accuracy_score(yTest, yPredBNB)
print(" Accuracy by Bernoulli: ",accuracy_bnb)

# MultiNomial Naive Bayes 
multinomial = MultinomialNB()
yPredMNB = multinomial.fit(xTrain, yTrain).predict(xTest)
accuracy_mnb = metrics.accuracy_score(yTest, yPredMNB)
print(" Accuracy by MultinomialNB: ",accuracy_mnb)


svmClf = SVC(kernel="rbf", random_state=42, verbose=3,C=9)
svmClf.fit(xTrain, yTrain)

# test the trained model
yTestPredSvm = svmClf.predict(xTest)
metrics.accuracy_score(yTest, yTestPredSvm)

# Reading File test.csv 
testingSet = pd.read_csv("train.csv")
testingSet = testingSet / 255
svmFinalpred=svmClf.predict(testingSet)

# Generate csv file for kaggle
finalPred=pd.DataFrame(svmFinalpred,columns=["Label"])
finalPred['ImageId']=finalPred.index+1
finalPred = finalPred.reindex(['ImageId','Label'], axis=1)
finalPred.to_csv('HafizAliHammad.csv',index=False)