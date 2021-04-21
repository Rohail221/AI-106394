# AI-106394 Fall 2021:Course Repository

### PROJECT MEMBERS ###
StdID | Name
------------ | -------------
64181 | Hafiz Ali Hammad Ansari 
62886 | Hassan Dawood
63454 | Rohail Shah


## Rohail Shah

I worked on loading the data and preparing it for training. After this I split the data set into training and testing sets and applied Linear Regression on the training data. After this I tested the Linear Regression Model against the testing set.

I learnt how to use numpy,pandas and scikit learn for training Machine Learning Models. I also learnt how to read and write data to csv files.


## Code

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

	df=pd.read_csv("C:/Users/Rohail/Desktop/Python/Theory/train.csv")


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
	testingSet = pd.read_csv("C:/Users/Rohail/Desktop/Python/Theory/test.csv")
	testingSet = testingSet / 255
	svmFinalpred=svmClf.predict(testingSet)

	# Generate csv file for kaggle
	finalPred=pd.DataFrame(svmFinalpred,columns=["Label"])
	finalPred['ImageId']=finalPred.index+1
	finalPred = finalPred.reindex(['ImageId','Label'], axis=1)
	finalPred.to_csv('RohailDigit3.csv',index=False)
  
![Kaggle Accuracy](https://github.com/Rohail221/AI-106394/blob/master/Assign01/RohailKAGGLE.png)
