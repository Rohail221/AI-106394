import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#function to perform convolution
def convolve2D(image, filter):
  fX, fY = filter.shape # Get filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2) #new dimension of the reduced image size
  newImage = np.zeros((nn,nn)) #empty new 2D imange
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

#Read Data from CSV
train = pd.read_csv("train.csv")
X = train.drop('label',axis=1)
Y = train['label']
# print(X)

#Create Filter for convolution
filter = np.array([[1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1]])

#convert from dataframe to numpy array
X = X.to_numpy()
print(X.shape)

#new array with reduced number of features to store the small size images
sX = np.empty((0,484), int)

# img = X[6]
ss = 42000 #subset size for dry runs change to 42000 to run on whole data

#Perform convolve on all images
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))
  # print(img2D.shape)
  # print(img2D)
  nImg = convolve2D(img2D,filter)
  # print(nImg.shape)
  # print(nImg)
  nImg1D = np.reshape(nImg, (-1,484))
  # print(nImg.shape)
  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]
# print(sY)
print(sY.shape)
print(sX.shape)


# train and test model
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
clf = MultinomialNB()
clf.fit(sXTrain, yTrain)
print(clf.class_count_)

print(clf.score(sXTest, yTest))
p = clf.predict(sXTest)
print(sX.shape)

sampleSubmission=pd.read_csv('sample_submission.csv')
sampleSubmission['Label'] = p
sampleSubmission.to_csv('sub.csv', index=False)