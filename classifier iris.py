from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()

# Describing Datasets
# print(iris.DESCR)

# features and labels
features = iris.data
labels = iris.target
# print(features[0], labels[0])

# Training the classifier

clf = KNeighborsClassifier()
clf.fit(features,labels)


#prediction

prd = clf.predict([[1,2,3,4]])
print("As per input prediction is :-  ", prd)
