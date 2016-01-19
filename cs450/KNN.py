from sklearn import datasets
import csv
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame, read_csv
import pandas as pd
from collections import Counter
from scipy.spatial.distance import euclidean
from sklearn import preprocessing

class Dataset:

    def __init__(self, csv, test_size, random_state):

        self.ds = pd.read_csv(csv, header=None)
        self.convertData()
        self.data_train = []
        self.data_test = []
        self.target_train = []
        self.target_test = []
        self.std_train = []
        self.std_test = []
        self.test_size = test_size
        self.random_state = random_state
        self.ds_len_column = len(self.ds.columns)
        self.data = self.ds.loc[:,: self.ds_len_column - 2]
        self.targets = self.ds[-1:]

        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
                self.data, self.targets, test_size=test_size, random_state=random_state)
        print(self.data_train)
        print(self.data)
        self.std_train = [0] * int(len(self.data_train))
        self.std_test = [0] * int(len(self.data_test))
        self.standardize_data()

    def standardize_data(self):
        for n in self.train_data:
            self.std_train[n] = self.train_data[n] - self.train_data.mean() / self.train_data.std()
        for n in self.test_data:
            self.std_test[n] = (self.test_data[n] - self.test_data.mean()) / self.test_data.std()

    def convertData(self):

        le = preprocessing.LabelEncoder()
        #ds = pd.read_csv('iris.csv', header=None)

        num_col = len(self.ds.columns)


        for i in range(0, num_col):
            has_string = False
            le.fit(self.ds[i])
            list_of_classes = list(le.classes_)
            print(list(le.classes_))
            for a_class in list_of_classes:
                if isinstance(a_class, str):
                    has_string = True
            if has_string:
                self.ds[i] = le.transform(self.ds[i])



class KNN:

    def __init__(self, k, data, targets, inputs):
        self.k = k
        self.data = data
        self.targets = targets
        self.inputs = inputs
        self.nInputs = np.shape(inputs)[0]
        self.closest = np.zeros(self.nInputs)
        print("NUM INPUTS: ", self.nInputs)

    def train(self, data_train, target_train):
        print("Training data: ", data_train, "Training targets: ", target_train)

    def predict(self, data_test):
        print("Predicting with: ", data_test)
        return_values = []
        for instance in data_test:
            return_values.append(0)

        return return_values

    def knn(self):


        for n in range(self.nInputs):

            #euclidian distance for each training vector from the current input n
            #distances = np.sum((data-inputs[n,:])**2,axis=1)

            self.distances = [0] * int(len(self.data))
            print(self.distances)
            for j in range(len(self.data)):
                self.distances[j] = euclidean(self.data[j], self.inputs[n])

            #sorts an array but keeps indexes the same
            indices = np.argsort(self.distances,axis=0)
            print("indices: ", indices)

            
            self.posibilities = []
            self.closest_classes = []
            self.counter_classes = []
            self.most_common_class = []
            for b in range(0,self.k):
                self.closest_classes.insert(b, self.targets[indices[b]])

            self.posibilities= np.unique(self.closest_classes)

            #classes = np.unique(dataClass[indices[:k]])


            # if all of the nearest neighbors are the same then we have found our prediction
            if self.k == 1:
                self.closest[n] = np.unique(self.posibilities)

            else:
                while True:
                    self.counter_classes = Counter(self.closest_classes)
                    self.most_common_class = self.counter_classes.most_common(len(self.posibilities))

                    if len(self.most_common_class) is 1:
                        print("length is :", self.most_common_class)
                        break
                    del self.closest_classes[-1]
                    print(self.closest_classes)
                self.closest[n] = self.most_common_class[0][0]

            print(self.most_common_class)


        return self.closest




iris = datasets.load_iris()
iris_data = iris.data
iris_targets = iris.target

#iris_ds = Dataset(iris_data, iris_targets, .3, 42)
#iris_ds = Dataset('iris.csv', .3, 42)
#le = preprocessing.LabelEncoder()
the_data_set = Dataset('iris.csv', .3, 42)

nearestneighbor = KNN(3, the_data_set.data_train, the_data_set.target_train, the_data_set.data_test)

closest = nearestneighbor.knn()

print("closest: ", closest)
