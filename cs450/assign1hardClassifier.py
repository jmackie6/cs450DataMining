from sklearn import datasets
from sklearn.utils import shuffle
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

iris = datasets.load_iris()

train_data, test_data, train_target, test_target = train_test_split(iris.data,iris.target, test_size=0.3, random_state=56)


class hardcoded:

    def Train(self, train_data, train_target):
        print("\nTraining Data: \n", train_data, "\n\n", "Training targets: \n", train_target, "\n")


    def predict(self, test_data):
        print("Dataset to predict: \n", test_data, "\n")
        values = []
        for instance in test_data:
            values.append(0)

        return values

hardcoded = hardcoded()

hardcoded.Train(train_data, train_target)


answers = hardcoded.predict(test_data)
print("predictions: \n", answers, "\n")

accuaracy_score = accuracy_score(test_target, answers)
print("percentage: ", accuaracy_score, "\n")

print("----------------------------n-fold cross validation attempted------------------")

X = iris.data
y = iris.target
kf = KFold(150, n_folds=40)

for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


