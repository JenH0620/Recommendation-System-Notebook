# using LR to catalogize the handwritting numbers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits 
#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# upload data
digits = load_digits()
data = digits.data

# split data，25% is used to test，the rest is used to train
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

# using Z-Score regularization
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

# creating CART classifier
model = DecisionTreeClassifier()
model.fit(train_ss_x, train_y)
print(model)
predict_y = model.predict(test_ss_x)
print('SVM accuracy rate: %0.4lf' % accuracy_score(test_y,predict_y))