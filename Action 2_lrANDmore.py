import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn import svm #SVM
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB #高斯朴素贝叶斯 GaussianNB/MultinomialNB/BernoulliNB
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.ensemble import  AdaBoostClassifier #AdaBoost
from xgboost import XGBClassifier #XGBoost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 数据加载
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
# 数据探索
'''print(train_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O']))
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())'''
# 数据清洗
# 使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
print(train_data['Embarked'].value_counts())

# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']


test_features = test_data[features]

dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)
test_features = dvec.fit_transform(test_features.to_dict(orient='record'))

#Data divided into two
train_x, test_x, train_y, test_y = train_test_split(train_features.astype(np.float64),
    train_labels.astype(np.float64), train_size=0.75, test_size=0.25)

# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

# 创建LR分类器
lr = LogisticRegression(solver='liblinear', multi_class='auto') #数据集比较小，使用liblinear，数据集大使用 sag或者saga
lr.fit(train_ss_x, train_y)
predict_y=lr.predict(test_ss_x)
print('LR准确率: %0.4lf' % accuracy_score(predict_y, test_y))
# 预测
pred_labels_lr = lr.predict(test_features)

# 创建LDA分类器
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(train_ss_x,train_y)
predict_y=lda.predict(test_ss_x)
print('LDA准确率: %0.4lf' %accuracy_score(predict_y,test_y))
# 预测
pred_labels_lda = lda.predict(test_features)

# 创建贝叶斯分类器
bayes = GaussianNB()
bayes.fit(train_x,train_y)
predict_y=bayes.predict(test_x)
print('朴素贝叶斯准确率: %0.4lf' %accuracy_score(predict_y,test_y))
# 预测
pred_labels_bayes = bayes.predict(test_features)

# 创建SVM分类器
svmmodel = svm.SVC(kernel='rbf', C=1.0, gamma='auto')
svmmodel.fit(train_ss_x,train_y)
predict_y=svmmodel.predict(test_ss_x)
print('SVM准确率: %0.4lf' %accuracy_score(predict_y,test_y))
# 预测
pred_labels_svm = svmmodel.predict(test_features)

# 创建KNN分类器
knnmodel = KNeighborsClassifier()
knnmodel.fit(train_ss_x,train_y)
predict_y=knnmodel.predict(test_ss_x)
print('KNN准确率: %0.4lf' %accuracy_score(predict_y,test_y))
# 预测
pred_labels_knn = knnmodel.predict(test_features)

# 创建AdaBoost分类器
# 弱分类器
dt_stump = DecisionTreeClassifier(max_depth=5,min_samples_leaf=1)
dt_stump.fit(train_ss_x, train_y)
#dt_stump_err = 1.0-dt_stump.score(test_x, test_y)
# 设置AdaBoost迭代次数
n_estimators=500
adamodel = AdaBoostClassifier(base_estimator=dt_stump,n_estimators=n_estimators)
adamodel.fit(train_ss_x,train_y)
predict_y=adamodel.predict(test_ss_x)
print('AdaBoost准确率: %0.4lf' %accuracy_score(predict_y,test_y))
# 预测
pred_labels_ada = adamodel.predict(test_features)

# 创建XGBoost分类器
xgmodel = XGBClassifier()
xgmodel.fit(train_ss_x,train_y)
predict_y = xgmodel.predict(test_ss_x)
print('XGBoost准确率: %0.4lf' %accuracy_score(predict_y,test_y))
# 预测
pred_labels_xg = xgmodel.predict(test_features)
