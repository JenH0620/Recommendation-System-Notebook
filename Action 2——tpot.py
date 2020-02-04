import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np


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
X_train, X_test, y_train, y_test = train_test_split(train_features.astype(np.float64),
    train_labels.astype(np.float64), train_size=0.75, test_size=0.25)

#TPOT
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

# 决策树tpot预测
pred_labels = tpot.predict(test_features)

# 得到tpot准确率
acc_tpot = round(tpot.score(train_features, train_labels), 6)
print(u'score 准确率为 %.4lf' % acc_tpot)
