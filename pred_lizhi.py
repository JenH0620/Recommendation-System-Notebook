# using LR to catalogize the handwritting numbers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import csv
import numpy as np

## a function that change the input matrix into digits
def Findindex(list1, b):
    r: int = -1  # -1 indicates no same string saved before
    for ii in range( 0, len( list1 ) ):
        try:
            r = list1.index( b )
        #           indices.append( idx )
        except ValueError:
            continue
    return r


def str2dig(A):
    Ar = A.copy()
    (m, n) = Ar.shape
    # n = len(array(Ar))
    for i in range( 1, n ):
        Avec = Ar[:, i]
        res = 0
        if any( c in Avec[1] for c in ('a', 'e', 'T', 'N', 'Y') ):
            res = 1
        # print( res ) #check
        if res == 1:  # when it is a string
            str_temp = [Avec[1]]  # initial
            indicator = -1
            for j in range( 1, m ):
                indicator = Findindex( str_temp, Avec[j] )
                # print( indicator )
                if indicator == -1:
                    str_temp.append( str( Avec[j] ) )
                    Ar[j, i] = len( str_temp ) - 1  # value can be changed to binary
                else:
                    Ar[j, i] = indicator
            str_temp.clear()
    return Ar

## functions that split data, use Z-Score regularization and create classifier
def lr_Pred(X,Y,Xtest,Name):
    score0 = 0
    useIDfinal = []
    for testtime1 in range( 0, 100 ):
        useID = []

        # split data，25% is used to test，the rest is used to train
        train_x, test_x, train_y, test_y = train_test_split( X, Y, test_size=0.25, random_state=33 )
        train_x = train_x.astype( np.float )
        test_x = test_x.astype( np.float )
        train_y = train_y.astype( np.float )
        test_y = test_y.astype( np.float )

        # using Z-Score regularization
        ss = preprocessing.StandardScaler()
        train_ss_x = ss.fit_transform( train_x )
        test_ss_x = ss.transform( test_x )

        # creating LR classifier
        lr = LogisticRegression()
        clf = lr.fit( train_ss_x, train_y )
        predict_y = clf.predict( test_ss_x )
        scoretemp = accuracy_score( test_y, predict_y )

        if scoretemp > score0:
            score0 = scoretemp
            test_ss_test = ss.transform( Xtest )
            predict_test = clf.predict( test_ss_test )
            for iii in range( 0, len( Name ) ):
                if predict_test[iii] == 1:
                    useID.append(Name[iii])
            useIDfinal.clear()
            useIDfinal = useID.copy()
        else:
            continue
    return score0, useIDfinal

def cart_Pred(X,Y,Xtest,Name):
    score0 = 0
    useIDfinal = []
    for testtime11 in range( 0, 100 ):
        useID = []

        # split data，25% is used to test，the rest is used to train
        train_x, test_x, train_y, test_y = train_test_split( X, Y, test_size=0.25, random_state=33 )
        train_x = train_x.astype( np.float )
        test_x = test_x.astype( np.float )
        train_y = train_y.astype( np.float )
        test_y = test_y.astype( np.float )

        # using Z-Score regularization
        ss = preprocessing.StandardScaler()
        train_ss_x = ss.fit_transform( train_x )
        test_ss_x = ss.transform( test_x )

        # creating CART classifier
        clf = DecisionTreeClassifier()
        clf.fit( train_ss_x, train_y )
        predict_y = clf.predict( test_ss_x )
        scoretemp = accuracy_score( test_y, predict_y )

        if scoretemp > score0:
            score0 = scoretemp
            test_ss_test = ss.transform( Xtest )
            predict_test = clf.predict( test_ss_test )
            for iii in range( 0, len( Name ) ):
                if predict_test[iii] == 1:
                    useID.append(Name[iii])
            useIDfinal.clear()
            useIDfinal = useID.copy()
        else:
            continue
    return score0, useIDfinal

def svm_Pred(X,Y,Xtest,Name):
    score0 = 0
    useIDfinal = []
    for testtime111 in range( 0, 100 ):
        useID = []

        # split data，25% is used to test，the rest is used to train
        train_x, test_x, train_y, test_y = train_test_split( X, Y, test_size=0.25, random_state=33 )
        train_x = train_x.astype( np.float )
        test_x = test_x.astype( np.float )
        train_y = train_y.astype( np.float )
        test_y = test_y.astype( np.float )

        # using Z-Score regularization
        ss = preprocessing.StandardScaler()
        train_ss_x = ss.fit_transform( train_x )
        test_ss_x = ss.transform( test_x )

        # creating svm classifier
        svc = SVC( kernel='linear' )
        svc.fit( train_ss_x, train_y )
        predict_y = svc.predict( test_ss_x )
        #print( 'SVM accuracy rate: %0.4lf' % accuracy_score( test_y, predict_y ) )
        scoretemp = accuracy_score( test_y, predict_y )

        if scoretemp > score0:
            score0 = scoretemp
            test_ss_test = ss.transform( Xtest )
            predict_test = svc.predict( test_ss_test )
            for iii in range( 0, len( Name ) ):
                if predict_test[iii] == 1:
                    useID.append(Name[iii])
            useIDfinal.clear()
            useIDfinal = useID.copy()
        else:
            continue
    return score0, useIDfinal

# upload data from train.csv
with open( '/Users/huangzhan/Documents/开课吧/第二章-Overview/Action 2/train.csv', 'r' ) as f:
    D = list( csv.reader( f, delimiter=',' ) )
Data = str2dig( np.array( D ) )
Y = Data[2:, 2]  # Target Variable #
Xt = np.delete( Data, [0, 2], 1 )  # Feature Matrix #t
X = Xt[2:,:]

# import the test data for prediction
with open( '/Users/huangzhan/Documents/开课吧/第二章-Overview/Action 2/test.csv', 'r' ) as f:
    Dtest = list( csv.reader( f, delimiter=',' ) )
DataTest = str2dig( np.array( Dtest ) )
Xtesttemp = np.delete( DataTest, 0, 1 )  # Feature Matrix #t
Xtest = Xtesttemp[2:,:]
Name = DataTest[2:,0]

'''Define a function to use lr/svm/cart
For multiple split, we want to find a better accuracy 
Then we use it on the test data'''

# the result of LR
print(lr_Pred(X,Y,Xtest,Name))
print(cart_Pred(X,Y,Xtest,Name))
print(svm_Pred(X,Y,Xtest,Name))


