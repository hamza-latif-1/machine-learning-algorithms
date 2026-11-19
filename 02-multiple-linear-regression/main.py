import numpy as np

class MyLinearRegression:

    def __init__(self):

        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):

        X_train = np.insert(X_train, 0, 1, axis=1)

        beta = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

    def predict(self, X_test):

        y_pred = np.dot(X_test, self.coef_) + self.intercept_

        return y_pred


## Taking the pre-processed dataset from the library scikit-learn

from sklearn.datasets import load_diabetes

X,y = load_diabetes(return_X_y=True)

print("Input Shape ",X.shape)
print("Output Shape ",y.shape)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

print("X_train Shape ", X_train.shape)
print("X_test Shape ", X_test.shape)


## Testing the dataset with original scikit-learn linear regression algorithm

from sklearn.linear_model import LinearRegression

originalLR = LinearRegression()

originalLR.fit(X_train,y_train)

print("Weights", originalLR.coef_)
print("Offbias", originalLR.intercept_)

orig_y_pred = originalLR.predict(X_test)

from sklearn.metrics import r2_score

print("Accuracy Score with original Algorithm: ",r2_score(y_test,orig_y_pred))


## Testing the dataset with My own written linear regression algorithm

myLR = MyLinearRegression()

myLR.fit(X_train,y_train)

print("Weights ", myLR.coef_)
print("Offbias ", myLR.intercept_)

y_pred = myLR.predict(X_test)

print("Accuracy score with My LR Algorithm: ",r2_score(y_test, y_pred))

## Both the algorithms show similar results
