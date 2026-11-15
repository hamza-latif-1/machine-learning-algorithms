class MyLinearRegression:

    def __init__(self):

        self.m = None
        self.b = None

    def fit(self, X_train, y_train):

        num = 0
        den = 0

        X_mean = X_train.mean()
        y_mean = y_train.mean()

        for i in range(X_train.shape[0]):            

            num += (X_train[i] - X_mean) * (y_train[i] - y_mean)
            den += (X_train[i] - X_mean) ** 2

        self.m = num / den
        self.b = y_mean - (self.m * X_mean)

    def predict(self, X_test):

        return (self.m * X_test) + self.b


import numpy as np

X_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10])

model = MyLinearRegression()

model.fit(X_train, y_train)

print("Slope:", model.m)
print("Intercept:", model.b)

X_test = np.array([6, 7, 8])

print("Predictions:", model.predict(X_test))