import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def train(lr,activation1,activation2,activation3,dropout): 
    # BreastSample_normal_vs_cancer_edit dataset
    dataset = pd.read_csv("BreastSample_normal_vs_cancer_v1.csv", delimiter=",")
    dataset.head()
    dataset.describe()
    dataset['label'].value_counts()
    # split into input (X) and output (Y) variables
    X = dataset.iloc[:,0:311].values
    Y = dataset.iloc[:,312].values
    X
    Y
    # data transform
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_transform = scaler.fit_transform(X)
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import optimizers
    from sklearn.model_selection import StratifiedKFold
    # fix random seed for reproducibility
    seed = 10
    np.random.seed(seed)

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):

        model = Sequential()
        model.add(Dense(200, input_dim=311, activation=activation1))
        model.add(Dropout(dropout))
        model.add(Dense(200, activation=activation2))
        model.add(Dropout(dropout))
        model.add(Dense(200, activation=activation3))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # Fit the model
        model.fit(X_transform[train], Y[train], epochs=150, batch_size=10, verbose=0)

        # evaluate the model
        scores = model.evaluate(X_transform[test], Y[test], verbose=0)
        print("scores:",scores)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        return scores[1]*100
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# for train, test in kfold.split(X, Y):

#     model = Sequential()
#     model.add(Dense(200, input_dim=311, activation='elu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(200, activation='elu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(200, activation='elu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(1, activation='sigmoid'))

#     # Compile model
#     sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

#     # Fit the model
#     model.fit(X_transform[train], Y[train], epochs=150, batch_size=10, verbose=0)

#     # evaluate the model
#     scores = model.evaluate(X_transform[test], Y[test], verbose=0)
#     print("scores:",scores)
#     print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)
#     return scores2[1]*100
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))