from statistics import variance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def forest(RATIO, ESTIMATORS, LEAF_NODES, MAX_DEPTH, SAMPLES_SPLIT, IMPURITY_DECREASE, X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=RATIO/100)

    random_forest = RandomForestClassifier(n_estimators=ESTIMATORS, max_leaf_nodes=LEAF_NODES, max_depth=MAX_DEPTH, min_samples_split=SAMPLES_SPLIT, min_impurity_decrease=IMPURITY_DECREASE, n_jobs=-1, random_state=69)
    random_forest.fit(X_train, Y_train)

    print('----- STARTING PREDICTIONS -----')
    print('Class 1 bad wine\nClass 2 good wine\nClass 3 excelent wine')

    Y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    #Calculating bias and variance
    print("----- BIAS AND VARIANCE ------")
    mse, bias, var = bias_variance_decomp(random_forest, np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test), loss='mse', num_rounds=200, random_seed=1)
    #Summarize results
    print('MSE: %.3f' % mse)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f' % var)
    print('Accuracy: %.3f' % accuracy)

    return mse, bias, var, accuracy

if __name__ == '__main__':
    data = pd.read_csv('wine.csv')
    Y = data['Class'] 
    X = data.drop(['Class'], axis=1)

    ratio_array =      [50, 60, 70, 80, 90]
    estimators_array = [10, 15, 20, 12, 17]
    leaf_nodes_array = [5, 10, 16, 18, 25]
    depth_array = [10, 20, 30, 40, 40]
    samples_split_array = [4, 3, 5, 3, 5]
    impurity_decrease_array = [0.1, 0.2, 0.2, 0.1, 0.2]

    metrics = []
    
    for i in range(len(leaf_nodes_array)):
        RATIO = ratio_array[i]
        ESTIMATORS = estimators_array[i]
        LEAF_NODES = leaf_nodes_array[i]
        MAX_DEPTH = depth_array[i]
        SAMPLES_SPLIT = samples_split_array[i]
        IMPURITY_DECREASE = impurity_decrease_array[i]
    
        mse, bias, var, accuracy = forest(RATIO, ESTIMATORS, LEAF_NODES, MAX_DEPTH, SAMPLES_SPLIT, IMPURITY_DECREASE, X, Y)
        metrics.append([mse, bias, var, accuracy])

    metrics = np.array(metrics)
    x = np.arange(metrics.shape[0])
    dx = (np.arange(metrics.shape[1]) - metrics.shape[1] / 2.) / (metrics.shape[1] + 2.)
    d = 1. / (metrics.shape[1] + 2.)
    labels = ['mse', 'bias', 'var', 'accuracy']
    
    fig, ax=plt.subplots()
    for i in range(metrics.shape[1]):
        ax.bar(x + dx[i], metrics[:, i], width=d, label=labels[i])
        
    plt.legend(framealpha=1).set_draggable(True)
    plt.show()