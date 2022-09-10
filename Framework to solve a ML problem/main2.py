from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

def forest(RATIO, ESTIMATORS, LEAF_NODES, X, Y, iteration):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=RATIO/100)

    random_forest = RandomForestClassifier(n_estimators=ESTIMATORS, max_leaf_nodes=LEAF_NODES, n_jobs=-1, random_state=69)
    random_forest.fit(X_train, Y_train)

    print('----- STARTING PREDICTIONS {iteration} -----'.format(iteration=iteration + 1))
    print('Class 1 bad wine\nClass 2 good wine\nClass 3 excelent wine')

    Y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    plt.figure(figsize=(10, 10))
    plt.plot(Y_pred, label='Predictions')
    plt.plot(list(Y_test), label='Real values')
    plt.title('Predictions vs. Real Values {iteration}'.format(iteration=iteration + 1))
    plt.legend()
    plt.show()
    
    return accuracy

if __name__ == '__main__':
    data = pd.read_csv('wine.csv')
    Y = data['Class'] 
    X = data.drop(['Class'], axis=1)

    ratio_array =      [50, 60, 70, 80, 90, 50, 60, 60, 70, 80]
    estimators_array = [5, 10, 15, 20, 5, 5, 10, 15, 20, 15]
    leaf_nodes_array = [5, 10, 16, 18, 25, 16, 17, 20, 18, 17]

    Y_test = []
    Y_pred = []
    accuracy = []
    
    for i in range(len(leaf_nodes_array)):
        RATIO = ratio_array[i]
        ESTIMATORS = estimators_array[i]
        LEAF_NODES = leaf_nodes_array[i]
    
        acc = forest(RATIO, ESTIMATORS, LEAF_NODES, X, Y, i)
        accuracy.append(acc)

    plt.scatter(ratio_array, accuracy)
    plt.title('Train-test ratio vs accuracy')
    plt.xlabel('Ratio')
    plt.ylabel('Accuracy')
    plt.show()

    plt.scatter(estimators_array, accuracy)
    plt.title('Number of trees vs accuracy')
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy')
    plt.show()

    plt.scatter(leaf_nodes_array, accuracy)
    plt.title('Number of leaves vs. accuracy')
    plt.xlabel('Number of leaves')
    plt.ylabel('Accuracy')
    plt.show()