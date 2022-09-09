from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd

def forest(RATIO, ESTIMATORS, LEAF_NODES):
    data = pd.read_csv('wine.csv')
    Y = data['Class'] 
    X = data.drop(['Class'], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=RATIO/100)

    random_forest = RandomForestClassifier(n_estimators=ESTIMATORS, max_leaf_nodes=LEAF_NODES, n_jobs=-1, random_state=69)
    random_forest.fit(X_train, Y_train)

    print('----- STARTING PREDICTIONS -----')
    print('Class 1 bad wine\nClass 2 good wine\nClass 3 excelent wine')

    Y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    alcohol = float(input('Enter the value of the alcohol: '))
    malic_acid = float(input('Enter the value of the malic acid: '))
    ash = float(input('Enter the value of the ash: '))
    alcalinity_ash = float(input('Enter the value of the alcalinity of ash: '))
    magnesium = float(input('Enter the value of the magnesium: '))
    total_phenols = float(input('Enter the total phenols: '))
    flavanoids = float(input('Enter the value of the flavanoids: '))
    nonfavanoid_phenols = float(input('Enter the value of the nonfavanoid phenols: '))
    proanthocyanins = float(input('Enter the value of the proanthocyanins: '))
    color_intensity = float(input('Enter the value of the color intensity: '))
    hue = float(input('Enter the value of the hue: '))
    diluted_wines = float(input('Enter the value of the OD280/OD315 diluted wines: '))
    proline = float(input('Enter the value of the proline: '))

    user_data = [{'Alcohol': alcohol, 'Malic acid': malic_acid, 'Ash': ash, 'Alcalinity of ash': alcalinity_ash, 'Magnesium': magnesium, 'Total phenols': total_phenols, 'Flavanoids': flavanoids, 'Nonflavanoid phenols': nonfavanoid_phenols, 'Proanthocyanins': proanthocyanins, 'Color intensity': color_intensity, 'Hue': hue, 'OD280/OD315 of diluted wines': diluted_wines, 'Proline': proline}]
    predict_data = pd.DataFrame(user_data)

    probability = random_forest.predict_proba(predict_data)
    prediction = random_forest.predict(predict_data)

    print('The probability for the data proportioned is as follows: {first_prob} that it will be a bad wine, {second_prob} that it will be a good wine, and {third_prob} that it will be an excelent wine'.format(first_prob=probability[0][0], second_prob=probability[0][1], third_prob=probability[0][2]))
    
    type_wine = ''
    if(prediction[0] == '1'):
        type_wine = 'bad wine'
    elif(prediction[0] == '2'):
        type_wine = 'good wine'
    elif(prediction[0] == '3'):
        type_wine = 'excelent wine'
    
    print('That is why we can say your wine will be a {type_wine}'.format(type_wine=type_wine))
    
    print('Trust us, our model has a {accuracy} of accuracy ;)'.format(accuracy=accuracy))

    print('Here is one of the trees of the forest that helped predict your result:')

    random_index = rnd.randint(0, ESTIMATORS - 1)
    plt.figure(figsize=(12, 12))
    tree.plot_tree(random_forest.estimators_[random_index], feature_names=X.columns, fontsize=10)
    plt.show()

if __name__ == '__main__':
    RATIO = float(input('Enter a percentage of training sample: '))
    ESTIMATORS = int(input('Enter the number of trees your forest will have: '))
    LEAF_NODES = int(input('Enter the maximum number of leaf nodes each tree will have: '))
    
    forest(RATIO, ESTIMATORS, LEAF_NODES)