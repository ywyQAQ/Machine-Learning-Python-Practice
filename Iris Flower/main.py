#   导入类库
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def main():
    filename = 'iris.data.csv'
    names = ['separ-length', 'separ-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(filename, names=names)
    # 分离数据集
    array = dataset.values
    X = array[:, 0:4 ]
    Y = array[:, 4]
    validation_size = 0.2
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    #   算法审查
    models = {}
    models['LR'] = LogisticRegression(max_iter=1000)
    models['LDA'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier()
    models['CART'] = DecisionTreeClassifier()
    models['NB'] = GaussianNB()
    models['SVM'] = SVC()
    #   评估算法
    results = []
    for key in models:
        kfold = KFold(n_splits=10, random_state=None)
        cv_results = cross_val_score(models[key], X_train, Y_train,cv=kfold, scoring='accuracy')
        results.append(cv_results)

    #   箱线图比较算法
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(models.keys())
    pyplot.show()


if __name__ == '__main__':
    main()