from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


def chi_squared():
    # 导入数据
    filename = 'pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    # 将数据分为输入数据和输出结果
    array = data.values
    X = array[:, 0:8]
    Y = array[:, 8]
    # 特征选定
    test = SelectKBest(score_func=chi2,k=4)
    fit = test.fit(X, Y)
    set_printoptions(precision=3)
    print(fit.scores_)
    features = test.fit_transform(X, Y)
    print(features)

def RFE_method():
    # 导入数据
    filename = 'pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    # 将数据分为输入数据和输出结果
    array = data.values
    X = array[:, 0:8]
    Y = array[:, 8]
    # 特征选定
    model = LogisticRegression()
    rfe = RFE(model, 3)
    fit = rfe.fit(X, Y)
    print("特征个数：")
    print(fit.n_features_)
    print('被选定的特征：')
    print(fit.support_)
    print('特征排名：')
    print(fit.ranking_)


def PCA_method():
    # 导入数据
    filename = 'pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    # 将数据分为输入数据和输出结果
    array = data.values
    X = array[:, 0:8]
    Y = array[:, 8]
    # 特征选定
    pca = PCA(n_components=3)
    fit = pca.fit(X)
    print("解释方差： %s" % fit.explained_variance_ratio_)
    print(fit.components_)


def Bagged_decision_trees():
    # 导入数据
    filename = 'pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    # 将数据分为输入数据和输出结果
    array = data.values
    X = array[:, 0:8]
    Y = array[:, 8]
    # 特征选定
    model = ExtraTreesClassifier()
    fit = model.fit(X, Y)
    print(fit.feature_importances_)

if __name__ == '__main__':
    Bagged_decision_trees()
