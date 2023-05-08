from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

def adjust_data_scale():
    # 导入数据
    filename = 'pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    # 将数据分为输入数据和输出结果
    array = data.values
    X = array[:,0:8]
    Y = array[:, 8]
    transformer = MinMaxScaler(feature_range=(0,1))
    # 数据转换
    newX = transformer.fit_transform(X)
    # 设定数据的打印格式
    set_printoptions(precision=3)
    print(newX)


def standardize_data():
    # 导入数据
    filename = 'pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    # 将数据分为输入数据和输出结果
    array = data.values
    X = array[:, 0:8]
    Y = array[:, 8]
    transformer = StandardScaler()
    # 数据转换
    newX = transformer.fit_transform(X)
    # 设定数据的打印格式
    set_printoptions(precision=3)
    print(newX)

def normalize_data():
    # 导入数据
    filename = 'pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    # 将数据分为输入数据和输出结果
    array = data.values
    X = array[:, 0:8]
    Y = array[:, 8]
    transformer = Normalizer()
    # 数据转换
    newX = transformer.fit_transform(X)
    # 设定数据的打印格式
    set_printoptions(precision=3)
    print(newX)

def binarize_data():
    # 导入数据
    filename = 'pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    # 将数据分为输入数据和输出结果
    array = data.values
    X = array[:, 0:8]
    Y = array[:, 8]
    transformer = Binarizer(threshold=0.5)
    # 数据转换
    newX = transformer.fit_transform(X)
    # 设定数据的打印格式
    set_printoptions(precision=3)
    print(newX)

if __name__ == '__main__':
    binarize_data()