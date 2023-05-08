from pandas import read_csv

def main():
    # 使用Pandas来导入CSV数据
    filename = 'pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    print(data)


if __name__ == '__main__':
    main()