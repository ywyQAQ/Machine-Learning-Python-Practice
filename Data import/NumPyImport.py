from numpy import loadtxt

def main():
    # 使用NumPy导入CSV数据
    filename = 'pima_data.csv'
    with open(filename, 'rt') as raw_data:
        data = loadtxt(raw_data, delimiter=',')
        print(data.shape)

if __name__ == "__main__":
    main()