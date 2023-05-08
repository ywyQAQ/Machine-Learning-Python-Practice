from csv import reader
import numpy as np

def main():
    # 使用标准的Python类库导入CSV数据
    filename = 'pima_data.csv'
    with open(filename, 'rt') as raw_data:
        readers = reader(raw_data, delimiter=',')
        x = list(readers)
        data = np.array(x).astype('float')
        print(data.shape)
if __name__ == '__main__':
    main()