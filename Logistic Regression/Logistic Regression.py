'''
Data:20181211
Author:NLPxiaoxu
Algorithm:Logistic Regression

'''
import numpy as np

def LoadData(filename):
    data_x = []
    label = []
    f = open(filename)
    for line in f:
        linearr = line.strip().split()
        data_x.append([1.0, float(linearr[0]), float(linearr[1])])
        label.append(int(linearr[2]))
    return data_x, label


def sigmoid(yi):#激活函数
    return 1.0/(1+np.exp(-yi))

def graddescent(learning_rate, epochs, data_x, label):
    x = np.mat(data_x) #列表转换为矩阵
    y = np.mat(label).transpose()  #列表转换为矩阵，并转置

    alpha = learning_rate #学习率
    MaxEpochs = epochs #迭代次数

    m, n= np.shape(x)
    w = np.ones((n, 1)) #学习参数

    for i in range(MaxEpochs):
        diff = sigmoid(np.dot(x, w)) - y
        cost = 1/(2*m)*np.sum(np.square(diff))#损失函数
        print(cost)
        theta = (1/m)*np.dot(np.transpose(x), diff)#权重求导
        w = w - alpha*theta#迭代更新权重

    return w


def main():
    dataMat, labelMat = LoadData('LR_data.txt')
    weights = graddescent(0.001, 20000, dataMat, labelMat)
    print(weights)

if __name__ == '__main__':
    main()
