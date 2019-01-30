'''
Data:20181224
Author:Jason
Algorithm:SVM

'''
import numpy as np

class SVM(object):
    def __init__(self, C, Maxepoch, kenel='liner',epsilon = 0.001):
        '''

        :param C:
        :param Maxepoch: 迭代次数
        :param kenel: 核函数
        :param epsilon: 精确度
        '''
        self.C = C
        self.kenel = kenel
        self.Maxepoch = Maxepoch
        self.epsilon = epsilon

    def _int_parameters(self, features, labels):
        '''
        初始化参数
        :param features:
        :param labels:
        :return:
        '''
        self.X = features
        self.Y = labels

        self.m, self.n = np.shape(self.X)
        self.alpha = np.zeros(self.m)
        self.b = 0
        self.E = [self._E(i) for i in range(self.m)]


    def _k(self, xi, xj):
        '''
        多项式核函数：(eta*（xi,xj）+c)^n
        :param eta:
        :param c:
        :param n:
        :param xi:
        :param xj:
        :return:
        '''
        if self.kenel == 'liner':
            return (np.dot(xi, xj))
        elif self.kenel == 'poly':
            return (np.dot(xi, xj))**2
        #elif self.kenel == 'rbf':


    def satisfy_KKT(self, i):
        a_i, x_i, y_i = self.alpha[i], self.X[i], self.Y[i]
        yg_x = y_i * self.g(x_i)
        if a_i == 0:
            return yg_x > 1 + self.epsilon or yg_x == 1 + self.epsilon
        elif a_i == self.C:
            return yg_x < 1 - self.epsilon or yg_x == 1 - self.epsilon
        else:
            return abs(yg_x - 1) < self.epsilon

    def select(self):
        l = [i for i in range(self.m)]
        layer1 = [i for i in range(self.m) if self.alpha[i] > 0 and self.alpha[i] < self.C]
        layer2 = [i for i in range(self.m) if i not in layer1]
        layer = layer1 + layer2
        for i in layer:
            if self.satisfy_KKT(i):
                continue
            E_i = self.E[i]
            max = (0, 0)
            for j in l:
                if i == j:
                    continue
                E_j = self.E[j]
                gap = abs(E_i - E_j)
                if gap > max[0]:
                    max = (gap, j)

            return i, max[1]

    def g(self, xi):
        '''
        :param xi:
        :return:
        '''
        k = [self._k(self.X[j], xi) for j in range(self.m)]
        k = np.array(k).T
        wx = np.dot(self.alpha, np.multiply(self.Y, k))
        g = wx + self.b
        return g

    def _E(self,i):
        x_i, y_i = self.X[i], self.Y[i]
        return self.g(x_i) - y_i


    def clip(self, x, L, H):
        if x > H:
            return H
        elif x < L:
            return L
        else:
            return x


    def train(self,features,labels):
        '''
        :param features: 训练集
        :param labels: 训练集标签
        :return: alpha,b
        '''
        self._int_parameters(features, labels)
        epoch = 0
        while epoch < self.Maxepoch:
            for i in range(self.m):
                i, j = self.select()
                #print(i, j)
                a_i, x_i, y_i = self.alpha[i], self.X[i], self.Y[i]
                a_j, x_j, y_j = self.alpha[j], self.X[j], self.Y[j]
                E_i, E_j = self.E[i], self.E[j]
                kii, kjj, kij = self._k(x_i, x_i), self._k(x_j, x_j), self._k(x_i, x_j)
                n = kii + kjj - 2 * kij
                a_i_old, a_j_old = a_i, a_j
                a_j_newun = a_j_old + y_j * (E_i - E_j) / n

                if y_i != y_j:
                    L = max(0, a_j_old - a_i_old)
                    H = min(self.C, self.C + a_j_old - a_i_old)
                else:
                    L = max(0, a_j_old + a_i_old - self.C)
                    H = min(self.C, a_j_old + a_i_old)

                a_j_new = self.clip(a_j_newun, L, H)
                a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)

                self.alpha[i], self.alpha[j] = a_i_new, a_j_new
                #print(self.alpha[i], self.alpha[j])

                b_i_new = - E_i - y_i * kii * (a_i_new - a_i_old) - y_j * kij * (a_j_new - a_j_old) + self.b
                b_j_new = - E_j - y_i * kij * (a_i_new - a_i_old) - y_j * kjj * (a_j_new - a_j_old) + self.b

                if 0 < a_i_new < self.C:
                    b_new = b_i_new
                elif 0 < a_j_new < self.C:
                    b_new = b_j_new
                else:
                    b_new = (b_i_new + b_j_new) / 2.0

                self.b = b_new
                self.E[i], self.E[j] = self._E(i), self._E(j)
            epoch += 1


    def predict(self,features):#对测试集进行预测
        '''
        :param features: 测试集
        :return:
        '''
        result = []
        m,n = np.shape(features)
        for i in range(m):
            x_i = features[i]
            y_i = self.g(x_i)
            if y_i > 0:
                result.append(1.0)
            elif y_i < 0:
                result.append(-1.0)
        return result

    def accracy(self, features, labels):
        '''
        :param features: #对测试集验证准确性
        :param labels:
        :return:
        '''
        result = self.predict(features)
        count = 0
        for i in range(len(result)):
            if int(result[i]) == int(labels[i]):
                count += 1

        return count/len(result)

'''
模型结束
'''



def LoadData(filename):
    data_x = []
    label = []
    f = open(filename)
    for line in f:
        linearr = line.strip().split()
        data_x.append([float(linearr[0]), float(linearr[1])])
        label.append(float(linearr[2]))
    return data_x, label

def Normalization(features):
    #归一化
    ma, mi = features.max(), features.min()
    return (features - mi) / (ma-mi)


data_x, labels = LoadData('SVM_data.txt')
data_x = np.array(data_x)
data_x = Normalization(data_x)
labels = np.array(labels)
svm = SVM(0.9, 5, kenel='poly', epsilon=0.001)
svm.train(data_x, labels)#训练模型
result = svm.predict(data_x)
accracy = svm.accracy(data_x, labels)
print(accracy)


'''
from sklearn import svm
data_x, labels = LoadData('SVM_data.txt')
data_x = np.array(data_x)
labels = np.array(labels)
clf = svm.SVC(kernel='rbf', C=100)
clf.fit(data_x, labels)
result = clf.predict(data_x)
score = clf.score(data_x,labels)
print(score)

'''


