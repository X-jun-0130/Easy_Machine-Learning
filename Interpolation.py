# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
'''
拉格朗日插值法
'''

def Lagrange_Parameters(data_x, data_y): #计算系数
    Parameters = []
    size = len(data_x)
    for i in range(0, size):
        temp = 1
        for j in range(0, size):
            if i!=j:
                temp *= data_x[i]-data_x[j]
            else:
                continue
        Parameters.append(data_y[i]/temp)
    return Parameters

def Lagrange_Expression(data_x, Parameters,x):#计算公式
    size = len(data_x)
    Expression = 0
    for i in range(0,size):
        temp = 1
        for j in range(0, size):
            if i != j:
                temp *= x-data_x[j]
            else:
                continue
        Expression += temp * Parameters[i]
    return Expression


'''
牛顿插值法
'''
def diff_quot(data_x,data_y): #计算差商
    size_x = len(data_x)
    size_y = len(data_y)
    if size_x >2 and size_y >2:
        return (diff_quot(data_x[:size_x-1],data_y[:size_y-1]) - diff_quot(data_x[1:size_x],data_y[1:size_y]))/float(data_x[0]-data_x[-1])
    return (data_y[0] - data_y[1])/float(data_x[0]-data_x[1])

def get_w(data_x,x):
    size_x = len(data_x)
    w = 1.0
    for i in range(size_x-1):
        w *= x-data_x[i]
    return w

def get_NewTon(data_x, data_y, x):
    size= len(data_x)
    result = data_y[0]
    for i in range(2, size+1):
        result += diff_quot(data_x[:i], data_y[:i])*get_w(data_x[:i], x)
    return result

data_x =[10, 11, 12, 14]
data_y =[100, 121, 144, 196]

n = get_NewTon(data_x, data_y, 13)

from scipy.interpolate import lagrange
t = lagrange(data_x, data_y)(13)

