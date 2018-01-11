# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy



#参数解释
#data1，data2：表示两类实验数据,data1的label为-1，data2的label为1
#已知是一个3-1-1网络
data1=[[0.28,1.31,-6.2],
[0.07,0.58,-0.78],
[1.54,2.01,-1.63],
[-0.44,1.18,-4.32],
[-0.81,0.21,5.73],
[1.52,3.16,2.77],
[2.20,2.42,-0.19],
[0.91,1.94,6.21],
[0.65,1.93,4.38],
[-0.26,0.82,-0.96]]
data2=[[0.011,1.03,-0.21],
[1.27,1.28,0.08],
[0.13,3.12,0.16],
[-0.21,1.23,-0.11],
[-2.18,1.39,-0.19],	
[0.34,1.96,-0.16],
[-1.38,0.94,0.45],
[-0.12,0.82,0.17],
[-1.44,2.31,0.14],
[0.26,1.94,0.08]]

class NeuralNetwork:

    LEARNING_RATE = 0.1
    j_terms=[]
    def __init__(self, num_inputs, num_hidden, num_outputs, weights_in, weights_out, data1, data2):

    	'''
    	一些全局的参数解释
    	num_inputs:输入层结点数
    	num_hidden:隐藏层结点数
    	num_outputs:输出层结点数
    	weights_in：从输入到隐藏层权重数组,1行n列，第一列表示bias
    	aij：表示第i类数据的第j层的激活值
    	'''
    	self.num_inputs=num_inputs
    	self.num_hidden=num_hidden
    	self.num_outputs=num_outputs
    	self.weights_in=np.mat(weights_in).T
    	self.weights_out=np.mat(weights_out).T
    	self.data1=np.mat(data1)
    	self.data2=np.mat(data2)
    
    
    def activation(self,z):
        '''
        激活函数
        返回的是z
        '''
        z=np.mat(z)
        size=z.shape
        for i in range(size[0]):
            for j in range(size[1]):
                z[i,j]=1.716*(sy.tanh((2/3)*z[i,j]))
        return z    
    
    def input_x(self,matrix_values,matrix_weights):
        '''
        输出的结果矩阵的每一行是一个样本点的z(x)的x
        '''
        ones= np.ones((np.mat(matrix_values).shape[0],1))
        matrix_values=np.mat(np.column_stack((ones,matrix_values)))
        return matrix_values*matrix_weights        
        

    #计算h(z)
    def active_value(self,matrix_values,matrix_weights):
        '''
        输出的结果矩阵的每一行是一个样本点的h
        '''
        temp=self.input_x(matrix_values,matrix_weights)
        return self.activation(temp)
    	  	

    #计算神经网络所有激活值和激活函数的输入值
    def overall_a(self):	
        '''
        一些参数解释
        aij：表示第i类数据的第j层的激活值,是一个n行一列的矩阵
        '''    
        self.a11_x=self.input_x(self.data1,self.weights_in)
        self.a11=self.activation(self.a11_x)
        self.a12_x=self.input_x(self.a11,self.weights_out)
        self.a12=self.activation(self.a12_x)
        
        self.a21_x=self.input_x(self.data2,self.weights_in)
        self.a21=self.activation(self.a21_x)
        self.a22_x=self.input_x(self.a21,self.weights_out)
        self.a22=self.activation(self.a22_x)

    def overall_cost(self):
        '''
        参数解释
        num_samples:样本数
        active_values:输出值矩阵，就是a11等
        label:划分的样本的那个值
        '''
        sum=0
        for i in range(self.data1.shape[0]):
            sum+=(1/2)*np.power((self.a12[i,0]+1),2)
        for i in range(self.data2.shape[0]):
            sum+=(1/2)*np.power((self.a22[i,0]-1),2) 
        j=(1/(self.data1.shape[0]+self.data2.shape[0]))*sum
        self.j_terms.append(j)
        return j
    
        
    #计算网络最后一层残差   
    #以下都为选取样本中一个点进行的计算
    def cal_error_lastlayer(self,active_value,label,x):
        '''
        active_value:网络最后一层激活值
        label:分类值
        x：选中的那一组数据的h（z）中的z
        '''
        a=1.716
        b=2/3
        theta=-(label-active_value)*a*b*(-sy.tanh(b*x)**2 + 1)
        return theta
    
    #计算第l层残差（非最后一层）
    def cal_error_otherlayer(self,matrix_weights,error,x):
        '''
        matrix_weights：第L层权重矩阵
        error:第L+1层残差
        x：第l层的那一组数据的h（z）中的z
        '''
        
        a=1.716
        b=2/3
        d_h=a*b*(-sy.tanh(b*x)**2 + 1)
        temp2=error*np.mat(matrix_weights).sum(axis=0)
        #test
        theta=d_h*temp2       
        return theta
    
    #计算最终需要的第L+1层权重偏导数值
    def cal_diff_w(self,error,matrix_a):
        '''
        error:第L+1层残差
        matrix_a:第L层激活值矩阵
        '''
        return error*np.mat(matrix_a)
    
    def cal_diff_b(self,error):
        error=error
        return error
    
    def updata_wb(self):
        #随机选择一个点,单独把这个点的信息提取出来
        self.overall_a()
        self.overall_cost()
        
        num_point=np.random.randint(0,20)
        if(num_point<10):
            sample=self.data1[num_point,:].T
            a_last=self.a12[num_point,0]
            label=-1
            x_last=self.a12_x[num_point,0]
            x_other=self.a11_x[num_point,0]
        else:
            sample=self.data2[num_point-10,:].T
            a_last=self.a22[num_point-10,0]
            label=1
            x_last=self.a22_x[num_point-10,0]
            x_other=self.a21_x[num_point-10,0]
            
        sample_weights_in=self.weights_in[1:,0]
        bias_in=self.weights_in[0,0]
        sample_weights_out=self.weights_out[1:,0]
        bias_out=self.weights_out[0,0]
        
        #计算偏差
        print(self.a12.tolist()[0])
        error_lastlayer=self.cal_error_lastlayer(a_last,label,x_last)
        error_otherlayer=self.cal_error_otherlayer(sample_weights_out,error_lastlayer,x_other)
        
        #计算导数
        diff_w_last=self.cal_diff_w(error_lastlayer,a_last)
        diff_w_other=self.cal_diff_w(error_otherlayer,sample)
        
        #更新权重
        sample_weights_in=np.mat(sample_weights_in)-(1/20)*self.LEARNING_RATE*diff_w_other
        sample_weights_out=np.mat(sample_weights_out)-(1/20)*self.LEARNING_RATE*diff_w_last
        sample_bias_in=bias_in-(1/20)*self.LEARNING_RATE*self.cal_diff_b(error_otherlayer)
        sample_bias_out=bias_out-(1/20)*self.LEARNING_RATE*self.cal_diff_b(error_lastlayer)
        
        self.weights_in=np.row_stack((np.mat(sample_bias_in),sample_weights_in))
        self.weights_out=np.row_stack((np.mat(sample_bias_out),sample_weights_out))
        
        print(self.j_terms)
       
        
def main():
    np.random.seed(0)
    w_in=np.random.uniform(-1, 1, [1,4])
    w_out=np.random.uniform(-1, 1, [1,2])
    bias_in=w_in[0,0]
    bias_out=w_out[0,0]
    
    nn=NeuralNetwork(3,1,1,w_in,w_out,data1,data2)
    

    x=[]
    for i in range(1000):
        x.append(i+1)
        nn.updata_wb()

    plt.figure()
    plt.plot(x,nn.j_terms)
    '''
    plt.figure()
    w2_in=[bias_in,0.5,0.5,0.5]
    w2_out=[bias_out,-0.5]
    nn2=NeuralNetwork(3,1,1,w2_in,w2_out,data1,data2)
    y=[]
    for h in range(1000):
        y.append(h+1)
        nn2.updata_wb()
    plt.plot(y,nn2.j_terms)
    plt.show()
    '''

main()
	
