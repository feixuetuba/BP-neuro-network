#-*- coding:utf-8 -*-
import numpy as np
import scipy.special
'''
最简单的BP神经网络，只有 输出、隐层（1层）、输出组成
'''

class BpNetwork:
	def __init__(self, inode_count, hnode_count, onode_count, lr):
		self.wight_ih = np.random.normal(0.0, pow(inode_count,-0.5), (hnode_count, inode_count))		#输入向量与隐层之间的权值
		self.wight_ho = np.random.normal(0.0, pow(onode_count,-0.5), (onode_count, hnode_count))		#隐层与输出向量之间的权值
		self.learn_rate = lr
		self.sigmoid = lambda x: scipy.special.expit(x)
		
	#input 和target 应该是1Xn的向量
	def train(self, input_vec, target_vec):
		input_vec = np.array(input_vec, ndmin=2).T
		target_vec = np.array(target_vec, ndmin=2).T
		#前向计算输出
		hide_in = np.dot(self.wight_ih, input_vec)
		hide_out = self.sigmoid(hide_in)
		final_outputs = self.sigmoid(np.dot(self.wight_ho, hide_out))
		
		#计算误差
		vec_e = target_vec - final_outputs
		
		#反馈，调整权重
		vec_he = np.dot(self.wight_ho.T, vec_e)
		# t = sigmoid(sum(上一级节点*到当前节点的权重)) = 当前节点的输出
		#梯度下降法：new_w = old_w + lr * E * t * (1-t) . 当前节点的输出
		scale = vec_e * final_outputs * (1.0 - final_outputs)
		self.wight_ho += self.learn_rate * np.dot(scale, np.transpose(hide_out))
		scale = vec_he * hide_out * (1.0 - hide_out)
		self.wight_ih += self.learn_rate * np.dot(scale, np.transpose(input_vec))
	
	def query(self, input_vec):
		input_vec = np.array(input_vec, ndmin=2).T
		hide_in = np.dot(self.wight_ih, input_vec)
		hide_out = self.sigmoid(hide_in)
		final_outputs = self.sigmoid(np.dot(self.wight_ho, hide_out))
		return final_outputs
		
if __name__ == '__main__':
	inode_count = 784
	hnode_count = 100
	onode_count = 10
	lr = 0.3
	bp = BpNetwork(inode_count, hnode_count, onode_count, lr)
	dataset = None
	with open('mnist_train.csv', 'r') as fd:
		dataset = fd.readlines()
	for line in dataset:
		all_value = line.split(',')
		inputs = (np.asfarray(all_value[1:]) / 255.0) * 0.99 + 0.01
		labels = np.zeros(onode_count) + 0.01
		labels[int(all_value[0])] = 0.99
		bp.train(inputs, labels)
	
	with open('mnist_test.csv', 'r') as fd:
		dataset = fd.readlines()
	score = []
	for line in dataset:
		all_value = line.split(',')
		inputs = (np.asfarray(all_value[1:]) / 255.0) * 0.99 + 0.01
		outputs = bp.query(inputs)
		label = np.argmax(outputs)
		if label == int(all_value[0]):
			score.append(1)
		else:
			score.append(0)
	score = np.asfarray(score)
	print "accracy:%d/%d"%(score.sum(),score.size)

