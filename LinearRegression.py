# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np

# 构造数据集
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### Tensorflow框架开始 ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 构造一个优化器
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### Tensorflow框架结束 ###

sess = tf.Session()
sess.run(init)  # 注意别遗漏

for step in range(200):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
        
        