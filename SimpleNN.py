# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 增加一层
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            # 权重矩阵大小：in_size * out_size
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 正态分布随机数，比全0好
            tf.histogram_summary(layer_name+'/weights', Weights)    
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 加0.1目的是使偏置项不为0
            tf.histogram_summary(layer_name+'/biases', Weights)           
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
       
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.histogram_summary(layer_name+'/outputs', Weights)    

        return outputs

x_data = np.linspace(-1,1,300)[:, np.newaxis] # 300行
# 增加一点噪声
noise = np.random.normal(0, 0.05, x_data.shape)  # 均值为0，方差为0.05的正态分布，形式同x_data
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 增加一个隐层
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# 增加输出层
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None) 

# 损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.scalar_summary('loss', loss) # 注意这是一个event事件
    
# 注：如果想在EVENTS显示，则用scalar_summary;
# 如果想在HISTOGRAMS显示，则用histogram_summary.    

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.merge_all_summaries()

# 神经网络结构
writer = tf.train.SummaryWriter("Logs/", sess.graph)
sess.run(init)

'''
# 结果可视化
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
#plt.show()
plt.show(block=False) # 使程序继续运行，不停顿
'''

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50==0:
        result = sess.run(merged,
                    feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result, i)
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        '''
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        
        plt.pause(0.1)
        '''
