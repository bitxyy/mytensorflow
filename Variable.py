# -*- coding: utf-8 -*-
# 进入一个交互式 TensorFlow 会话.
import tensorflow as tf

 # 创建一个变量，初始化为标量0
state = tf.Variable(0, name="counter")
 
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

 # 初始化
init_op = tf.initialize_all_variables()

 # 启动图，运营op
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(state)

    for _ in range(3):
        sess.run(update)
        print sess.run(state)
    