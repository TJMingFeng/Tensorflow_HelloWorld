# 导入数据集
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# one-hot 一种有效的编码形式.
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# mnist数据集: .shape是数据集大小
# 每一个图片是28*28,在此扩展为784维的一维向量.
# 28*28矩阵中,每一位是一个数字,数值越大,代表颜色越深,即灰度值越高.
# 特征是一个55000*784的Tensor.
# 对应的label是十维的向量. Ex:[0,0,0,0,0,1,0,0,0,0]代表5.
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# 模型对一张图片进行预测时,Softmax Regression模型会对每一种类别估算一个概率,所有概率和是1,概率最大的数字作为模型的输出结果.
# Softmax Regression模型工作原理：将可以判定为某类的特征相加,然后将这些特征转化为判定是这一类的概率.

# 创建一个Session,之后的运算默认在该Session里,不同Session之间的运算和数据都是相互独立的.
sess = tf.InteractiveSession()
# 数据输入的地方.第一个参数是数据类型,第二个参数代表tensor的shape.None代表不限条数的输入.
x = tf.placeholder(tf.float32,[None,784])

# Weights和biases对象,初始化全为0.
# Tensor一旦使用掉就会消失,Variable在训练模型迭代中是持久的.
# 对于该模型初值不太重要.
# W的shape是[784,10],784是维度,后面的10代表10类.
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# y = softmax(Wx+b)
# softmax是tf.nn下的一个函数,tf.nn包含了大量的神经网络组件,tf.matmul是Tensorflow中的矩阵乘法函数.
y = tf.nn.softmax(tf.matmul(x,W)+b)

# loss function描述模型对分类问题的分类精度.
# 使用cross-entropy作为loss-function.
# cross-entropy是信息熵.
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

# SGD(Stochastic Gredient Descent)梯度下降优化算法,最小化cross_entropy,反向传播优化参数.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 初始化所有的Tensorflow变量.
tf.global_variables_initializer().run()

# 每次随机从训练集中抽取100条样本,构成一个mini-batch(作业),使用这些数据进行梯度下降。
# 相比于每次使用全部数据梯度下降,计算量更小,容易跳出局部最优,收敛速度也比较快.
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

# 验证正确率,tf.equal方法则用来判断预测的标签是否就是正确的类别.
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

# tf.cast将bool转化为float32,再求均值.
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))