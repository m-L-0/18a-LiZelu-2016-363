import tensorflow as tf
import numpy as np

#1.边edge
#数据依赖
a = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]],name = 'a')
b = tf.constant([[1.0,2.0],[3.0,4.0],[5.0,6.0]],name  = 'b')
c = tf.matmul(a,b)
d = tf.add(c,5)

with tf.Session() as sess:
    print(sess.run(d))


#控制依赖
# g = tf.get_default_graph()
with tf.control_dependencies([c]):
    d = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]],name='d')
    e = tf.constant([[1.0,2.0],[3.0,4.0],[5.0,6.0]],name='e')
    f = tf.matmul(d,e)
with tf.Session() as sess:
    print(sess.run(f))


#张量的阶、形状、数据类型
#tensor的阶
t = [[1,2,3],[4,5,6],[7,8,9]]

a1 = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
a2 = tf.rank(a1)#获取张量的阶
with tf.Session() as sess:
    print(sess.run(a2))

#张量的形状shape
a2 = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]],shape=[2,3])

#方法一
a2.shape
#方法二
a2.get_shape()
#方法三
tf.shape(a2)

#将tensorshape对象转换为python的list对象
a2.get_shape().as_list()
#使用list构建一个tensorshape对象
ts = tf.TensorShape([2,3])

#test1
with tf.Session() as sess:
    q1 = tf.constant([])
    q2 = tf.constant(12)
    q3 = tf.constant([[1],[2],[3]])
    q4 = tf.constant([[1,2,3],[1,2,3]])
    q5 = tf.constant([[[1]],[[2]],[[3]]])
    q6 = tf.constant([[[1,2],[1,2]]])

    print(sess.run(tf.rank(q1)))
    print(sess.run(tf.rank(q2)))
    print(sess.run(tf.rank(q3)))
    print(sess.run(tf.rank(q4)))
    print(sess.run(tf.rank(q5)))
    print(sess.run(tf.rank(q6)))
    print(sess.run(tf.shape(q1)))
    print(sess.run(tf.shape(q2)))
    print(sess.run(tf.shape(q3)))
    print(sess.run(tf.shape(q4)))
    print(sess.run(tf.shape(q5)))
    print(sess.run(tf.shape(q6)))


#设置和获取tensor的数据类型
#设置tensor的数据类型
#方法一：tensorflow会自动推断出类型为tf.float32
a = tf.constant([[1,2,3],[4,5,6]])

#方法二：手动设置
a = tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)

#方法三：（不推荐）设置np类型，未来可能不兼容
a = tf.constant([[1,2,3],[4,5,6]],dtype=np.float32)


#获取tensor的数据类型
a = tf.constant([[1,2,3],[4,5,6]],name='a')
print(a.dtype)

b = tf.constant(2+3j)
print(b.dtype)

c = tf.constant([True,False],tf.bool)
print(c.dtype)


#数据类型转换
a = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]],name='a')
b = tf.cast(a,tf.int16)
print(b.dtype)

#string转换成number类型
a = tf.constant([['1.0','2.0','3.0'],['4.0','5.0','6.0']],name='a')
num = tf.string_to_number(a)
print(num.dtype)



#test2
w1 = tf.constant([1,2,3],dtype=tf.float64)
w2 = tf.constant([1,2,3],dtype=tf.complex64)
# w3 = tf.constant([1,2,3],dtype=tf.string)   错误❌
# w4 = tf.constant([1,'2','3'])    错误❌
# w5 = tf.constant([1,[2,3]])    错误❌



#节点
#变量：用于存储张量，用list，tensor来初始化
#使用纯量0进行初始化一个变量
var = tf.Variable(0)


#张量元素运算
#tf.add() 两个张量相加，等价于+
tf.add(1,2)
tf.add([1,2],[3,4])
tf.constant([1,2]) + tf.constant([3,4])

#tf.subtract() 两个张量相减，等价于-
tf.subtract(2,1)
tf.subtract([1,2],[3,4])
tf.constant([1,2]) + tf.constant([3,4])

#tf.multiply() 两个张量相乘，等价于*
tf.multiply(1,2)
tf.multiply([1,2],[3,4])
tf.constant([1,2]) * tf.constant([3,4])

#tf.scalar_mul() 一个纯量分别与张量中每一个元素相乘，等价于a * B
print(sess.run(tf.scalar_mul(10.,tf.constant([1.,2.]))))

#tf.divide() 两个张量对应元素相除，等价于/，不接受list或常量
tf.divide(1,2)
tf.divide(tf.constant([1,2]),tf.constant([3,4]))
tf.constant([1,2]) / tf.constant([3,4])

#tf.div() 不推荐

#tf.floordiv() shape相同的两个张量对应元素相除取整数部分，等价于//
tf.floordiv(1,2)
tf.floordiv([4,3],[2,5])
tf.constant([4,3]) // tf.constant([2,5])

#tf.mod() shape相同的两个张量进行模运算，等价于%
tf.mod(1,2)
tf.mod([4,3],[2,5])
tf.constant([4,3]) % tf.constant([2,5])


#张量常用运算
#tf.matmul() 通常用来作矩阵乘法
#tf.transpose() 转置张量
a = tf.constant([[1.,2.,3.],[4.,5.,6.0]])
tf.matmul(a,tf.transpose(a))

#张量切片与索引
#张量变形
#将张量变为指定shape的新张量
new_t = tf.reshape(t,[3,3])
new_t = tf.reshape(new_t,[-1])

#张量拼接
#沿着某个维度对两个或多个张量进行拼接
t1 = [[1,2,3],[4,5,6]]
t2 = [[7,8,9],[10,11,12]]
tf.concat([t1,t2],0)

#张量切割
#对输入的张量进行切片
tf.slice(input,[1,0,0],[1,1,3])
tf.slice(input,[1,0,0],[1,2,3])
tf.slice(input,[1,0,0],[2,1,3])

#将张量分裂成子张量
# split0,split1,split2 = tf.split(value=[5,30],[4,15,11],1)



#test3
#1   28*28
#2
new_i = tf.reshape(img,[28,28])

#3
tf.slice(new_i,[4,26],[1,5])




#homework
#1
split1 = tf.split(img,1,1)

#2
tf.slice(img,[1,0,0,0],[1,28,28,3])

#3
#索引只能从图片的某个维度将此维度的数据取出，切片可以在任何维度取出任何想要的数据

#4
tf.slice(img,[0,0,0,0],[1,28,28,3])
tf.slice(img,[2,0,0,0],[1,28,28,3])
tf.slice(img,[4,0,0,0],[1,28,28,3])
tf.slice(img,[6,0,0,0],[1,28,28,3])

#5
tf.slice(img,[5,7,7,0],[2,14,14,3])

#6
new_img = tf.reshape(img,[30,28,28,1])

#7
#阶为4
#TensorShape([Dimension(10),Dimension(28),Dimension(28),Dimension(3)])

