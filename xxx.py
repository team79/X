import tensorflow as tf
import numpy as np

print("----------------------------------------\n" * 2)
a = tf.placeholder( tf.float32, [ 2, 3 ] )
aaa = a[0:2,0:3]
b = tf.Variable( tf.constant([[1,1.0],[1,1]]) )
c = tf.Variable( tf.constant(1.0, shape=[1,3]) )
logits = tf.add( tf.matmul( b, aaa ), c )
print("----------------------------------------\n" * 2)
m1 = logits[0]
m2 = logits[1]
m3 = logits[0:2]
m4 = tf.concat([logits,logits],1)
m5 = m4[0:2,:]
#sum = tf.Variable( tf.constant(0.0,shape = [1,3]))
# t = tf.Variable( tf.constant(0.0,shape = [1]))
print("----------------------------------------\n" * 2)
# sum =  tf.multiply(logits[0], logits[0])
t = tf.reduce_sum(tf.multiply(logits[0], logits[0]))
print("----------------------------------------\n" * 2)
for i in range( 1 , logits.get_shape()[0].value):
    t = tf.maximum( tf.add( t, tf.reduce_sum(tf.multiply(logits[i], logits[i])) ), 1 )

train_op = tf.train.AdamOptimizer( 1e-3 ).minimize( t )
# for i in range(logits.shape):
#     sum += logits[i]
xxx = np.array([[1,1,1],[1,1,1]])
print("----------------------------------------\n" * 2)
# print(sum

print(t)
print("----------------------------------------\n" * 2)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print("----------------------------------------\n" * 2)
t1 = 0
t2 = 0
for i in range( 1 ):
    _, t1,t2 = sess.run([train_op,t,b],feed_dict = {a:xxx})
    # if i % 1000 == 0:
    #     print(t1)
    #     print(t2)
    
print(t1)
print(t2)
print(type(t2))
#print( np.sum(t,1) )
#print(x)