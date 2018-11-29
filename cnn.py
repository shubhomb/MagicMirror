import tensorflow as tf
import numpy as np

BATCH_SIZE = 16
IMGX = 32
IMGY = 32
dim = IMGX * IMGY
LAYER1_SIZE = 512
LAYER2_SIZE = 1024
LAYER3_SIZE = 5 #output layer aka number classes
NUM_CLASSES = LAYER3_SIZE
trainsteps = 1000
lr = 0.001

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# Vars
input = tf.placeholder(shape=[None,IMGX,IMGY,1],dtype=tf.float32) # BATCH_SIZE * (IMGX*IMGY) nonflattened
labels = tf.placeholder(shape = [None,LAYER3_SIZE],dtype=tf.int8) # BATCH_SIZE * 1 (each label is a scalar)

with tf.name_scope('conv1'):
    w1 = tf.get_variable(name='w1',initializer=tf.truncated_normal(shape=[dim,LAYER1_SIZE]))
    b1 = tf.get_variable(name='b1',shape=[LAYER1_SIZE])
    variable_summaries(w1)
with tf.name_scope('conv2'):
    w2 = tf.get_variable(name='w2',shape=[LAYER1_SIZE,LAYER2_SIZE])
    b2 = tf.get_variable(name='b2',shape=[LAYER2_SIZE])
    variable_summaries(w2)
with tf.name_scope('fc-3'):
    w3 = tf.get_variable(name='w3',shape=[LAYER2_SIZE,LAYER3_SIZE])
    b3 = tf.get_variable(name='b3',shape=[LAYER3_SIZE])
    variable_summaries(w3)



# Ops
NUM_FILTERS_1 = 32
NUM_FILTERS_2 = 16
f1 = tf.get_variable(name='f1', initializer=tf.truncated_normal([3, 3, 1, NUM_FILTERS_1], stddev=0.5),dtype=tf.float32)
h1 = tf.nn.relu(tf.nn.conv2d(input=input,filter=f1,strides=(1,1,1,1),padding='SAME'))
# Output is  BATCH_SIZE x IMGX x IMGY x NUM_FILTERS 1
h1_c = tf.nn.max_pool(value=h1,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
# Output is  BATCH_SIZE x IMGX/2 x IMGY/2 x NUM_FILTERS_1
f2 = tf.get_variable(name='f2', initializer=tf.truncated_normal([3, 3, NUM_FILTERS_1, NUM_FILTERS_2], stddev=0.5),dtype=tf.float32)
h2 = tf.nn.relu(tf.nn.conv2d(input=h1_c,filter=f2,strides=(1,1,1,1),padding='SAME'))
h2_c = tf.nn.max_pool(value=h2,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
print (h2_c.shape)
h2_cr = tf.reshape(h2_c,tf.constant([-1,np.product(h2_c.shape[1:].as_list())])) # check why this is the way it is
# Output is BATCH_SIZE x IMGX/4 x IMGY/4 x NUM_FILTERS_2

for var in tf.global_variables():
    print (var.name, ': ',var.shape)

logits = tf.matmul(h2_cr,w3) + b3
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits))
variable_summaries(loss)
opt = tf.train.AdamOptimizer(learning_rate=lr)
train_op = opt.minimize(loss)
merged = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range (trainsteps):
        feed_dict = {input: np.ones([BATCH_SIZE, IMGX,IMGY,1]),labels: np.ones([BATCH_SIZE,NUM_CLASSES])}
        out_loss,_,_ = sess.run([loss,merged,train_op],feed_dict=feed_dict)
        print (out_loss)


