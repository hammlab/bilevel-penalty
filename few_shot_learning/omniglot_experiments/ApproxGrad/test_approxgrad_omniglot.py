import numpy as np
from scipy.ndimage.interpolation import rotate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.contrib import layers as tcl
import tensorflow as tf
from bilevel_approxgrad_metalearning import bilevel_meta
import time
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#x_omniglot_1: ON pixels are white and OFF are black works better

## Load data
if False:
    import os
    from PIL import Image
    n = 32460 * 4  
    rotations = [0, 90, 180, 270]
    X = np.zeros((n,28,28),np.float32)
    y = np.zeros((n),np.int32)

    dir_data = 'path_to_data/omniglot_resized'
    cnt = 0
    nclass = 0
    for d1 in os.listdir(dir_data):
        print('%s'%(d1))
        for d2 in os.listdir(dir_data+'/'+d1):
            for rot in rotations:
                for d3 in os.listdir(dir_data+'/'+d1+'/'+d2):
                    fname = dir_data+'/'+d1+'/'+d2+'/'+d3
                    img = Image.open(fname)
                    img_data = 1. - np.array(img.getdata(),np.float32).reshape((28,28))/255.0
                    X[cnt,:,:] = rotate(img_data, rot, reshape=False)
                    y[cnt] = nclass
                    cnt += 1
                nclass += 1    
    X = X.reshape(n,28,28,1)
    assert(n==cnt)
    assert(1623*4==nclass)
    print(' ')
    print('n=%d'%(n)) # 32460*4
    print('nclass=%d'%(nclass)) #1623*4
    print('X.shape=', X.shape)
    print('Y.shape=', y.shape)
    
    np.save('../x_omniglot_1.npy',X)
    np.save('../y_omniglot_1.npy',y)
else:
    X = np.load('../x_omniglot_1.npy')
    y = np.load('../y_omniglot_1.npy')
    nclass = 1623*4
    n = 32460*4

test_classes_start = 1200
nclass_per_task = 20# N-way. Same for training and testing    
ntrain_per_cls = 5 # K-shot
ntest_per_cls = 15

height = 28
width = 28
nch = 1

if nclass_per_task == 5:
    n_meta_iterations = 5001
    meta_batch_size = 30
    test_times = 20
else:
    n_meta_iterations = 10001
    meta_batch_size = 15
    test_times = 40

lr_u = 1E-2
lr_v = 1E-2
lr_p = 1E-2
sig = 1E-3

niter_simple = 100
niter_u = 1
niter_v = 20
niter_p = niter_v

def get_tasks(X, Y, for_train = True):
    
    X_train = np.zeros((meta_batch_size, nclass_per_task * ntrain_per_cls, height, width, nch),np.float32)
    Y_train = np.zeros((meta_batch_size, nclass_per_task * ntrain_per_cls, nclass_per_task),np.float32)
    
    X_test = np.zeros((meta_batch_size, nclass_per_task * ntest_per_cls, height, width, nch),np.float32)
    Y_test = np.zeros((meta_batch_size, nclass_per_task * ntest_per_cls, nclass_per_task),np.float32)
    
    already_selected = []
    for batch in range(meta_batch_size):
        
        if for_train:
            class_idx = np.arange(test_classes_start*4) #4800
            start_in = 0
        else:
            class_idx = test_classes_start * 4 + np.arange((1623-test_classes_start) * 4) #4800 + 423*4
            start_in = test_classes_start*4
        
        if len(already_selected) + nclass_per_task >= len(class_idx):
            print("here")
            already_selected = []
        
        class_idx = np.delete(class_idx, already_selected)
    
        np.random.shuffle(class_idx)
        
        selected_idx = class_idx[:nclass_per_task]
        for li_it in selected_idx:
            already_selected.append(li_it - start_in)
        #print(sorted(selected_idx))
            
        start_train = 0
        start_test = 0
        for num_cls in range(nclass_per_task):
            
            idxs = np.argwhere(Y == selected_idx[num_cls]).flatten()
            np.random.shuffle(idxs)
            idxs_train = idxs[:ntrain_per_cls]
            idxs_test = idxs[ntrain_per_cls:ntrain_per_cls+ntest_per_cls]
            
            X_train[batch][start_train:start_train+ntrain_per_cls] = X[idxs_train]
            X_test[batch][start_test:start_test+ntest_per_cls] = X[idxs_test]
            
            Y_train[batch][start_train:start_train+ntrain_per_cls] = np.zeros([ntrain_per_cls, nclass_per_task])
            for j in range(ntrain_per_cls):
                Y_train[batch][start_train + j][num_cls] = 1
            
            Y_test[batch][start_test:start_test+ntest_per_cls] = np.zeros([ntest_per_cls, nclass_per_task])
            for j in range(ntest_per_cls):
                Y_test[batch][start_test + j][num_cls] = 1
            
            start_train += ntrain_per_cls
            start_test += ntest_per_cls
            
        task_idx_train = np.arange((ntrain_per_cls) * nclass_per_task)
        np.random.shuffle(task_idx_train)
        X_train[batch] = X_train[batch][task_idx_train]
        Y_train[batch] = Y_train[batch][task_idx_train]
        
        task_idx_test = np.arange((ntest_per_cls) * nclass_per_task)
        np.random.shuffle(task_idx_test)
        X_test[batch] = X_test[batch][task_idx_test]
        Y_test[batch] = Y_test[batch][task_idx_test]
     
    #print(sorted(already_selected), len(np.unique(already_selected)))
    return X_train, Y_train, X_test, Y_test


if False:
    print("Testing splits")
    X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test = get_tasks(X, y, for_train = False)
    print(X_metatrain_train.shape, Y_metatrain_train.shape, X_metatrain_test.shape, Y_metatrain_test.shape)
    for meta_it in range(meta_batch_size):
        print("train", Y_metatrain_train[meta_it])
        print("test", Y_metatrain_test[meta_it])
        
        fig = plt.figure(1)
        grid = ImageGrid(fig, 121, nrows_ncols=(10, 10), axes_pad = 0.01)
        for i in range(10**2):
            if i < 25:
                grid[i].imshow(X_metatrain_train[meta_it][i].reshape(28, 28), cmap='gray')
            else:
                grid[i].imshow(X_metatrain_test[meta_it][i-25].reshape(28, 28), cmap='gray')
            grid[i].axis('off')  
        plt.savefig('metatrain_batch_'+str(meta_it)+'.pdf', bbox_inches='tight')
        print("\n\n")

########################################  MODEL  ###############################################
istraining_ph = tf.placeholder_with_default(True,shape=())
def make_filter1(ins):
    W1 = tf.get_variable('W1',[3,3,1,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[64],initializer=tf.constant_initializer(0.0))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
    p1 = tf.layers.dropout(p1, rate=0.25, training=istraining_ph)

    W2 = tf.get_variable('W2',[3,3,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable('b2',[64],initializer=tf.constant_initializer(0.0))
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
    p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
    p2 = tf.layers.dropout(p2, rate=0.25, training=istraining_ph)
    
    W3 = tf.get_variable('W3',[3,3,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b3 = tf.get_variable('b3',[64],initializer=tf.constant_initializer(0.0))
    c3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p2, W3, strides=[1,1,1,1], padding='SAME'), b3))
    p3 = tf.nn.max_pool(c3, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
    p3 = tf.layers.dropout(p3, rate=0.25, training=istraining_ph)    
    
    W4 = tf.get_variable('W4',[3,3,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b4 = tf.get_variable('b4',[64],initializer=tf.constant_initializer(0.0))
    c4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p3, W4, strides=[1,1,1,1], padding='SAME'), b4))
    p4 = tf.nn.max_pool(c4, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
    p4 = tf.layers.dropout(p4, rate=0.25, training=istraining_ph)    
    a4 = tf.reshape(p4,[-1,64])
    out = a4
    return out

def make_classifier1(ins,d=64,K=nclass_per_task):
    W1 = tf.get_variable('W1',[d,K],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[K],initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(tf.matmul(ins,W1),b1)
    return out

def conv_block_old(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tcl.conv2d(inputs, out_channels, 3, stride = 2, activation_fn=None, normalizer_fn=tcl.batch_norm, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32))
        if name != 'conv_4':
            conv = tf.nn.relu(conv)
        
        return conv
    
def conv_block_new(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tcl.conv2d(inputs, out_channels, 3, activation_fn=None, normalizer_fn=tcl.batch_norm, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32))
        if name != 'conv_4':
            conv = tf.nn.relu(conv)
        conv = tf.layers.max_pooling2d(conv, pool_size=2, strides=2)
        return conv
    
def encoder(x, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = conv_block_new(x, 64, name='conv_1')
        net = conv_block_new(net, 64, name='conv_2')
        net = conv_block_new(net, 64, name='conv_3')
        net = conv_block_new(net, 64, name='conv_4')
        net = tf.reshape(net, (-1, 64))
        return net   
    
def top_layer(ins, K=20):
    W_top = tf.get_variable('W_top',[64, K],initializer=tf.random_normal_initializer(stddev=0.01))
    b_top = tf.get_variable('b_top',[K],initializer=tf.constant_initializer(0.001))
    
    wat = 0.001
    ins_norm = tf.sqrt(tf.reduce_sum(ins**2, axis = 1, keep_dims = True) + wat**2)
    W_top_norm = tf.sqrt(tf.reduce_sum(W_top**2, axis = 0, keep_dims=True) + b_top**2)
    
    out = 10. * (tf.matmul(ins, W_top) + wat * b_top)/(ins_norm * W_top_norm)
    return out

def top_layer_norm(ins, K=20):
    W_top = tf.get_variable('W_top',[64, K],initializer=tf.random_normal_initializer(stddev=0.01))
    b_top = tf.get_variable('b_top',[K],initializer=tf.constant_initializer(0.001))
    
    out = (tf.matmul(ins, W_top) + b_top)
    
    return out


tf.set_random_seed(1234)
sess=tf.Session(config=config)

print('\n\nDefining models:')
x_train_ph = tf.placeholder(tf.float32, shape=(meta_batch_size,None,height,width,nch))
y_train_ph = tf.placeholder(tf.float32, shape=(meta_batch_size,None,nclass_per_task))
x_test_ph = tf.placeholder(tf.float32, shape=(meta_batch_size,None,height,width,nch))
y_test_ph = tf.placeholder(tf.float32, shape=(meta_batch_size,None,nclass_per_task))

filt_train = [[] for i in range(meta_batch_size)]
filt_test = [[] for i in range(meta_batch_size)]    
cls_train = [[] for i in range(meta_batch_size)]
cls_test = [[] for i in range(meta_batch_size)]    

with tf.variable_scope('filt',reuse=False):
    #_ = make_filter1(tf.zeros_like(x_train_ph[0]))
    _ = encoder(tf.zeros_like(x_train_ph[0]))
var_filt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='filt')

total_parameters = 0
for variable in var_filt:
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print(total_parameters) 

with tf.variable_scope('filt',reuse=True):
    for i in range(meta_batch_size):
        #filt_train[i] = make_filter1(x_train_ph[i])
        #filt_test[i] = make_filter1(x_test_ph[i])
        filt_train[i] = encoder(x_train_ph[i])
        filt_test[i] = encoder(x_test_ph[i])

var_cls = [[] for i in range(meta_batch_size)]
for i in range(meta_batch_size):
    with tf.variable_scope('cls'+str(i),reuse=False):
        #cls_train[i] = make_classifier1(filt_train[i])
        cls_train[i] = top_layer(filt_train[i],nclass_per_task)
    with tf.variable_scope('cls'+str(i),reuse=True):
        #cls_test[i] = make_classifier1(filt_test[i])
        cls_test[i] = top_layer(filt_test[i],nclass_per_task)
    var_cls[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls'+str(i))

total_parameters = 0
for variable in var_cls[0]:
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print(total_parameters) 

print('Done')

#########################################################################################################
## Bilevel training
#########################################################################################################

blmt = bilevel_meta(sess,x_train_ph,x_test_ph,y_train_ph,y_test_ph,
    cls_train,cls_test,var_filt,var_cls,
    meta_batch_size,ntrain_per_cls,ntest_per_cls,nclass_per_task,
    lr_u, lr_v, lr_p, sig,istraining_ph)

sess.run(tf.global_variables_initializer())

if False:
    print('\n\nPre-training:')
    for meta_it in range(1000):
        
        X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test = get_tasks(X, y, for_train = True)
        blmt.reset_v_func()
       
        X_batch = np.concatenate([X_metatrain_train, X_metatrain_test], axis = 1)
        Y_batch = np.concatenate([Y_metatrain_train, Y_metatrain_test], axis = 1)
        for i in range(niter_u):
            g = blmt.update_simple(X_batch, Y_batch, niter_v)

        if meta_it % 100 == 0:
            print('meta_it %d: g=%f'%(meta_it, g))
            
            ## Metatrain-test
            print('Metatrain-test:')
            accs = np.nan*np.ones(test_times * meta_batch_size)
            counter = 0
            for i in range(test_times):
                X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test = get_tasks(X, y, for_train = True)
                blmt.reset_v_func()
                l1 = blmt.update_cls_simple(X_metatrain_train,Y_metatrain_train,niter_simple)
                pred = sess.run(cls_test, {x_test_ph:X_metatrain_test, istraining_ph:False})
    
                ## Metatrain-test error
                for j in range(meta_batch_size):
                    accs[counter] = np.mean(np.argmax(pred[j], 1)==np.argmax(Y_metatrain_test[j], 1))
                    counter += 1
            print('mean acc = %f'%(accs.mean()))
            
            ## Metatrain-test
            print('Metatest-test:')
            accs = np.nan*np.ones(test_times * meta_batch_size)
            counter = 0
            for i in range(test_times):
                X_metatest_train, Y_metatest_train, X_metatest_test, Y_metatest_test = get_tasks(X, y, for_train = False)
                blmt.reset_v_func()
                l1 = blmt.update_cls_simple(X_metatest_train,Y_metatest_train,niter_simple)
                pred = sess.run(cls_test, {x_test_ph:X_metatest_test, istraining_ph:False})

                ## Metatrain-test error
                for j in range(meta_batch_size):
                    accs[counter] = np.mean(np.argmax(pred[j], 1)==np.argmax(Y_metatest_test[j], 1))
                    counter += 1
            print('mean acc = %f\n\n'%(accs.mean()))


if True:
    print('\n\nBilevel-training:')
    max_acc = 0
    start = time.time()
    for meta_it in range(1, n_meta_iterations):
        #if meta_it == 2:
        #    start = time.time()
        X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test = get_tasks(X, y, for_train = True)
        blmt.reset_v_func()
        for i in range(niter_u):
            fval, gval, hval = blmt.update(X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test, niter_v, niter_p)
            
        if meta_it % 1000 == 0:
            print(meta_it, nclass_per_task, ntrain_per_cls, niter_v)
            print('meta_it=', meta_it, ' fval = ', fval, ' gval = ', gval, ' hval = ', hval)
            
            ## Test phase is not bilevel. For each task, simply train with metatest-train and test with metatest-test.
            ## Metatrain-test
            print('Metatest-test:', ntrain_per_cls)
            accs = np.nan*np.ones(test_times * meta_batch_size)
            counter = 0
            for i in range(test_times):
                X_metatest_train, Y_metatest_train, X_metatest_test, Y_metatest_test = get_tasks(X, y, for_train = False)
                blmt.reset_v_func()
                l1 = blmt.update_cls_simple(X_metatest_train,Y_metatest_train,niter_simple)
                pred = sess.run(cls_test, {x_test_ph:X_metatest_test, istraining_ph:False})
    
                ## Metatrain-test error
                for j in range(meta_batch_size):
                    accs[counter] = np.mean(np.argmax(pred[j], 1)==np.argmax(Y_metatest_test[j], 1))
                    counter += 1
            print('mean acc = %f'%(accs.mean()))