import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import pickle
from bilevel_approxgrad_metalearning import bilevel_meta
from tensorflow.contrib import layers as tcl
from keras.preprocessing.image import ImageDataGenerator
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

datagen = ImageDataGenerator(
        #rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True#,
        #fill_mode='nearest'
        )

X_train = np.zeros([64*600, 84, 84, 3])
X_val = np.zeros([16*600, 84, 84, 3])
X_test = np.zeros([20*600, 84, 84, 3])

Y_train = np.zeros([64*600])
Y_val = np.zeros([16*600])
Y_test = np.zeros([20*600])

data_augmentation = False
conv = False
pretrain = False

nclass_per_task = 5# N-way. Same for training and testing    
ntrain_per_cls = 1 # K-shot
ntest_per_cls = 15

if conv:
    meta_batch_size = 4
else:
    meta_batch_size = 2
    
test_times = 600//meta_batch_size
height = 84
width = 84
nch = 3

lr_u = 1E-3
lr_v = 1E-3
lr_p = 1E-3
sig = 1E-3

if conv:
    scale = 10.
    if pretrain:
        niter_simple = 100
        niter_u = 2
        niter_v = 10
        niter_p = niter_v
    else:
        niter_simple = 100
        niter_u = 2
        niter_v = 20
        niter_p = niter_v
else:
    #Resnet
    scale = 10.
    if pretrain:
        niter_simple = 100
        niter_u = 2
        niter_v = 10
        niter_p = niter_v
    else:
        niter_simple = 100
        niter_u = 1
        niter_v = 20
        niter_p = niter_v

with open('../dataset_miniimagenet/mini-imagenet-cache-train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('../dataset_miniimagenet/mini-imagenet-cache-val.pkl', 'rb') as f:
    val_data = pickle.load(f)

with open('../dataset_miniimagenet/mini-imagenet-cache-test.pkl', 'rb') as f:
    test_data = pickle.load(f)    
    

for i in range(len(train_data['image_data'])):
    X_train[i] = train_data['image_data'][i]/255.
    Y_train[i] = int(i/600.)
    
for i in range(len(val_data['image_data'])):
    X_val[i] = val_data['image_data'][i]/255.
    Y_val[i] = int(i/600.)
    
for i in range(len(test_data['image_data'])):
    X_test[i] = test_data['image_data'][i]/255.
    Y_test[i] = int(i/600.)
    
def get_tasks(X, Y, num_classes, test = False):
    
    train_items_per_class = ntrain_per_cls
    
    X_train_batch = np.zeros((meta_batch_size, nclass_per_task * train_items_per_class, height, width, nch),np.float32)
    Y_train_batch = np.zeros((meta_batch_size, nclass_per_task * train_items_per_class, nclass_per_task),np.float32)
    
    X_test_batch = np.zeros((meta_batch_size, nclass_per_task * ntest_per_cls, height, width, nch),np.float32)
    Y_test_batch = np.zeros((meta_batch_size, nclass_per_task * ntest_per_cls, nclass_per_task),np.float32)
    
    already_selected = []
    for batch in range(meta_batch_size):
        class_idx = np.arange(num_classes)
        if len(already_selected) + nclass_per_task >= len(class_idx):
            #print("here")
            already_selected = []
        
        class_idx = np.delete(class_idx, already_selected)
        
        np.random.shuffle(class_idx)
        selected_idx = class_idx[:nclass_per_task]
        #print(selected_idx)
        for li_it in selected_idx:
            already_selected.append(li_it)
        
        start_train = 0
        start_test = 0
        for num_cls in range(nclass_per_task):
            idxs = np.argwhere(Y == selected_idx[num_cls]).flatten()
            np.random.shuffle(idxs)
            idxs_train = idxs[:train_items_per_class]
            idxs_test = idxs[train_items_per_class:train_items_per_class+ntest_per_cls]
            
            if data_augmentation and not test:
                for train_X_batch in datagen.flow(X[idxs_train], batch_size=len(idxs_train), shuffle=False):
                    break
                
                for test_X_batch in datagen.flow(X[idxs_test], batch_size=len(idxs_test), shuffle=False):
                    break
                
                X_train_batch[batch][start_train:start_train+train_items_per_class] = train_X_batch
                X_test_batch[batch][start_test:start_test+ntest_per_cls] = test_X_batch
            
            else:
                X_train_batch[batch][start_train:start_train+train_items_per_class] = X[idxs_train]
                X_test_batch[batch][start_test:start_test+ntest_per_cls] = X[idxs_test]
            
            Y_train_batch[batch][start_train:start_train+train_items_per_class] = np.zeros([train_items_per_class, nclass_per_task])
            for j in range(train_items_per_class):
                Y_train_batch[batch][start_train + j][num_cls] = 1
            
            Y_test_batch[batch][start_test:start_test+ntest_per_cls] = np.zeros([ntest_per_cls, nclass_per_task])
            for j in range(ntest_per_cls):
                Y_test_batch[batch][start_test + j][num_cls] = 1
            
            start_train += train_items_per_class
            start_test += ntest_per_cls
        
        task_idx_train = np.arange((train_items_per_class) * nclass_per_task)
        np.random.shuffle(task_idx_train)
        X_train_batch[batch] = X_train_batch[batch][task_idx_train]
        Y_train_batch[batch] = Y_train_batch[batch][task_idx_train]
        
        task_idx_test = np.arange((ntest_per_cls) * nclass_per_task)
        np.random.shuffle(task_idx_test)
        X_test_batch[batch] = X_test_batch[batch][task_idx_test]
        Y_test_batch[batch] = Y_test_batch[batch][task_idx_test]
    
    #print(sorted(already_selected), len(np.unique(already_selected)))
    return X_train_batch, Y_train_batch, X_test_batch, Y_test_batch

if False:
    print("Testing splits by plotting meta-batches")
    X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test = get_tasks(X_train, Y_train, num_classes = 64, test = False)
    if True:
        print(X_metatrain_train.shape, Y_metatrain_train.shape, X_metatrain_test.shape, Y_metatrain_test.shape)
        for meta_it in range(meta_batch_size):
            print("train", Y_metatrain_train[meta_it])
            print("test", Y_metatrain_test[meta_it])
            
            fig = plt.figure(1)
            grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad = 0.01)
            for i in range(10**2):
                if i < 25:
                    grid[i].imshow(X_metatrain_train[meta_it][i])
                else:
                    grid[i].imshow(X_metatrain_test[meta_it][i-25])
                grid[i].axis('off')  
            plt.savefig('batch_'+str(meta_it)+'.pdf', bbox_inches='tight')
            print("\n\n")

istraining_ph = tf.placeholder_with_default(True,shape=())
def conv_block(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True, is_training = istraining_ph)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv
    
def conv_block_new(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tcl.conv2d(inputs, out_channels, 3, activation_fn=None, normalizer_fn=tcl.batch_norm)
        if name != 'conv_4':
            conv = tf.nn.relu(conv)
        conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        return conv
    
def encoder(x, h_dim = 32, z_dim = 32, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = conv_block_new(x, h_dim, name='conv_1')
        net = conv_block_new(net, h_dim, name='conv_2')
        net = conv_block_new(net, z_dim, name='conv_3')
        net = conv_block_new(net, z_dim, name='conv_4')
        net = tf.contrib.layers.flatten(net)
        return net        

def make_resnet_filter(ins, reuse=False):
    def residual_block(x, n_filters, name='res'):

        def conv_block_resnet(xx, name='res'):
            with tf.variable_scope(name):
                out = tcl.conv2d(xx, n_filters, 3, activation_fn=None, normalizer_fn=tcl.batch_norm)
                return tf.nn.leaky_relu(out, 0.1)

        with tf.variable_scope(name):
            skip_c = tcl.conv2d(x, n_filters, 1, activation_fn=None)
            out = x
            for idx in range(3):
                out = conv_block_resnet(out, name+'_conv_'+str(idx))
    
            add = tf.add(skip_c, out)
    
            return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    with tf.variable_scope('resnet_filter', reuse=reuse):
        x = residual_block(ins, 64, name='res_1')
        x = residual_block(x, 96, name='res_2')
        x = residual_block(x, 128, name='res_3')
        x = residual_block(x, 256, name='res_4')
        x = tcl.conv2d(x, 2048, 1)
        x = tf.nn.avg_pool(x, [1, 6, 6, 1], [1, 6, 6, 1], 'VALID')
        x = tcl.conv2d(x, 384, 1, activation_fn = None)
        x = tf.reshape(x, (-1, 384))  
        return x

def top_layer(ins, K=5):
    
    if conv:
        W_top = tf.get_variable('W_top',[800, K],initializer=tf.random_normal_initializer(stddev=0.01))
        b_top = tf.get_variable('b_top',[K],initializer=tf.constant_initializer(0.001))
    else:
        W_top = tf.get_variable('W_top',[384, K],initializer=tf.random_normal_initializer(stddev=0.01))
        b_top = tf.get_variable('b_top',[K],initializer=tf.constant_initializer(0.001))
        
    wat = 0.001
    ins_norm = tf.sqrt(tf.reduce_sum(ins**2, axis = 1, keep_dims = True) + wat**2)
    W_top_norm = tf.sqrt(tf.reduce_sum(W_top**2, axis = 0, keep_dims=True) + b_top**2)
    
    out = 10. * (tf.matmul(ins, W_top) + wat * b_top)/(ins_norm * W_top_norm)
    return out

tf.set_random_seed(1234)
sess=tf.Session(config=config)

x_train_ph = tf.placeholder(tf.float32, shape=(meta_batch_size,None,height,width,nch))
y_train_ph = tf.placeholder(tf.float32, shape=(meta_batch_size,None,nclass_per_task))
x_test_ph = tf.placeholder(tf.float32, shape=(meta_batch_size,None,height,width,nch))
y_test_ph = tf.placeholder(tf.float32, shape=(meta_batch_size,None,nclass_per_task))

x_pretrain = tf.placeholder(tf.float32, shape=(None, height,width,nch))
y_pretrain = tf.placeholder(tf.float32, shape=(None, 64))

filt_train = [[] for i in range(meta_batch_size)]
filt_test = [[] for i in range(meta_batch_size)]    
cls_train = [[] for i in range(meta_batch_size)]
cls_test = [[] for i in range(meta_batch_size)]    

with tf.variable_scope('filt', reuse=False):
    if conv:
        _ = encoder(tf.zeros_like(x_train_ph[0]))
    else:
        _ = make_resnet_filter(tf.zeros_like(x_train_ph[0]))

var_filt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='filt')

with tf.variable_scope('filt',reuse=True):
    for i in range(meta_batch_size):
        if conv:
            filt_train[i] = encoder(x_train_ph[i], reuse=True)
            filt_test[i] = encoder(x_test_ph[i], reuse = True)
        else:
            filt_train[i] = make_resnet_filter(x_train_ph[i], reuse=True)
            filt_test[i] = make_resnet_filter(x_test_ph[i], reuse = True)
        
var_cls = [[] for i in range(meta_batch_size)]
for i in range(meta_batch_size):
    with tf.variable_scope('cls'+str(i), reuse=False):
        cls_train[i] = top_layer(filt_train[i])
    with tf.variable_scope('cls'+str(i), reuse=True):
        cls_test[i] = top_layer(filt_test[i])
    var_cls[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls'+str(i))
    
blmt = bilevel_meta(sess,x_train_ph,x_test_ph,y_train_ph,y_test_ph,
    cls_train,cls_test,var_filt,var_cls,
    meta_batch_size,ntrain_per_cls,ntest_per_cls,nclass_per_task,
    lr_u,lr_v,lr_p,sig,istraining_ph)

sess.run(tf.global_variables_initializer())

if pretrain:
    print('\n\nPre-training (No Bilevel training is involved):')
    n_meta_iterations = 10000
    max_mean_acc = 0
    for meta_it in range(n_meta_iterations):
        
        X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test = get_tasks(X_train, Y_train, 64, test = False)
        blmt.reset_v_func()
        
        X_batch = np.concatenate([X_metatrain_train, X_metatrain_test], axis = 1)
        Y_batch = np.concatenate([Y_metatrain_train, Y_metatrain_test], axis = 1)
        for i in range(niter_u):
            g = blmt.update_simple(X_batch, Y_batch, niter_v)
            
        if meta_it % (test_times/5.) == 0:
            print(meta_it, g, " shot=", ntrain_per_cls)
        
        if (meta_it + 1) % test_times == 0:
            print('meta_it %d: g=%f'%(meta_it, g))
            
            ## Metatrain-test
            print('Metatest-test:')
            accs = np.nan*np.ones(test_times * meta_batch_size)
            counter = 0
            for i in range(test_times):
                X_metatest_train, Y_metatest_train, X_metatest_test, Y_metatest_test = get_tasks(X_test, Y_test, 20, test = True)
                blmt.reset_v_func()
                l1 = blmt.update_cls_simple(X_metatest_train,Y_metatest_train,niter_simple)
                pred = sess.run(cls_test, {x_test_ph:X_metatest_test, istraining_ph:False})
    
                ## Metatrain-test error
                for j in range(meta_batch_size):
                    accs[counter] = np.mean(np.argmax(pred[j], 1)==np.argmax(Y_metatest_test[j], 1))
                    counter += 1
            print('mean acc = %f'%(accs.mean()), '\n\n')
            
            
            
if not pretrain:
    print('\n\nBilevel-training-Approxgrad:')
    
    if ntrain_per_cls == 5:
        n_meta_iterations = 10001
    else:
        n_meta_iterations = 20001
        
        
    max_mean_acc = 0
    for meta_it in range(n_meta_iterations):
        
        X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test = get_tasks(X_train, Y_train, 64, test = False)
        blmt.reset_v_func()
        for it in range(niter_u):
            fval, gval, hval = blmt.update(X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test, niter_v, niter_p)
        
        if (meta_it + 1) % 1000 == 0:#test_times == 0 or meta_it == n_meta_iterations - 1:
            print(meta_it, nclass_per_task, ntrain_per_cls)
            print('meta_it=', meta_it, ' fval = ', fval, ' gval = ', gval, ' hval = ', hval)
        
            
            ## Test phase is not bilevel. For each task, simply train with metatest-train and test with metatest-test.
            ## Metatrain-test
            print('Metatest-test:')
            accs = np.nan*np.ones(test_times * meta_batch_size)
            counter = 0
            for i in range(test_times):
                X_metatest_train, Y_metatest_train, X_metatest_test, Y_metatest_test = get_tasks(X_test, Y_test, 20, test = True)
                blmt.reset_v_func()
                l1 = blmt.update_cls_simple(X_metatest_train,Y_metatest_train,niter_simple)
                pred = sess.run(cls_test, {x_test_ph:X_metatest_test, istraining_ph:False})
    
                ## Metatrain-test error
                for j in range(meta_batch_size):
                    #print(counter)
                    accs[counter] = np.mean(np.argmax(pred[j], 1)==np.argmax(Y_metatest_test[j], 1))
                    counter += 1
            
            print('mean acc = %f'%(accs.mean()))
           