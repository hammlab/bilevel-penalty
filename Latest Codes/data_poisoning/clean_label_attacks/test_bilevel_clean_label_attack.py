############## Imports ##############
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from bilevel_poisoning import bilevel_poisoning

################ Hyperparameters #####################
lr_u = 1E-2
lr_v = 1E-3

nepochs = 2001
nepochs_bl = 1001
niter = 10
full_epochs = 5001

beta = 1E-2

Npoison = 1

rho_0 = 1E0
lamb_0 = 1E0
eps_0 = 1E0
nu_0 = 0

c_rho = 1.1
c_lamb = 0.9
c_eps = 0.9

height = 299
width = 299
nch = 3

dogsfishes = np.load('./dogfish_dataset/dataset_dog-fish_train-900_test-300.npz')
X_train = np.array(dogsfishes['X_train'])
Y_train = np.array(dogsfishes['Y_train'])
X_test = np.array(dogsfishes['X_test'])
Y_test = np.array(dogsfishes['Y_test'])

X_train_features = np.load("./dogfish_dataset/X_train_features_inception.npy")

X_test_features = np.load("./dogfish_dataset/X_test_features_inception.npy")

mini = np.min(X_train)
maxi = np.max(X_train)

Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)

sess = tf.Session()
K.set_session(sess)

x_dogfish_tf = tf.placeholder(tf.float32, shape=(None, 2048))
y_dogfish_tf = tf.placeholder(tf.float32, shape=(None, 2))
lr_tf = tf.placeholder(tf.float32,[],'lr_tf') 

# create the base pre-trained model
with tf.variable_scope('base_model', reuse=False):
    base_model = InceptionV3(weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False
    
x_base_tf = tf.placeholder(tf.float32, shape=(None, height, width, 3), name='base')
x_base_features_tf = tf.placeholder(tf.float32, shape=(None, 2048), name='base_features')
y_base_tf = tf.placeholder(tf.float32, shape=(None, 2), name='y_base')
x_target_tf = tf.placeholder(tf.float32, shape=(None, 2048), name='target')
y_target_tf = tf.placeholder(tf.float32, shape=(None, 2), name='y_target')
x_poison_tf = tf.placeholder(tf.float32, shape=(None, height, width, 3), name='poison')
y_poison_tf = tf.placeholder(tf.float32, shape=(None, 2), name='y_poison')

bl_poisoning = bilevel_poisoning(sess, x_dogfish_tf, y_dogfish_tf,
                                 x_base_tf, x_base_features_tf, y_base_tf, x_target_tf, y_target_tf, x_poison_tf, y_poison_tf,
                                 Npoison, height, width, nch,
                                 base_model, beta, mini, maxi,
                                 lr_u, lr_v, rho_0, lamb_0, eps_0, nu_0, c_rho, c_lamb, c_eps)

print("\n\nStarting poisoning procedure\n\n")

found = 0
target_index = -1
unsuccessful = []

save_X_target = []
save_X_base = []
save_X_poisoned = []

for i in range(len(X_test)):
    
    sess.run(tf.variables_initializer(bl_poisoning.var_dogfish))
    sess.run(tf.variables_initializer(bl_poisoning.optimizer_min_dogfish.variables()))
    sess.run(tf.variables_initializer(bl_poisoning.bl.optim_u.variables()))
    sess.run(tf.variables_initializer(bl_poisoning.bl.optim_v.variables()))
    bl_poisoning.bl.reset_penalty_parameters()

    _ = bl_poisoning.train_simple(X_train_features, Y_train, X_test_features, Y_test, nepochs)
    print "Train Accuracy:", bl_poisoning.eval_accuracy(X_train_features, Y_train)
    print "Test Accuracy:", bl_poisoning.eval_accuracy(X_test_features, Y_test)
        
    target_index, pred_label, correct_label = bl_poisoning.find_correct_example(X_test_features, Y_test, target_index + 1)
    print "finding poisoned instance for test item:", target_index, pred_label[0], correct_label[0]
    
    target_instance = np.array(X_test_features[target_index].reshape([1, 2048]))
    target_plot = np.array(X_test[target_index])
    target_correct_label = Y_test[target_index].reshape([1,2])
    print "Target instance Accuracy:", bl_poisoning.eval_accuracy(target_instance, target_correct_label)
    
    a = np.array(X_train_features)
    b = np.array(X_train)
    c = np.array(Y_train)
    
    dist, base_index, correct_label_base = bl_poisoning.find_closest_example(a, c, target_index, correct_label[0], Npoison)
    print "Feature Distance:", dist, base_index, correct_label_base
    
    base_instance = np.array(b[base_index].reshape([Npoison, height, width, 3]))
    base_correct_label = c[base_index].reshape([Npoison,2])
    
    poison_instance = np.array(base_instance.reshape([Npoison, height, width, 3]))
    poison_correct_label = np.array(c[base_index].reshape([Npoison,2]))
    
    bl_poisoning.bl.reset_lower_level()
    
    for epoch in range(nepochs_bl):
        
        f, gvnorm, gv_nu, lamb_g, new_X_poisoned = bl_poisoning.train(X_train_features, Y_train, target_instance, target_correct_label, base_instance, poison_instance, poison_correct_label, niter)
        poison_instance = np.array(new_X_poisoned)
        
        if epoch%200==0:
           rho_t,lamb_t,eps_t = sess.run([bl_poisoning.bl.rho_t,bl_poisoning.bl.lamb_t,bl_poisoning.bl.eps_t])
           print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f + %f= %f'%
               (epoch,rho_t,lamb_t,eps_t,f,gvnorm, gv_nu,lamb_g,f+gvnorm+lamb_g+gv_nu))
           
    rep_u = sess.run(bl_poisoning.representation_u)

    new_X_train = np.array(np.concatenate([X_train_features, rep_u]))
    new_Y_train = np.array(np.concatenate([Y_train, poison_correct_label]))
    
    sess.run(tf.variables_initializer(bl_poisoning.var_dogfish))
    sess.run(tf.variables_initializer(bl_poisoning.optimizer_min_dogfish.variables()))
    
    acc = bl_poisoning.train_simple(new_X_train, new_Y_train, target_instance, target_correct_label, nepochs)
    
    if acc < 1.0:
        print "Yay!! Success"
        found += 1
    else:
        print "No success for this example"
    
    print "Found:", found, " out of ", i+1, "\n\n"
