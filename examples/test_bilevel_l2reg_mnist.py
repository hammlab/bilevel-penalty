## test_bilevel_l2reg_mnist.py

# min_s sum_val logreg(y*x^Tw) s.t. w = argmin sum_train logreg(yx^Tw) + \sum_ij exp(s_ij)*w_ij^2


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
sys.path.append('/home/hammj/Dropbox/Research/AdversarialLearning/codes/lib/cleverhans-master')

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from cleverhans.utils_mnist import data_mnist

from bilevel_l2reg import bilevel_l2reg

import time

lr_u = 1E-3
lr_v = 1E-3

nepochs = 301
niter = 1
batch_size = 500#128

rho_0 = 1E0
lamb_0 = 1E0
eps_0 = 1E0

c_rho = 2
c_lamb = 0.5
c_eps = 0.5


def make_classifier(ins,K=10):
    d = np.prod(ins.shape[1:])
    W1 = tf.get_variable('W1',[d,K],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[K],initializer=tf.constant_initializer(0.0))
    return tf.nn.bias_add(tf.matmul(tf.reshape(ins,[-1,d]),W1),b1)


def main(argv=None):

    tf.set_random_seed(1234)
    sess = tf.Session()

    ## Read data
    height = 28
    width = 28
    nch = 1
    nclass = 10
    #num_class = 10
    
    tX, tY, X_test, Y_test = data_mnist(train_start=0, train_end=60000,test_start=0,test_end=10000)
    X_train = tX[:50000,:]
    Y_train = tY[:50000,:]
    X_val = tX[50000:60000,:]
    Y_val = tY[50000:60000,:]
    Ntrain = X_train.shape[0]
    Nval = X_val.shape[0]
    Ntest = X_test.shape[0]

    ## Define model
    #print('\n\nDefining models:')
    x_train_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_train_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))
    x_test_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_test_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))

    with tf.variable_scope('cls',reuse=False):
        cls_train = make_classifier(x_train_tf)
    var_cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls')
    with tf.variable_scope('cls',reuse=True):
        cls_test = make_classifier(x_test_tf)
    #print('Done')
    #saver_model = tf.train.Saver(var_model,max_to_keep=none)



    #########################################################################################################
    ## Bilevel training
    #########################################################################################################

    #print('\n\nSetting up graphs:')
    bl_l2reg = bilevel_l2reg(sess,x_train_tf,x_test_tf,y_train_tf,y_test_tf,cls_train,cls_test,var_cls,
        batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)
    #print('Done')
    sess.run(tf.global_variables_initializer())

    ## Metatrain
    #print('\n\nTraining start:')
    if True:
        for epoch in range(nepochs):
            #tick = time.time()        
            nb_batches = int(np.floor(float(Ntrain) / batch_size))
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
                ind_tr = ind_shuf[ind_batch]
                ind_val = np.random.choice(Nval, size=(batch_size), replace=False)
                f,gvnorm,lamb_g,l2reg = bl_l2reg.train(X_train[ind_tr,:],Y_train[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:],niter)

            if epoch%10==0:
                rho_t,lamb_t,eps_t = sess.run([bl_l2reg.bl.rho_t,bl_l2reg.bl.lamb_t,bl_l2reg.bl.eps_t])
                #sig = sess.run(bl.sig)
                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f, l2reg=%f'%
                    (epoch,rho_t,lamb_t,eps_t,f,gvnorm,lamb_g,f+gvnorm+lamb_g,l2reg))

            if epoch%10==0:
                ## Measure test error
                nb_batches = int(np.floor(float(Ntest) / batch_size))
                acc = 0#np.nan*np.ones(nb_batches)
                for batch in range(nb_batches):
                    ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
                    pred = sess.run(cls_test, {x_test_tf:X_test[ind_batch,:]})
                    acc += np.sum(np.argmax(pred,1)==np.argmax(Y_test[ind_batch,:],1))
                acc /= np.float32(nb_batches*batch_size)
                print('mean acc = %f\n'%(acc))





    #########################################################################################################
    ## Single-level training
    #########################################################################################################
    ## rho=0, lamb fixed

    if True:
        lambs = [1E-2, 1E-1, 1E0, 1E1, 1E2]
        for lamb in lambs:
            print('\nlamb=%f'%(lamb))
            bl_l2reg = bilevel_l2reg(sess,x_train_tf,x_test_tf,y_train_tf,y_test_tf,cls_train,cls_test,var_cls,
                batch_size,lr_u,lr_v,0.,lamb,0.,1.,1.,1.)
            sess.run(tf.global_variables_initializer())
            for epoch in range(nepochs):
                #tick = time.time()        
                nb_batches = int(np.floor(float(Ntrain) / batch_size))
                ind_shuf = np.arange(Ntrain)
                np.random.shuffle(ind_shuf)
                for batch in range(nb_batches):
                    ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
                    ind_tr = ind_shuf[ind_batch]
                    ind_val = np.random.choice(Nval, size=(batch_size), replace=False)
                    l = bl_l2reg.train_singlelevel(X_train[ind_tr,:],Y_train[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:],niter)

                if epoch%50==0:
                    ## Measure test error
                    nb_batches = int(np.floor(float(Ntest) / batch_size))
                    acc = 0#np.nan*np.ones(nb_batches)
                    for batch in range(nb_batches):
                        ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
                        pred = sess.run(cls_test, {x_test_tf:X_test[ind_batch,:]})
                        acc += np.sum(np.argmax(pred,1)==np.argmax(Y_test[ind_batch,:],1))
                    acc /= np.float32(nb_batches*batch_size)
                    print('mean acc = %f\n'%(acc))









                
    sess.close()




##############################################################################################################

if __name__ == '__main__':

    tf.app.run()



'''
mean acc = 0.887921

mean acc = 0.906450

mean acc = 0.901242

mean acc = 0.898237

mean acc = 0.896735

mean acc = 0.897035

mean acc = 0.897937

mean acc = 0.897536

mean acc = 0.898538

mean acc = 0.898938

mean acc = 0.899439

mean acc = 0.899038

mean acc = 0.900040


lamb=0.100000
mean acc = 0.873097

mean acc = 0.914363

mean acc = 0.912961

mean acc = 0.909355

mean acc = 0.908554

mean acc = 0.909555

mean acc = 0.907652

mean acc = 0.908153

mean acc = 0.905749

mean acc = 0.908554

mean acc = 0.906851

mean acc = 0.906150

mean acc = 0.906550


lamb=1.000000
mean acc = 0.832933

mean acc = 0.924880

mean acc = 0.924279

mean acc = 0.924579

mean acc = 0.922276

mean acc = 0.922075

mean acc = 0.921374

mean acc = 0.921074

mean acc = 0.920172

mean acc = 0.919071

mean acc = 0.918470

mean acc = 0.919671

mean acc = 0.919171


lamb=10.000000
mean acc = 0.805389

mean acc = 0.927684

mean acc = 0.927985

mean acc = 0.924980

mean acc = 0.927083

mean acc = 0.924479

mean acc = 0.924880

mean acc = 0.925982

mean acc = 0.924479

mean acc = 0.924079

mean acc = 0.923778

mean acc = 0.923678

mean acc = 0.923377


lamb=100.000000
mean acc = 0.814804

mean acc = 0.926282

mean acc = 0.926883

mean acc = 0.925881

mean acc = 0.925881

mean acc = 0.924079

mean acc = 0.923478

mean acc = 0.923177

mean acc = 0.921875

mean acc = 0.923377

mean acc = 0.921274

mean acc = 0.921575

mean acc = 0.920773

epoch 0 (rho=1.000000, lamb=1.000000, eps=1.000000): h=1.304885 + 0.174115 + 1.878200= 3.357201, l2reg=0.570757
mean acc = 0.820613

epoch 10 (rho=4.000000, lamb=0.250000, eps=0.250000): h=0.323551 + 0.293847 + 0.252133= 0.869531, l2reg=0.516919
epoch 20 (rho=8.000000, lamb=0.125000, eps=0.125000): h=0.243467 + 0.378158 + 0.097165= 0.718790, l2reg=0.244984
epoch 30 (rho=8.000000, lamb=0.125000, eps=0.125000): h=0.204220 + 0.412204 + 0.055400= 0.671825, l2reg=0.121704
epoch 40 (rho=8.000000, lamb=0.125000, eps=0.125000): h=0.165349 + 0.315520 + 0.068279= 0.549148, l2reg=0.066003
epoch 50 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.147352 + 0.905669 + 0.026546= 1.079568, l2reg=0.058139
mean acc = 0.927784

epoch 60 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.152770 + 0.406927 + 0.019230= 0.578927, l2reg=0.048880
epoch 70 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.138702 + 0.672417 + 0.016507= 0.827625, l2reg=0.041971
epoch 80 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.065768 + 1.193455 + 0.045359= 1.304582, l2reg=0.034632
epoch 90 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.119750 + 0.338315 + 0.009252= 0.467317, l2reg=0.029374
epoch 100 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.136452 + 0.217564 + 0.009527= 0.363543, l2reg=0.025251
mean acc = 0.929888

epoch 110 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.205220 + 0.751543 + 0.020122= 0.976886, l2reg=0.021299
epoch 120 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.167244 + 0.480274 + 0.014251= 0.661770, l2reg=0.018203
epoch 130 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.246811 + 0.538770 + 0.014283= 0.799863, l2reg=0.015834
epoch 140 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.129097 + 0.547377 + 0.014838= 0.691311, l2reg=0.013862
epoch 150 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.183007 + 0.903759 + 0.027154= 1.113920, l2reg=0.011924
mean acc = 0.929187

epoch 160 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.246024 + 0.607000 + 0.015228= 0.868252, l2reg=0.010846
epoch 170 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.116473 + 0.184750 + 0.006388= 0.307612, l2reg=0.009793
epoch 180 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.090430 + 0.403276 + 0.014768= 0.508475, l2reg=0.008842
epoch 190 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.108554 + 0.244027 + 0.008887= 0.361467, l2reg=0.007695
epoch 200 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.070808 + 0.512845 + 0.028272= 0.611925, l2reg=0.007198
mean acc = 0.929387

epoch 210 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.097449 + 0.311826 + 0.011959= 0.421234, l2reg=0.006523
epoch 220 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.119362 + 0.866734 + 0.036423= 1.022519, l2reg=0.006088
epoch 230 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.114647 + 0.784396 + 0.014984= 0.914027, l2reg=0.005628
epoch 240 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.168652 + 0.183456 + 0.007209= 0.359316, l2reg=0.005201
epoch 250 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.151567 + 0.769947 + 0.034152= 0.955665, l2reg=0.004713
mean acc = 0.928285

epoch 260 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.105195 + 0.935354 + 0.037923= 1.078472, l2reg=0.004531
epoch 270 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.228173 + 0.296767 + 0.009192= 0.534131, l2reg=0.004074
epoch 280 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.093264 + 0.414724 + 0.024614= 0.532603, l2reg=0.003792
epoch 290 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.256225 + 0.649590 + 0.030814= 0.936629, l2reg=0.003711
epoch 300 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.087962 + 0.662804 + 0.019474= 0.770240, l2reg=0.003477
mean acc = 0.928586

epoch 310 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.137499 + 0.758331 + 0.031880= 0.927710, l2reg=0.003238
epoch 320 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.077103 + 0.344939 + 0.013349= 0.435391, l2reg=0.003182
epoch 330 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.068417 + 0.571560 + 0.035767= 0.675743, l2reg=0.002974
epoch 340 (rho=16.000000, lamb=0.062500, eps=0.062500): h=0.269449 + 0.904595 + 0.033096= 1.207139, l2reg=0.002898
epoch 350 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.360408 + 0.513373 + 0.007073= 0.880853, l2reg=0.003339
mean acc = 0.929788

epoch 360 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.127421 + 0.925660 + 0.007779= 1.060860, l2reg=0.003830
epoch 370 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.113543 + 0.782660 + 0.012334= 0.908537, l2reg=0.004202
epoch 380 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.122634 + 0.428285 + 0.003720= 0.554639, l2reg=0.004731
epoch 390 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.160353 + 0.566571 + 0.003398= 0.730321, l2reg=0.004970
epoch 400 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.179310 + 0.841878 + 0.013875= 1.035063, l2reg=0.005448
mean acc = 0.929187

epoch 410 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.125757 + 0.550552 + 0.005050= 0.681359, l2reg=0.005704
epoch 420 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.263610 + 0.744666 + 0.005440= 1.013716, l2reg=0.006175
epoch 430 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.127623 + 0.710971 + 0.013111= 0.851705, l2reg=0.006267
epoch 440 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.154885 + 1.035450 + 0.010759= 1.201094, l2reg=0.006542
epoch 450 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.200827 + 0.965545 + 0.005126= 1.171498, l2reg=0.007149
mean acc = 0.929888

epoch 460 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.131448 + 0.788588 + 0.011281= 0.931318, l2reg=0.007228
epoch 470 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.152272 + 0.620357 + 0.006471= 0.779100, l2reg=0.007436
epoch 480 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.139532 + 1.751283 + 0.024983= 1.915798, l2reg=0.007783
epoch 490 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.296159 + 1.188512 + 0.013452= 1.498123, l2reg=0.007627
epoch 500 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.148977 + 0.426884 + 0.008815= 0.584676, l2reg=0.007961
mean acc = 0.928886

epoch 510 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.209239 + 1.308774 + 0.009591= 1.527605, l2reg=0.008229
epoch 520 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.087575 + 0.813361 + 0.013731= 0.914667, l2reg=0.008554
epoch 530 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.087518 + 0.597601 + 0.003010= 0.688130, l2reg=0.008609
epoch 540 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.131401 + 0.656260 + 0.008514= 0.796175, l2reg=0.008795
epoch 550 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.135458 + 0.574050 + 0.008956= 0.718464, l2reg=0.008729
mean acc = 0.929287

epoch 560 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.213556 + 0.666861 + 0.013971= 0.894389, l2reg=0.008948
epoch 570 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.060625 + 0.215907 + 0.002679= 0.279210, l2reg=0.009054
epoch 580 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.171115 + 0.871447 + 0.004992= 1.047554, l2reg=0.009203
epoch 590 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.170746 + 0.964186 + 0.011830= 1.146762, l2reg=0.009294
epoch 600 (rho=32.000000, lamb=0.031250, eps=0.031250): h=0.197268 + 1.085001 + 0.009156= 1.291424, l2reg=0.009266
mean acc = 0.928586

'''
