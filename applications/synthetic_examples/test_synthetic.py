from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import sys
sys.path.append("../../../../../optimizers/synthetic_examples")
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.backends.backend_pdf

import tensorflow as tf
from tensorflow.python.platform import flags


#from cleverhans.utils_mnist import data_mnist
#import keras
#from keras.datasets import cifar10, cifar100
#from keras.datasets import mnist
import time

import bilevel_penalty_augm_lag
import bilevel_approxgrad
import bilevel_rmd
import bilevel_gd
#import bilevel_invhess

import bilevel_synthetic

#############################################################################################################

lr_u = 1E-3
lr_v = 1E-4
lr_p = 1E-4

nepoch = 40001
nstep = 10

rho_0 = 1E0
lamb_0 = 1E1
eps_0 = 1E0

c_rho = 1.1
#c_lamb = .9
#c_eps = .9

p,q,r = 10,10,10
ntrial = 20
Ts = [1,5,10]#,20,30]
#Ts = [10]
#Ts = [1]
std = 1

examples = ['SMD20','SMD21','SMD24','SMD23']
#examples = ['SMD20']
#examples = ['SMD24','SMD23']
#methods = ['ApproxGrad']#['GD','RMD','ApproxGrad','Penalty']
#methods=  ['RMD']
methods = ['ApproxGrad','RMD']

nTs = len(Ts)
nexamples = len(examples)
nmethods = len(methods)
nrec = len(np.arange(0,nepoch,nstep))


def main(argv=None):

    tf.set_random_seed(1234)

    for m,method in enumerate(methods):
        for e,ex in enumerate(examples):
            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            if ex=='SMD20':
                u1,u2,v1,v2,ulb,uub,vlb,vub = bilevel_synthetic.uv20(p,q,r,std)
                f = bilevel_synthetic.f20(u1,u2,v1,v2)
                g = bilevel_synthetic.g20(u1,u2,v1,v2)
                err = bilevel_synthetic.err20(u1,u2,v1,v2)
            elif ex=='SMD21':
                u1,u2,v1,v2,ulb,uub,vlb,vub = bilevel_synthetic.uv21(p,q,r,std)
                f = bilevel_synthetic.f21(u1,u2,v1,v2)
                g = bilevel_synthetic.g21(u1,u2,v1,v2)
                err = bilevel_synthetic.err21(u1,u2,v1,v2)
            elif ex=='SMD22':
                p_ = np.int(np.ceil(p*0.5))
                A1 = np.float32(np.random.normal(size=(p_,p)))
                A2 = np.float32(np.random.normal(size=(p_,p)))
                # Err = P*u -> Orth proj onto row-space: P = A'*inv(AA')*A 
                P1 = np.matmul(A1.T,np.matmul(np.linalg.inv(np.matmul(A1,A1.T)),A1))
                P2 = np.matmul(A2.T,np.matmul(np.linalg.inv(np.matmul(A2,A2.T)),A2))
                u1,u2,v1,v2,ulb,uub,vlb,vub = bilevel_synthetic.uv22(p,q,r,std)
                f = bilevel_synthetic.f22(u1,u2,v1,v2,A1,A2)
                g = bilevel_synthetic.g22(u1,u2,v1,v2,A1,A2)
                err = bilevel_synthetic.err22(u1,u2,v1,v2,P1,P2)
            elif ex=='SMD22full':
                p_ = p#np.int(np.ceil(p*0.5))
                A1 = np.float32(np.random.normal(size=(p_,p)))
                A2 = np.float32(np.random.normal(size=(p_,p)))
                u1,u2,v1,v2,ulb,uub,vlb,vub = bilevel_synthetic.uv22(p,q,r,std)
                f = bilevel_synthetic.f22(u1,u2,v1,v2,A1,A2)
                g = bilevel_synthetic.g22(u1,u2,v1,v2,A1,A2)
                err = bilevel_synthetic.err22(u1,u2,v1,v2,A1,A2)
            elif ex=='SMD23':
                p_ = np.int(np.ceil(p*0.5))
                A1 = np.float32(np.random.normal(size=(p_,p)))
                A2 = np.float32(np.random.normal(size=(p_,p)))
                # Err = P*u -> Orth proj onto row-space: P = A'*inv(AA')*A 
                P1 = np.matmul(A1.T,np.matmul(np.linalg.inv(np.matmul(A1,A1.T)),A1))
                P2 = np.matmul(A2.T,np.matmul(np.linalg.inv(np.matmul(A2,A2.T)),A2))
                u1,u2,v1,v2,ulb,uub,vlb,vub = bilevel_synthetic.uv23(p,q,r,std)
                f = bilevel_synthetic.f23(u1,u2,v1,v2,A1,A2)
                g = bilevel_synthetic.g23(u1,u2,v1,v2,A1,A2)
                err = bilevel_synthetic.err23(u1,u2,v1,v2,P1,P2)
            elif ex=='SMD24':
                p_ = np.int(np.ceil(p*0.5))
                A1 = np.float32(np.random.normal(size=(p_,p)))
                A2 = np.float32(np.random.normal(size=(p_,p)))
                P1 = np.matmul(A1.T,np.matmul(np.linalg.inv(np.matmul(A1,A1.T)),A1))
                P2 = np.matmul(A2.T,np.matmul(np.linalg.inv(np.matmul(A2,A2.T)),A2))
                u1,u2,v1,v2,ulb,uub,vlb,vub = bilevel_synthetic.uv24(p,q,r,std)
                f = bilevel_synthetic.f24(u1,u2,v1,v2,A1,A2)
                g = bilevel_synthetic.g24(u1,u2,v1,v2,A1,A2)
                err = bilevel_synthetic.err24(u1,u2,v1,v2,P1,P2)



            if method=='GD': ## gradient descent
                bl = bilevel_gd.bilevel_gd(sess,f,g,[u1,u2],[v1,v2],lr_u,lr_v,ulb,uub,vlb,vub)
                for n,niter in enumerate(Ts):
                    errs = np.nan*np.ones((nrec,ntrial))
                    fs = np.nan*np.ones((nrec,ntrial))
                    gs = np.nan*np.ones((nrec,ntrial))
                    ts = np.nan*np.ones(ntrial)
                    for trial in range(ntrial):
                        print('\n%s, ex=%s, n=%d/%d, trial=%d/%d'%(method,ex,n+1,nTs,trial+1,ntrial))
                        sess.run(tf.global_variables_initializer())
                        ## warm up
                        #for i in range(20): sess.run(bl.min_v)
                        cnt = 0
                        t0 = time.time()
                        for epoch in range(nepoch):
                            #tick = time.time()        
                            fval,gval = bl.update({},niter)
                            if epoch%nstep==0:
                                terr = sess.run(err)
                                errs[cnt,trial] = terr
                                fs[cnt,trial] = fval
                                gs[cnt,trial] = gval
                                cnt +=1
                                #if np.isnan(fval) or np.isnan(gval): return#.any()
                            if False:#epoch%1000==0:
                                u1val,u2val,v1val,v2val = sess.run([u1,u2,v1,v2])
                                print('epoch %d: f=%f, g=%f, err=%f'%(epoch,fval,gval,terr))
                                tu1,tu2,tv1,tv2 = sess.run([u1,u2,v1,v2])
                                print('u1=%f,u2=%f,l1=%f,l2=%f'%(tu1[0],tu2[0],tv1[0],tv2[0]))
                        ts[trial] = time.time()-t0
                        print('time=%f'%(ts[trial]))
                        print('epoch %d: f=%f, g=%f, err=%f'%(epoch,fval,gval,terr))
                        ##tu1,tu2,tv1,tv2 = sess.run([u1,u2,v1,v2])
                        ##print('u1=%f,u2=%f,l1=%f,l2=%f'%(tu1,tu2,tv1,tv2))
                    if True:
                        np.save('./results/simple_ts_%s_%s_T%d.npy'%(method,ex,niter),ts)
                        np.save('./results/simple_errs_%s_%s_T%d.npy'%(method,ex,niter),errs)
                        np.save('./results/simple_fs_%s_%s_T%d.npy'%(method,ex,niter),fs)
                        np.save('./results/simple_gs_%s_%s_T%d.npy'%(method,ex,niter),gs)

						
            elif method=='RMD': ## Reverse mode differentiation
                bl = bilevel_rmd.bilevel_rmd(sess,f,g,[u1,u2],[v1,v2],lr_u,lr_v,ulb,uub,vlb,vub)
                for n,niter in enumerate(Ts):
                    errs = np.nan*np.ones((nrec,ntrial))
                    fs = np.nan*np.ones((nrec,ntrial))
                    gs = np.nan*np.ones((nrec,ntrial))
                    ts = np.nan*np.ones(ntrial)
                    for trial in range(ntrial):
                        print('\n%s, ex=%s, n=%d/%d, trial=%d/%d'%(method,ex,n+1,nTs,trial+1,ntrial))
                        sess.run(tf.global_variables_initializer())
                        ## warm up
                        #for i in range(20): sess.run(bl.min_v)
                        cnt = 0
                        t0 = time.time()						
                        for epoch in range(nepoch):
                            fval,gval = bl.update({},niter)
                            if epoch%nstep==0:
                                terr = sess.run(err)
                                errs[cnt,trial] = terr
                                fs[cnt,trial] = fval
                                gs[cnt,trial] = gval
                                cnt +=1
                                #if np.isnan(fval) or np.isnan(gval): return#.any()                                    
                            if epoch%1000==0:
                                print('epoch %d: f=%f, g=%f, err=%f'%(epoch,fval,gval,terr))
                                tu1,tu2,tv1,tv2 = sess.run([u1,u2,v1,v2])
                                print('u1=%f,u2=%f,l1=%f,l2=%f'%(tu1[0],tu2[0],tv1[0],tv2[0]))
                        ts[trial] = time.time()-t0
                        print('time=%f'%(ts[trial]))
                        print('epoch %d: f=%f, g=%f, err=%f'%(epoch,fval,gval,terr))
                        ##tu1,tu2,tv1,tv2 = sess.run([u1,u2,v1,v2])
                        ##print('u1=%f,u2=%f,l1=%f,l2=%f'%(tu1,tu2,tv1,tv2))
                    
                    if True:
                        np.save('./results/simple_ts_%s_%s_T%d.npy'%(method,ex,niter),ts)
                        np.save('./results/simple_errs_%s_%s_T%d.npy'%(method,ex,niter),errs)
                        np.save('./results/simple_fs_%s_%s_T%d.npy'%(method,ex,niter),fs)
                        np.save('./results/simple_gs_%s_%s_T%d.npy'%(method,ex,niter),gs)

						
            elif method=='ApproxGrad': ## Approximate hypergradient
                bl = bilevel_approxgrad.bilevel_approxgrad(sess,f,g,[u1,u2],[v1,v2],lr_u,lr_v,lr_p,ulb,uub,vlb,vub)
                for n,niter in enumerate(Ts):
                    errs = np.nan*np.ones((nrec,ntrial))
                    fs = np.nan*np.ones((nrec,ntrial))
                    gs = np.nan*np.ones((nrec,ntrial))
                    ts = np.nan*np.ones(ntrial)
                    for trial in range(ntrial):
                        print('\n%s, ex=%s, n=%d/%d, trial=%d/%d'%(method,ex,n+1,nTs,trial+1,ntrial))
                        sess.run(tf.global_variables_initializer())
                        ## warm up
                        #for i in range(20): sess.run(bl.min_v)
                        cnt = 0
                        t0 = time.time()
                        for epoch in range(nepoch):
                            fval,gval,hval = bl.update({},niter,niter)
                            if epoch%nstep==0:
                                terr = sess.run(err)
                                errs[cnt,trial] = terr
                                fs[cnt,trial] = fval
                                gs[cnt,trial] = gval
                                cnt +=1
                                #if np.isnan(fval) or np.isnan(gval): return#.any()                                    
                            if epoch%1000==0:
                                print('epoch %d: f=%f, g=%f, h=%f, err=%f'%(epoch,fval,gval,hval,terr))
                                tu1,tu2,tv1,tv2 = sess.run([u1,u2,v1,v2])
                                print('u1=%f,u2=%f,l1=%f,l2=%f'%(tu1[0],tu2[0],tv1[0],tv2[0]))
                        ts[trial] = time.time()-t0
                        print('time=%f'%(ts[trial]))
                        print('epoch %d: f=%f, g=%f, h=%f, err=%f'%(epoch,fval,gval,hval,terr))
                        ##tu1,tu2,tv1,tv2 = sess.run([u1,u2,v1,v2])
                        ##print('u1=%f,u2=%f,l1=%f,l2=%f'%(tu1,tu2,tv1,tv2))
                    
                    if True:
                        np.save('./results/simple_ts_%s_%s_T%d.npy'%(method,ex,niter),ts)
                        np.save('./results/simple_errs_%s_%s_T%d.npy'%(method,ex,niter),errs)
                        np.save('./results/simple_fs_%s_%s_T%d.npy'%(method,ex,niter),fs)
                        np.save('./results/simple_gs_%s_%s_T%d.npy'%(method,ex,niter),gs)

                
            elif method=='Penalty': ## Penalty method
                bl = bilevel_penalty_augm_lag.bilevel_penalty(sess,f,g,[u1,u2],[v1,v2],lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,ulb,uub,vlb,vub)
                #bl = bilevel_penalty.bilevel_penalty(sess,f,g,[u1,u2],[v1,v2],lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)
                for n,niter in enumerate(Ts):
                    errs = np.nan*np.ones((nrec,ntrial))
                    rhos = np.nan*np.ones((nrec,ntrial))
                    fs = np.nan*np.ones((nrec,ntrial))
                    gs = np.nan*np.ones((nrec,ntrial))
                    ftilde = np.nan*np.ones((nrec,ntrial))
                    ts = np.nan*np.ones(ntrial)
                    for trial in range(ntrial):
                        print('\n%s, ex=%s, n=%d/%d, trial=%d/%d'%(method,ex,n+1,nTs,trial+1,ntrial))
                        sess.run(tf.global_variables_initializer())
						#bl.reset_optim()
                        ## warm up
                        #for i in range(20): sess.run(bl.min_v)
                        cnt = 0
                        t0 = time.time()
                        for epoch in range(nepoch):
                            fval,gvnorm,lamb_g = bl.update({},niter)
                            if epoch%nstep==0:
                                rho_t,lamb_t,eps_t = sess.run([bl.rho_t,bl.lamb_t,bl.eps_t])
                                gval,terr = sess.run([bl.f,err])
                                errs[cnt,trial] = terr
                                fs[cnt,trial] = fval
                                gs[cnt,trial] = gval
                                ftilde[cnt,trial] = fval+gvnorm+lamb_g
                                rhos[cnt,trial] = rho_t
                                cnt +=1
                                #if np.isnan(fval) or np.isnan(gval): return#.any()
                            if False:#epoch%1000==0:
                                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f, err=%f'%
                                    (epoch,rho_t,lamb_t,eps_t,fval,gvnorm,lamb_g,fval+gvnorm+lamb_g,terr))
                                tu1,tu2,tv1,tv2 = sess.run([u1,u2,v1,v2])
                                print('u1=%f,u2=%f,l1=%f,l2=%f'%(tu1[0],tu2[0],tv1[0],tv2[0]))
                        ts[trial] = time.time()-t0
                        print('time=%f'%(ts[trial]))
                        print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f, err=%f'%
                            (epoch,rho_t,lamb_t,eps_t,fval,gvnorm,lamb_g,fval+gvnorm+lamb_g,terr))
                        ##tu1,tu2,tv1,tv2 = sess.run([u1,u2,v1,v2])
                        ##print('u1=%f,u2=%f,l1=%f,l2=%f'%(tu1,tu2,tv1,tv2))
                    
                    if True:
                        np.save('./results/simple_ts_%s_%s_T%d.npy'%(method,ex,niter),ts)
                        np.save('./results/simple_errs_%s_%s_T%d.npy'%(method,ex,niter),errs)
                        np.save('./results/simple_fs_%s_%s_T%d.npy'%(method,ex,niter),fs)
                        np.save('./results/simple_gs_%s_%s_T%d.npy'%(method,ex,niter),gs)
                        np.save('./results/simple_ftilde_%s_%s_T%d.npy'%(method,ex,niter),ftilde)
                        np.save('./results/simple_rhos_%s_%s_T%d.npy'%(method,ex,niter),rhos)


            sess.close()

    return
    


##############################################################################################################

if __name__ == '__main__':

    tf.app.run()




'''
## Results


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import sys

import numpy as np
from six.moves import xrange

import matplotlib
matplotlib.use('Agg') # no UI backend

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.backends.backend_pdf

ntrial = 20
#Ts = [1,2,5,10]#,20,30]
Ts = [1,5,10]
stds = [1.]#[.99]
examples = ['SMD20','SMD21','SMD24','SMD23']
#examples = ['SMD23']
methods = ['GD','RMD','ApproxGrad','Penalty']

nTs = len(Ts)
nstds = len(stds)
nexamples = len(examples)
nmethods = len(methods)

nepoch = 40001
nstep = 10
rec = np.arange(0,nepoch,nstep)
nrec = len(rec)


errs = np.nan*np.ones((nrec,ntrial,nTs,nexamples,nmethods))
rhos = np.nan*np.ones((nrec,ntrial,nTs,nexamples,nmethods))
fs = np.nan*np.ones((nrec,ntrial,nTs,nexamples,nmethods))
gs = np.nan*np.ones((nrec,ntrial,nTs,nexamples,nmethods))
ftilde = np.nan*np.ones((nrec,ntrial,nTs,nexamples,nmethods))
ts = np.nan*np.ones((ntrial,nTs,nexamples,nmethods))

for m,method in enumerate(methods):
    for e,ex in enumerate(examples):
        for t,T in enumerate(Ts):
            errs[:,:,t,e,m] = np.load('./results/simple_errs_%s_%s_T%d.npy'%(method,ex,T)).squeeze()
            fs[:,:,t,e,m] = np.load('./results/simple_fs_%s_%s_T%d.npy'%(method,ex,T)).squeeze()
            gs[:,:,t,e,m] = np.load('./results/simple_gs_%s_%s_T%d.npy'%(method,ex,T)).squeeze()
            #ftilde[:,:,t,e,m] = np.load('./results/simple_ftilde_%s_%s_T%d.npy'%(method,ex,T)).squeeze()
            #rhos[:,:,t,e,m] =np.load('./results/simple_rhos_%s_%s_T%d.npy'%(method,ex,T)).squeeze()
            #ts[:,t,e,m] = np.load('./results/simple_ts_%s_%s_T%d.npy'%(method,ex,T)).squeeze()


print(methods)
for e,ex in enumerate(examples):
    print(ex)
    for t,T in enumerate(Ts):
        print(T)
        for m,method in enumerate(methods):
            print('%.2f pm %.2f '%(np.mean(ts[:,t,e,m].squeeze()),np.std(ts[:,t,e,m].squeeze())))
			



for t,T in enumerate(Ts):
    print(T)
    for e,ex in enumerate(examples):
        print(ex)
        for m,method in enumerate(methods):
            print('%d & '%(np.mean(ts[:,t,e,m].squeeze())))        
			
			

for e,ex in enumerate(examples):
    fig = plt.figure(e+1)
    plt.clf()
    for row in range(nTs):
        for col in range(nmethods):
            subplt = plt.subplot(nTs+1,nmethods,row*nmethods+col+1)
            plt.plot(rec,np.squeeze(errs[:,:,row,e,col]),'y-',alpha=.5)            
            plt.plot(rec,np.squeeze(errs[:,:,row,e,col].mean(axis=1)),'b-',linewidth=2.)
            #plt.axis(0.,np.float(len(rec)),0., 50.)
            plt.ylim(0.,40.)
            #plt.ylim(0.,30.)
            subplt.tick_params(axis='x', labelsize=8)
            subplt.tick_params(axis='y', labelsize=8)                                    
            if row==0:
                plt.title('%s'%(methods[col]),fontsize=12)
            if col==0:
                plt.ylabel('T=%d'%Ts[row],rotation='horizontal',horizontalalignment='right',fontsize=12)
            #if row==0: 
            #    plt.legend()
    plt.tight_layout(pad=0.25)
    #plt.show(block=False)
    
    with matplotlib.backends.backend_pdf.PdfPages('./results/bilevel_simple_%s.pdf'%(ex)) as pdf:
        pdf.savefig(bbox_inches='tight')






#################################################################

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.patches import FancyArrowPatch
#from mpl_toolkits.mplot3d import proj3d

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
#from matplotlib import colors as mcolors
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


if False:
    ## SMD20: f = u^2+v^2 - 1, g = (1-u-v)^2
    u,v = np.linspace(-5.,5.,100), np.linspace(-5.,5.,100)
    U,V = np.meshgrid(u,v)
    F = U**2 + V**2 - 1.
    G = (1.-U-V)**2
    vs = 1.-u
    Fs = u**2 + vs**2 - 1.
if True:
    ## SMD21: f = v^2 - (u-v)^2,  g = (u-v)^2. Not saddle. Sol: (0,0)
    u,v = np.linspace(-5.,5.,100), np.linspace(-5.,5.,100)
    U,V = np.meshgrid(u,v)
    F = V**2 - (U-V)**2
    G = (U-V)**2
    vs = u
    Fs = vs**2 - (u-vs)**2


fig = plt.figure(1)
#plt.clf()
ax = fig.gca(projection='3d')
if True:
    ax.plot_surface(U,V,F,cmap=cm.coolwarm,linewidth=0,edgecolors='none',antialiased=True)
    #ax.plot_surface(U,V,F,cmap=cm.coolwarm,rstride=5,cstride=5,edgecolors='none',antialiased=True)    
else:
    ax.plot_wireframe(U,V,F,rstride=5,cstride=5,linewidth=0.5,color='k')#gray')

ax.view_init(elev=45, azim=-45) #Reproduce view
ax.set_xlabel('U',fontsize=18)
ax.set_ylabel('V',fontsize=18)
#plt.legend(fontsize=18, loc='upper right')

plt.plot(u,vs,Fs,color='k')
#plt.plot(range(max_iter),gss[i],specs[i],label=labels[i],markersize=6,linewidth=2)

plt.show(block=False)

with matplotlib.backends.backend_pdf.PdfPages('SMD21.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')

            
'''

