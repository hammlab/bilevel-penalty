## bilevel_synthetic.py

import os
import numpy as np
import tensorflow as tf


#######################################################################################################################
eps = 1E-10

def uv2xul(lb,ub,u1,u2,v1,v2):
    xu1 = u1#(ub[0]-lb[0])*.5*(tf.tanh(u1)+1.)+lb[0]
    xu2 = u2#(ub[1]-lb[1])*.5*(tf.tanh(u2)+1.)+lb[1]
    xl1 = v1#(ub[2]-lb[2])*.5*(tf.tanh(v1)+1.)+lb[2]
    xl2 = v2#(ub[3]-lb[3])*.5*(tf.tanh(v2)+1.)+lb[3]

    return xu1,xu2,xl1,xl2

   

#######################################################################################################################

## f = u^2+v^2 - 1, g = (1-u-v)^2

def uv20(p,q,r,std):
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    #ulb,uub,vlb,vub = lb[0:2],ub[0:2],lb[2:4],ub[2:4]

    xu1_init,xu2_init,xl1_init,xl2_init = 0.5*np.ones((p)),0.5*np.ones((r)),0.5*np.ones((q)),0.5*np.ones((r))
    u1_init = xu1_init#np.arctanh(2.*(xu1_init-lb[0])/(ub[0]-lb[0])-1.)
    u2_init = xu2_init#np.arctanh(2.*(xu2_init-lb[1])/(ub[1]-lb[1])-1.)
    v1_init = xl1_init#np.arctanh(2.*(xl1_init-lb[2])/(ub[2]-lb[2])-1.)
    v2_init = xl2_init#np.arctanh(2.*(xl2_init-lb[3])/(ub[3]-lb[3])-1.)            

    # Random initialization
    if False:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_normal_initializer(mean=u1_init,stddev=std))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_normal_initializer(mean=u2_init,stddev=std))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_normal_initializer(mean=v1_init,stddev=std))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_normal_initializer(mean=v2_init,stddev=std))
    else:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_uniform_initializer(minval=u1_init+std*(lb[0]-u1_init),maxval=u1_init+std*(ub[0]-u1_init)),constraint=lambda t: tf.clip_by_value(t,lb[0],ub[0]))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_uniform_initializer(minval=u2_init+std*(lb[1]-u2_init),maxval=u2_init+std*(ub[1]-u2_init)),constraint=lambda t: tf.clip_by_value(t,lb[1],ub[1]))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_uniform_initializer(minval=v1_init+std*(lb[2]-v1_init),maxval=v1_init+std*(ub[2]-v1_init)),constraint=lambda t: tf.clip_by_value(t,lb[2],ub[2]))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_uniform_initializer(minval=v2_init+std*(lb[3]-v2_init),maxval=v2_init+std*(ub[3]-v2_init)),constraint=lambda t: tf.clip_by_value(t,lb[3],ub[3]))
    
    return [u1,u2,v1,v2,lb[0:2],ub[0:2],lb[2:4],ub[2:4]]



def f20(u1,u2,v1,v2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
   
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    f = tf.reduce_sum(xl1**2+xl2**2) + tf.reduce_sum(xu1**2+xu2**2) -1

    return f


def g20(u1,u2,v1,v2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    g = tf.reduce_sum((1.-xu1-xl1)**2) + tf.reduce_sum((1.-xu2-xl2)**2)

    return g


def err20(u1,u2,v1,v2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]

    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    du1 = tf.reduce_sum((xu1-.5*tf.ones_like(xu1))**2)
    du2 = tf.reduce_sum((xu2-.5*tf.ones_like(xu2))**2)
    dl1 = tf.reduce_sum((xl1-.5*tf.ones_like(xl1))**2)
    dl2 = tf.reduce_sum((xl2-.5*tf.ones_like(xl2))**2)

    return tf.sqrt(du1+du2+dl1+dl2)

#######################################################################################################################

## f = v^2 - (u-v)^2,  g = (u-v)^2. Not saddle. Sol: (0,0)


def uv21(p,q,r,std):
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    #ulb,uub,vlb,vub = lb[0:2],ub[0:2],lb[2:4],ub[2:4]

    xu1_init,xu2_init,xl1_init,xl2_init = 0.*np.ones((p)),0.*np.ones((r)),0.*np.ones((q)),0.*np.ones((r))
    u1_init = xu1_init#np.arctanh(2.*(xu1_init-lb[0])/(ub[0]-lb[0])-1.)
    u2_init = xu2_init#np.arctanh(2.*(xu2_init-lb[1])/(ub[1]-lb[1])-1.)
    v1_init = xl1_init#np.arctanh(2.*(xl1_init-lb[2])/(ub[2]-lb[2])-1.)
    v2_init = xl2_init#np.arctanh(2.*(xl2_init-lb[3])/(ub[3]-lb[3])-1.)            

    # Random initialization
    if False:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_normal_initializer(mean=u1_init,stddev=std))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_normal_initializer(mean=u2_init,stddev=std))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_normal_initializer(mean=v1_init,stddev=std))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_normal_initializer(mean=v2_init,stddev=std))
    else:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_uniform_initializer(minval=u1_init+std*(lb[0]-u1_init),maxval=u1_init+std*(ub[0]-u1_init)),constraint=lambda t: tf.clip_by_value(t,lb[0],ub[0]))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_uniform_initializer(minval=u2_init+std*(lb[1]-u2_init),maxval=u2_init+std*(ub[1]-u2_init)),constraint=lambda t: tf.clip_by_value(t,lb[1],ub[1]))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_uniform_initializer(minval=v1_init+std*(lb[2]-v1_init),maxval=v1_init+std*(ub[2]-v1_init)),constraint=lambda t: tf.clip_by_value(t,lb[2],ub[2]))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_uniform_initializer(minval=v2_init+std*(lb[3]-v2_init),maxval=v2_init+std*(ub[3]-v2_init)),constraint=lambda t: tf.clip_by_value(t,lb[3],ub[3]))
    
    return [u1,u2,v1,v2,lb[0:2],ub[0:2],lb[2:4],ub[2:4]]


def f21(u1,u2,v1,v2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
   
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    f = 1*tf.reduce_sum(xl1**2+xl2**2) - (tf.reduce_sum((xu1-xl1)**2) + tf.reduce_sum((xu2-xl2)**2))

    return f


def g21(u1,u2,v1,v2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    g = tf.reduce_sum((xu1-xl1)**2) + 1*tf.reduce_sum((xu2-xl2)**2)

    return g


def err21(u1,u2,v1,v2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]

    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    du1 = tf.reduce_sum((xu1-tf.zeros_like(xu1))**2)
    du2 = tf.reduce_sum((xu2-tf.zeros_like(xu2))**2)
    dl1 = tf.reduce_sum((xl1-tf.zeros_like(xl1))**2)
    dl2 = tf.reduce_sum((xl2-tf.zeros_like(xl2))**2)

    return tf.sqrt(du1+du2+dl1+dl2)


#######################################################################################################################
## smd22?  f = v^T*A^T*A*v - (u-v)^T*A^TA*(u-v),  g=(u-v)^T*A^TA*(u-v)
# sol: vs = u + q, us = p: (p,q), p,q in Null(A)

def uv22(p,q,r,std):
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    #ulb,uub,vlb,vub = lb[0:2],ub[0:2],lb[2:4],ub[2:4]

    xu1_init,xu2_init,xl1_init,xl2_init = 0.*np.ones((p)),0.*np.ones((r)),0.*np.ones((q)),0.*np.ones((r))
    u1_init = xu1_init#np.arctanh(2.*(xu1_init-lb[0])/(ub[0]-lb[0])-1.)
    u2_init = xu2_init#np.arctanh(2.*(xu2_init-lb[1])/(ub[1]-lb[1])-1.)
    v1_init = xl1_init#np.arctanh(2.*(xl1_init-lb[2])/(ub[2]-lb[2])-1.)
    v2_init = xl2_init#np.arctanh(2.*(xl2_init-lb[3])/(ub[3]-lb[3])-1.)            

    # Random initialization
    if False:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_normal_initializer(mean=u1_init,stddev=std))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_normal_initializer(mean=u2_init,stddev=std))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_normal_initializer(mean=v1_init,stddev=std))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_normal_initializer(mean=v2_init,stddev=std))
    else:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_uniform_initializer(minval=u1_init+std*(lb[0]-u1_init),maxval=u1_init+std*(ub[0]-u1_init)),constraint=lambda t: tf.clip_by_value(t,lb[0],ub[0]))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_uniform_initializer(minval=u2_init+std*(lb[1]-u2_init),maxval=u2_init+std*(ub[1]-u2_init)),constraint=lambda t: tf.clip_by_value(t,lb[1],ub[1]))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_uniform_initializer(minval=v1_init+std*(lb[2]-v1_init),maxval=v1_init+std*(ub[2]-v1_init)),constraint=lambda t: tf.clip_by_value(t,lb[2],ub[2]))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_uniform_initializer(minval=v2_init+std*(lb[3]-v2_init),maxval=v2_init+std*(ub[3]-v2_init)),constraint=lambda t: tf.clip_by_value(t,lb[3],ub[3]))
    
    return [u1,u2,v1,v2,lb[0:2],ub[0:2],lb[2:4],ub[2:4]]



def f22(u1,u2,v1,v2,A1,A2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
   
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    #A1 = tf.random.normal((p,np.int(0.5*p)))
    #A2 = tf.random.normal((r,np.int(0.5*r)))
    # x^TA^TA*x = ||Ax||_F^2
    #f = 1.*tf.reduce_sum(xl1**2+xl2**2) - (tf.reduce_sum(tf.matmul(A1,tf.reshape(xu1-xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xu2-xl2,[-1,1]))**2))
    f = tf.reduce_sum(tf.matmul(A1,tf.reshape(xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xl2,[-1,1]))**2) - \
         (tf.reduce_sum(tf.matmul(A1,tf.reshape(xu1-xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xu2-xl2,[-1,1]))**2))

    return f


def g22(u1,u2,v1,v2,A1,A2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    g = (tf.reduce_sum(tf.matmul(A1,tf.reshape(xu1-xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xu2-xl2,[-1,1]))**2))

    return g


def err22(u1,u2,v1,v2,P1,P2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]

    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    #du1 = tf.reduce_sum((xu1-0.*tf.ones_like(xu1))**2)
    #du2 = tf.reduce_sum((xu2-0.*tf.ones_like(xu2))**2)
    #dl1 = tf.reduce_sum((xl1-0.*tf.ones_like(xl1))**2)
    #dl2 = tf.reduce_sum((xl2-0.*tf.ones_like(xl2))**2)
    du1 = tf.reduce_sum(tf.matmul(P1,tf.reshape(xu1,[-1,1]))**2)
    du2 = tf.reduce_sum(tf.matmul(P2,tf.reshape(xu2,[-1,1]))**2)
    dl1 = tf.reduce_sum(tf.matmul(P1,tf.reshape(xl1,[-1,1]))**2)
    dl2 = tf.reduce_sum(tf.matmul(P2,tf.reshape(xl2,[-1,1]))**2)

    return tf.sqrt(du1+du2+dl1+dl2)

#######################################################################################################################
## smd23  f = v^Tv - (u-v)^T*A^TA*(u-v),  g=(u-v)^T*A^TA*(u-v)
# sol: vs = us + q, us = -q. (-q,0). f(-q,0) = 0. 

def uv23(p,q,r,std):
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    #ulb,uub,vlb,vub = lb[0:2],ub[0:2],lb[2:4],ub[2:4]

    xu1_init,xu2_init,xl1_init,xl2_init = 0.*np.ones((p)),0.*np.ones((r)),0.*np.ones((q)),0.*np.ones((r))
    u1_init = xu1_init#np.arctanh(2.*(xu1_init-lb[0])/(ub[0]-lb[0])-1.)
    u2_init = xu2_init#np.arctanh(2.*(xu2_init-lb[1])/(ub[1]-lb[1])-1.)
    v1_init = xl1_init#np.arctanh(2.*(xl1_init-lb[2])/(ub[2]-lb[2])-1.)
    v2_init = xl2_init#np.arctanh(2.*(xl2_init-lb[3])/(ub[3]-lb[3])-1.)            

    # Random initialization
    if False:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_normal_initializer(mean=u1_init,stddev=std))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_normal_initializer(mean=u2_init,stddev=std))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_normal_initializer(mean=v1_init,stddev=std))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_normal_initializer(mean=v2_init,stddev=std))
    else:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_uniform_initializer(minval=u1_init+std*(lb[0]-u1_init),maxval=u1_init+std*(ub[0]-u1_init)),constraint=lambda t: tf.clip_by_value(t,lb[0],ub[0]))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_uniform_initializer(minval=u2_init+std*(lb[1]-u2_init),maxval=u2_init+std*(ub[1]-u2_init)),constraint=lambda t: tf.clip_by_value(t,lb[1],ub[1]))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_uniform_initializer(minval=v1_init+std*(lb[2]-v1_init),maxval=v1_init+std*(ub[2]-v1_init)),constraint=lambda t: tf.clip_by_value(t,lb[2],ub[2]))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_uniform_initializer(minval=v2_init+std*(lb[3]-v2_init),maxval=v2_init+std*(ub[3]-v2_init)),constraint=lambda t: tf.clip_by_value(t,lb[3],ub[3]))
    
    return [u1,u2,v1,v2,lb[0:2],ub[0:2],lb[2:4],ub[2:4]]



def f23(u1,u2,v1,v2,A1,A2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
   
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    #A1 = tf.random.normal((p,np.int(0.5*p)))
    #A2 = tf.random.normal((r,np.int(0.5*r)))
    # x^TA^TA*x = ||Ax||_F^2
    #f = 1.*tf.reduce_sum(xl1**2+xl2**2) - (tf.reduce_sum(tf.matmul(A1,tf.reshape(xu1-xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xu2-xl2,[-1,1]))**2))
    #f = tf.reduce_sum(tf.matmul(A1,tf.reshape(xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xl2,[-1,1]))**2) - \
    #     (tf.reduce_sum(tf.matmul(A1,tf.reshape(xu1-xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xu2-xl2,[-1,1]))**2))

    f = tf.reduce_sum(xl1**2+xl2**2) - (tf.reduce_sum(tf.matmul(A1,tf.reshape(xu1-xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xu2-xl2,[-1,1]))**2))
    return f


def g23(u1,u2,v1,v2,A1,A2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    g = (tf.reduce_sum(tf.matmul(A1,tf.reshape(xu1-xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xu2-xl2,[-1,1]))**2))

    return g


def err23(u1,u2,v1,v2,P1,P2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]

    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    #du1 = tf.reduce_sum((xu1-0.*tf.ones_like(xu1))**2)
    #du2 = tf.reduce_sum((xu2-0.*tf.ones_like(xu2))**2)
    dl1 = tf.reduce_sum((xl1-0.*tf.ones_like(xl1))**2)
    dl2 = tf.reduce_sum((xl2-0.*tf.ones_like(xl2))**2)
    du1 = tf.reduce_sum(tf.matmul(P1,tf.reshape(xu1,[-1,1]))**2)
    du2 = tf.reduce_sum(tf.matmul(P2,tf.reshape(xu2,[-1,1]))**2)
    #dl1 = tf.reduce_sum(tf.matmul(P1,tf.reshape(xl1,[-1,1]))**2)
    #dl2 = tf.reduce_sum(tf.matmul(P2,tf.reshape(xl2,[-1,1]))**2)

    return tf.sqrt(du1+du2+dl1+dl2)


#######################################################################################################################
## smd24  f = u^2 + v^2,  g=(1-u-v)^T*A^TA*(1-u-v)
# sol: us=1/2+1/2*p, vs=1-us+p = 1/2+1/2*p=us. 
# If min_{u,v} = ...?

def uv24(p,q,r,std):
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    #ulb,uub,vlb,vub = lb[0:2],ub[0:2],lb[2:4],ub[2:4]

    xu1_init,xu2_init,xl1_init,xl2_init = 0.5*np.ones((p)),0.5*np.ones((r)),0.5*np.ones((q)),0.5*np.ones((r))
    u1_init = xu1_init#np.arctanh(2.*(xu1_init-lb[0])/(ub[0]-lb[0])-1.)
    u2_init = xu2_init#np.arctanh(2.*(xu2_init-lb[1])/(ub[1]-lb[1])-1.)
    v1_init = xl1_init#np.arctanh(2.*(xl1_init-lb[2])/(ub[2]-lb[2])-1.)
    v2_init = xl2_init#np.arctanh(2.*(xl2_init-lb[3])/(ub[3]-lb[3])-1.)            

    # Random initialization
    if False:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_normal_initializer(mean=u1_init,stddev=std))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_normal_initializer(mean=u2_init,stddev=std))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_normal_initializer(mean=v1_init,stddev=std))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_normal_initializer(mean=v2_init,stddev=std))
    else:
        u1 = tf.get_variable('u1',shape=(p),initializer=tf.random_uniform_initializer(minval=u1_init+std*(lb[0]-u1_init),maxval=u1_init+std*(ub[0]-u1_init)),constraint=lambda t: tf.clip_by_value(t,lb[0],ub[0]))
        u2 = tf.get_variable('u2',shape=(r),initializer=tf.random_uniform_initializer(minval=u2_init+std*(lb[1]-u2_init),maxval=u2_init+std*(ub[1]-u2_init)),constraint=lambda t: tf.clip_by_value(t,lb[1],ub[1]))
        v1 = tf.get_variable('v1',shape=(q),initializer=tf.random_uniform_initializer(minval=v1_init+std*(lb[2]-v1_init),maxval=v1_init+std*(ub[2]-v1_init)),constraint=lambda t: tf.clip_by_value(t,lb[2],ub[2]))
        v2 = tf.get_variable('v2',shape=(r),initializer=tf.random_uniform_initializer(minval=v2_init+std*(lb[3]-v2_init),maxval=v2_init+std*(ub[3]-v2_init)),constraint=lambda t: tf.clip_by_value(t,lb[3],ub[3]))
    
    return [u1,u2,v1,v2,lb[0:2],ub[0:2],lb[2:4],ub[2:4]]

## smd24  f = u^2 + v^2,  g=(1-u-v)^T*A^TA*(1-u-v)
# sol: us=1/2+1/2*p, vs=1-us+p = 1/2+1/2*p=us


def f24(u1,u2,v1,v2,A1,A2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
   
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    #A1 = tf.random.normal((p,np.int(0.5*p)))
    #A2 = tf.random.normal((r,np.int(0.5*r)))
    # x^TA^TA*x = ||Ax||_F^2
    #f = 1.*tf.reduce_sum(xl1**2+xl2**2) - (tf.reduce_sum(tf.matmul(A1,tf.reshape(xu1-xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(xu2-xl2,[-1,1]))**2))
    f = tf.reduce_sum(xl1**2)+tf.reduce_sum(xl2**2)+tf.reduce_sum(xu1**2)+tf.reduce_sum(xu2**2)

    return f


def g24(u1,u2,v1,v2,A1,A2):
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]
    
    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    g = (tf.reduce_sum(tf.matmul(A1,tf.reshape(1.-xu1-xl1,[-1,1]))**2) + tf.reduce_sum(tf.matmul(A2,tf.reshape(1.-xu2-xl2,[-1,1]))**2))

    return g


def err24(u1,u2,v1,v2,P1,P2):
    #P = I-A'*inv(AA')*A
    p,q,r = u1.shape,v1.shape,u2.shape
    lb = [(-5.+eps)*np.ones(p),(-5.+eps)*np.ones(r),(-5.+eps)*np.ones(q),(-5.+eps)*np.ones(r)] 
    ub = [(5.-eps)*np.ones(p),(5.-eps)*np.ones(r),(5.-eps)*np.ones(q),(5.-eps)*np.ones(r)]
    #lb = [(-1.+eps)*np.ones(p),(-1.+eps)*np.ones(r),(-1.+eps)*np.ones(q),(-1.+eps)*np.ones(r)] 
    #ub = [(1.-eps)*np.ones(p),(1.-eps)*np.ones(r),(1.-eps)*np.ones(q),(1.-eps)*np.ones(r)]

    xu1,xu2,xl1,xl2 = uv2xul(lb,ub,u1,u2,v1,v2)
    
    #du1 = tf.reduce_sum((xu1-0.*tf.ones_like(xu1))**2)
    #du2 = tf.reduce_sum((xu2-0.*tf.ones_like(xu2))**2)
    #dl1 = tf.reduce_sum((xl1-0.*tf.ones_like(xl1))**2)
    #dl2 = tf.reduce_sum((xl2-0.*tf.ones_like(xl2))**2)
    du1 = tf.reduce_sum(tf.matmul(P1,tf.reshape(xu1-.5,[-1,1]))**2)
    du2 = tf.reduce_sum(tf.matmul(P2,tf.reshape(xu2-.5,[-1,1]))**2)
    dl1 = tf.reduce_sum(tf.matmul(P1,tf.reshape(xl1-.5,[-1,1]))**2)
    dl2 = tf.reduce_sum(tf.matmul(P2,tf.reshape(xl2-.5,[-1,1]))**2)

    return tf.sqrt(du1+du2+dl1+dl2)


