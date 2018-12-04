import numpy as np
from numpy import array, asarray, float64, int32, zeros
from scipy import linalg
from scipy.optimize.lbfgsb import _lbfgsb
from scipy.sparse import linalg as splinalg
from sklearn import datasets, linear_model
import time
import os


def hoag_lbfgs(
    test_loss, h_func_grad, h_hessian, h_crossed, g_func_grad, x0, bounds=None,
    lambda0=0., disp=None, maxcor=10,
    maxiter=100, maxiter_inner=10000,
    only_fit=False,
    iprint=-1, maxls=20, tolerance_decrease='exponential',
    callback=None, verbose=0, epsilon_tol_init=1e-3, exponential_decrease_factor=0.9,
    projection=None):
    """
    HOAG algorithm using L-BFGS-B in the inner optimization algorithm.

    Options
    -------
    eps : float
        Step size used for numerical approximation of the Jacobian.
    disp : int
        Set to True to print convergence messages.
    maxfun : int
        Maximum number of function evaluations.
    maxiter : int
        Maximum number of iterations.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    """
    m = maxcor
    lambdak = lambda0
    if verbose > 0:
        print('started hoag')

    x0 = asarray(x0).ravel()
    n, = x0.shape

    if bounds is None:
        bounds = [(None, None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')
    # unbounded variables must use None, not +-inf, for optimizer to work properly
    bounds = [(None if l == -np.inf else l, None if u == np.inf else u) for l, u in bounds]

    if disp is not None:
        if disp == 0:
            iprint = -1
        else:
            iprint = disp

    nbd = zeros(n, int32)
    low_bnd = zeros(n, float64)
    upper_bnd = zeros(n, float64)
    bounds_map = {(None, None): 0,
                  (1, None): 1,
                  (1, 1): 2,
                  (None, 1): 3}
    for i in range(0, n):
        l, u = bounds[i]
        if l is not None:
            low_bnd[i] = l
            l = 1
        if u is not None:
            upper_bnd[i] = u
            u = 1
        nbd[i] = bounds_map[l, u]

    if not maxls > 0:
        raise ValueError('maxls must be positive.')

    x = array(x0, float64)
    wa = zeros(2*m*n + 5*n + 11*m*m + 8*m, float64)
    iwa = zeros(3*n, int32)
    task = zeros(1, 'S60')
    csave = zeros(1, 'S60')
    lsave = zeros(4, int32)
    isave = zeros(44, int32)
    dsave = zeros(29, float64)

    exact_epsilon = 1e-12
    if tolerance_decrease == 'exact':
        epsilon_tol = exact_epsilon
    else:
        epsilon_tol = epsilon_tol_init

    Bxk = None
    L_lambda = None
    g_func_old = np.inf

    if callback is not None:
        callback(x, lambdak)

    # n_eval, F = wrap_function(F, ())
    h_func, h_grad = h_func_grad(x, lambdak)
    norm_init = linalg.norm(h_grad)
    old_grads = []
    old_lambdak = lambdak.copy()

    print(os.listdir(os.getcwd()))
    loss_total = np.load("loss_hoag.npy")
    time_total = np.load("time_hoag.npy")
    cum_time = 0
    for it in range(1, maxiter):
        tick = time.time()
        h_func, h_grad = h_func_grad(x, lambdak)
        n_iterations = 0
        task[:] = 'START'
        old_x = x.copy()
        while 1:
            pgtol_lbfgs = 1e-120
            factr = 1e-120  # / np.finfo(float).eps
            _lbfgsb.setulb(
                m, x, low_bnd, upper_bnd, nbd, h_func, h_grad,
                factr, pgtol_lbfgs, wa, iwa, task, iprint, csave, lsave,
                isave, dsave, maxls)
            task_str = task.tostring()
            if task_str.startswith(b'FG'):
                # minimization routine wants h_func and h_grad at the current x
                # Overwrite h_func and h_grad:
                h_func, h_grad = h_func_grad(x, lambdak)
                if linalg.norm(h_grad)  < \
                    epsilon_tol * norm_init * np.exp(np.min(old_lambdak) - np.min(lambda0)):
                    # this one is finished
                    break

            elif task_str.startswith(b'NEW_X'):
                # new iteration
                if n_iterations > maxiter_inner:
                    task[:] = 'STOP: TOTAL NO. of ITERATIONS EXCEEDS LIMIT'
                    print('ITERATIONS EXCEEDS LIMIT')
                    continue
                    # break
                else:
                    n_iterations += 1
            else:
                if verbose > 1:
                    print('LBFGS decided finish!')
                    print(task_str)
                break
        else:
            pass

        if only_fit:
            break

        if verbose > 0:
            h_func, h_grad = h_func_grad(x, lambdak)
            #print('inner level iterations: %s, inner objective %s, grad norm %s' % (n_iterations, h_func, linalg.norm(h_grad)))
            
            

        fhs = h_hessian(x, lambdak)
        B_op = splinalg.LinearOperator(
            shape=(x.size, x.size),
            matvec=lambda z: fhs(z))

        g_func, g_grad = g_func_grad(x, lambdak)
        if Bxk is None:
            Bxk = x.copy()
        tol_CG = epsilon_tol
        if verbose > 1:
            print('Inverting matrix with precision %s' % tol_CG)
        Bxk, success = splinalg.cg(B_op, g_grad, x0=Bxk, tol=tol_CG, maxiter=maxiter_inner)
        if success != 0:
            print('CG did not converge to the desired precision')
        old_epsilon_tol = epsilon_tol
        if tolerance_decrease == 'quadratic':
            epsilon_tol = epsilon_tol_init / (it ** 2)
        elif tolerance_decrease == 'cubic':
            epsilon_tol = epsilon_tol_init / (it ** 3)
        elif tolerance_decrease == 'exponential':
            epsilon_tol *= exponential_decrease_factor
        elif tolerance_decrease == 'exact':
            epsilon_tol = 1e-24
        else:
            raise NotImplementedError

        epsilon_tol = max(epsilon_tol, exact_epsilon)
        # .. update hyperparameters ..
        grad_lambda = - h_crossed(x, lambdak).dot(Bxk)
        if linalg.norm(grad_lambda) == 0:
            # increase tolerance
            if verbose > 0:
                print('too low tolerance %s, moving to next iteration' % epsilon_tol)
            continue
        old_grads.append(linalg.norm(grad_lambda))

        if L_lambda is None:
            if old_grads[-1] > 1e-3:
                # make sure we are not selecting a step size that is too smal
                L_lambda = old_grads[-1] / np.sqrt(len(lambdak))
            else:
                L_lambda = 1

        step_size = (1./L_lambda)

        old_lambdak = lambdak.copy()
        lambdak -= step_size * grad_lambda

        # projection
        lambdak[lambdak < -6] = -6
        lambdak[lambdak > 6] = 6
        incr = linalg.norm(step_size * grad_lambda)

        C = 0.25
        factor_L_lambda = 1.0
        if g_func <= g_func_old + C * epsilon_tol + \
                old_epsilon_tol * (C + factor_L_lambda) * incr - factor_L_lambda * (L_lambda) * incr * incr:
            L_lambda *= 0.95
            if verbose > 1:
                print('increased step size')
            lambdak -= step_size * grad_lambda
        elif g_func >= 1.2 * g_func_old:
            if verbose > 1:
                print('decrease step size')
            # decrease step size
            L_lambda *= 2
            lambdak = old_lambdak.copy()
            print('!!step size rejected!!', g_func, g_func_old)
            g_func_old, g_grad_old = g_func_grad(x, old_lambdak)
            # tighten tolerance
            epsilon_tol *= 0.5
        else:
            old_lambdak = lambdak.copy()
            lambdak -= step_size * grad_lambda

        # projection
        if projection is None:
            pass
        else:
            lambdak = projection(lambdak)


        # if g_func - g_func_old > 0:
        #     raise ValueError
        norm_grad_lambda = linalg.norm(grad_lambda)
        if verbose == -1:
            #print(('it %s, g: %s, incr: %s, sum lambda %s, epsilon: %s, ' +
            #      'L: %s, norm grad_lambda: %s') %
            #      (it, g_func, g_func - g_func_old, lambdak.sum(), epsilon_tol, L_lambda,
            #       norm_grad_lambda))
            st_time = time.time() - tick
            ##############
            t_loss = test_loss(x, lambdak)
            print st_time, ",", t_loss, ",", g_func, ",", lambdak.sum()
            cum_time += st_time
            loss_total[it] += t_loss
            time_total[it] += cum_time
            '''
            from sklearn.datasets import fetch_20newsgroups
            from sklearn.feature_extraction.text import TfidfVectorizer
            from scipy.sparse import vstack
            categories = None
            #remove = ('headers', 'footers', 'quotes')
            
            data_train_f = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)#, remove=remove)
            data_test_f = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)#, remove=remove)
            #print('data loaded')
            
            target_names_f = data_train_f.target_names
            y_train_f, y_test_f = data_train_f.target, data_test_f.target
            vectorizer_f = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
            X_train_f = vectorizer_f.fit_transform(data_train_f.data)
            #####
            #X_train = X_train.todense()
            #print("n_samples: %d, n_features: %d" % X_train_f.shape)
            X_test_f = vectorizer_f.transform(data_test_f.data)
            #####
            #X_test = X_test.todense()
            #print("n_samples: %d, n_features: %d" % X_test_f.shape)
            
            #####
            X_all_f = vstack([X_train_f, X_test_f])
            #X_all = np.concatenate(([X_train, X_test]), axis = 0)
            
            # binarize labels
            y_train_f[data_train_f.target < 10] = -1
            y_train_f[data_train_f.target >= 10] = 1
            y_test_f[data_test_f.target < 10] = -1
            y_test_f[data_test_f.target >= 10] = 1
              
            Y_all_f = np.concatenate([y_train_f, y_test_f], axis = 0)
            #####
            #Y_all = Y_all.reshape(18846, 1)
            shuff_idx_f = np.load("20news_shuff_idx.npy")
            X_all_f = X_all_f[shuff_idx_f]
            Y_all_f = Y_all_f[shuff_idx_f]
            
            val_f = int(X_all_f.shape[0]/3)     
             
            X_train_f = X_all_f[:val_f]
            y_train_f = Y_all_f[:val_f]
                  
            X_val_f = X_all_f[val_f: 2*val_f]
            y_val_f = Y_all_f[val_f: 2*val_f]
            
            X_test_f = X_all_f[2*val_f: ]
            y_test_f = Y_all_f[2*val_f: ]
            
      
            clf = linear_model.LogisticRegression(solver='lbfgs', C=np.exp(-lambdak.sum()), fit_intercept=False, tol=1e-22, max_iter=500)
            clf.fit(X_train_f, y_train_f)
            cost_1 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_val_f, y_val_f, 0.)
            #print "val Cost ", cost
            
            cost_2 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test_f, y_test_f, 0.)
            #print "Test Cost ", cost
            
            #'''
        g_func_old = g_func

        if callback is not None:
            callback(x, lambdak)

    task_str = task.tostring().strip(b'\x00').strip()
    if task_str.startswith(b'CONV'):
        warnflag = 0
    else:
        warnflag = 2
        
    np.save("loss_hoag.npy", loss_total)
    np.save("time_hoag.npy", time_total)
    return x, lambdak, warnflag
