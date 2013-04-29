'''
Created on 24 Apr 2013

@author: maxz
'''
from GPy.inference.gradient_descent_update_rules import FletcherReeves
import numpy
from multiprocessing import Value
from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from Queue import Empty
import sys
from threading import Thread

RUNNING = "running"
CONVERGED = "converged"
MAXITER = "maximum number of iterations reached"
MAX_F_EVAL = "maximum number of function calls reached"
LINE_SEARCH = "line search failed"
KBINTERRUPT = "interrupted"

SENTINEL = None

class _Async_Optimization(Thread):
    def __init__(self, f, df, x0, update_rule, runsignal,
                 report_every=10, messages=0, maxiter=5e3, max_f_eval=15e3,
                 gtol=1e-6, outqueue=None, *args, **kw):
        """
        Helper Process class for async optimization
        
        f_call and df_call are Multiprocessing Values, for synchronized assignment
        """
        self.f_call = Value('i', 0)
        self.df_call = Value('i', 0)
        self.f = self.f_wrapper(f, self.f_call)
        self.df = self.f_wrapper(df, self.df_call)
        self.x0 = x0
        self.update_rule = update_rule
        self.report_every = report_every
        self.messages = messages
        self.maxiter = maxiter
        self.max_f_eval = max_f_eval
        self.gtol = gtol
        self.runsignal = runsignal
#         self.parent = parent
#         self.result = None
        self.outq = outqueue
        super(_Async_Optimization, self).__init__(target=self.run,
                                            name="CG Optimization",
                                            *args, **kw)

#     def __enter__(self):
#         return self
#
#     def __exit__(self, type, value, traceback):
#         return isinstance(value, TypeError)

    def f_wrapper(self, f, counter):
        def f_w(*a, **kw):
            counter.value += 1
            return f(*a, **kw)
        return f_w

    def callback(self, *a):
        self.outq.put(a)
#         self.parent and self.parent.callback(*a, **kw)
        pass
        # print "callback done"

    def callback_return(self, *a):
        self.callback(*a)
        self.outq.put(SENTINEL)
        self.runsignal.clear()

    def run(self, *args, **kwargs):
        raise NotImplementedError("Overwrite this with optimization (for async use)")
        pass

class _CGDAsync(_Async_Optimization):

    def reset(self, xi, *a, **kw):
        gi = -self.df(xi, *a, **kw)
        si = gi
        ur = self.update_rule(gi)
        return gi, ur, si

    def run(self, *a, **kw):
        status = RUNNING

        fi = self.f(self.x0)
        fi_old = fi + 5000

        gi, ur, si = self.reset(self.x0, *a, **kw)
        xi = self.x0
        xi_old = numpy.nan
        it = 0

        while it < self.maxiter:
            if not self.runsignal.is_set():
                break

            if self.f_call.value > self.max_f_eval:
                status = MAX_F_EVAL

            gi = -self.df(xi, *a, **kw)
            if numpy.dot(gi.T, gi) < self.gtol:
                status = CONVERGED
                break
            if numpy.isnan(numpy.dot(gi.T, gi)):
                if numpy.any(numpy.isnan(xi_old)):
                    status = CONVERGED
                    break
                self.reset(xi_old)

            gammai = ur(gi)
            if gammai < 1e-6 or it % xi.shape[0] == 0:
                gi, ur, si = self.reset(xi, *a, **kw)
            si = gi + gammai * si
            alphai, _, _, fi2, fi_old2, gfi = line_search_wolfe1(self.f,
                                                                 self.df,
                                                                 xi,
                                                                 si, gi,
                                                                 fi, fi_old)
            if alphai is not None and fi2 < fi:
                fi, fi_old = fi2, fi_old2
            else:
                alphai, _, _, fi, fi_old, gfi = \
                         line_search_wolfe2(self.f, self.df,
                                            xi, si, gi,
                                            fi, fi_old)
                if alphai is None:
                    # This line search also failed to find a better solution.
                    status = LINE_SEARCH
                    break
            if gfi is not None:
                gi = gfi

            if fi_old > fi:
                gi, ur, si = self.reset(xi, *a, **kw)
            else:
                xi += numpy.dot(alphai, si)
                if self.messages:
                    sys.stdout.write("\r")
                    sys.stdout.flush()
                    sys.stdout.write("iteration: {0:> 6g}  f:{1:> 12e}  |g|:{2:> 12e}".format(it, fi, numpy.dot(gi.T, gi)))

                if it % self.report_every == 0:
                    self.callback(xi, fi, it, self.f_call.value, self.df_call.value, status)
            it += 1
        else:
            status = MAXITER
        # self.result = [xi, fi, it, self.f_call.value, self.df_call.value, status]
        self.callback_return(xi, fi, it, self.f_call.value, self.df_call.value, status)

class Async_Optimize(object):
    callback = lambda *x: None
    runsignal = Event()

    def async_callback_collect(self, q):
        while self.runsignal.is_set():
            try:
                for ret in iter(lambda: q.get(timeout=1), SENTINEL):
                    self.callback(*ret)
            except Empty:
                pass

    def fmin_async(self, f, df, x0, callback, update_rule=FletcherReeves,
                   messages=0, maxiter=5e3, max_f_eval=15e3, gtol=1e-6,
                   report_every=10, *args, **kwargs):
        self.runsignal.set()
        outqueue = Queue(5)
        if callback:
            self.callback = callback
        c = Thread(target=self.async_callback_collect, args=(outqueue,))
        c.start()
        p = _CGDAsync(f, df, x0, update_rule, self.runsignal,
                 report_every=report_every, messages=messages, maxiter=maxiter,
                 max_f_eval=max_f_eval, gtol=gtol, outqueue=outqueue, *args, **kwargs)
        p.run()
        return p, c

    def fmin(self, f, df, x0, callback=None, update_rule=FletcherReeves,
                   messages=0, maxiter=5e3, max_f_eval=15e3, gtol=1e-6,
                   report_every=10, *args, **kwargs):
        p, c = self.fmin_async(f, df, x0, callback, update_rule, messages,
                            maxiter, max_f_eval, gtol,
                            report_every, *args, **kwargs)
        while self.runsignal.is_set():
            try:
                p.join(1)
                c.join(1)
            except KeyboardInterrupt:
                # print "^C"
                self.runsignal.clear()
                p.join()
                c.join()

class CGD(Async_Optimize):
    '''
    Conjugate gradient descent algorithm to minimize
    function f with gradients df, starting at x0
    with update rule update_rule
    
    if df returns tuple (grad, natgrad) it will optimize according 
    to natural gradient rules
    '''
    name = "Conjugate Gradient Descent"

    def fmin_async(self, *a, **kw):
        """
        fmin_async(self, f, df, x0, callback, update_rule=FletcherReeves,
               messages=0, maxiter=5e3, max_f_eval=15e3, gtol=1e-6,
               report_every=10, *args, **kwargs)
        
        callback gets called every `report_every` iterations

            callback(xi, fi, iteration, function_calls, gradient_calls, status_message)
        
        if df returns tuple (grad, natgrad) it will optimize according 
        to natural gradient rules
    
        f, and df will be called with
            
            f(xi, *args, **kwargs)
            df(xi, *args, **kwargs)
        
        **returns**
        -----------
        
            Started `Process` object, optimizing asynchronously 
        
        **calls** 
        ---------
        
            callback(x_opt, f_opt, iteration, function_calls, gradient_calls, status_message)
        
        at end of optimization!
        """
        return super(CGD, self).fmin_async(*a, **kw)

    def fmin(self, *a, **kw):
        """
        fmin(self, f, df, x0, callback=None, update_rule=FletcherReeves,
               messages=0, maxiter=5e3, max_f_eval=15e3, gtol=1e-6,
               report_every=10, *args, **kwargs)
        
        Minimize f, calling callback every `report_every` iterations with following syntax:
        
            callback(xi, fi, iteration, function_calls, gradient_calls, status_message)
        
        if df returns tuple (grad, natgrad) it will optimize according 
        to natural gradient rules
    
        f, and df will be called with
            
            f(xi, *args, **kwargs)
            df(xi, *args, **kwargs)
                
        **returns** 
        ---------
        
            x_opt, f_opt, iteration, function_calls, gradient_calls, status_message
        
        at end of optimization
        """
        return super(CGD, self).fmin(*a, **kw)

