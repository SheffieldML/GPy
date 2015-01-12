# Copyright (c) 2012-2014, Max Zwiessele.
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import sys

def exponents(fnow, current_grad):
    exps = [np.abs(np.float(fnow)), current_grad]
    return np.sign(exps) * np.log10(exps).astype(int)

class VerboseOptimization(object):
    def __init__(self, model, maxiters, verbose=True, current_iteration=0, ipython_notebook=False):
        self.verbose = verbose
        if self.verbose:
            self.model = model
            self.iteration = current_iteration
            self.ipython_notebook = ipython_notebook
            self.p_iter = self.iteration
            self.maxiters = maxiters
            self.len_maxiters = len(str(maxiters))
            self.model.add_observer(self, self.print_status)

            self.update()

            if self.ipython_notebook:
                from IPython.display import display
                from IPython.html.widgets import IntProgressWidget, HTMLWidget
                self.text = HTMLWidget()
                display(self.text)
                self.progress = IntProgressWidget()
                display(self.progress)
            else:
                self.exps = exponents(self.fnow, self.current_gradient)
                print ' {0:{mi}s}   {1:11s}    {2:11s}'.format("I", "F", "|g|", mi=self.len_maxiters)

    def __enter__(self):
        return self

    def print_out(self):
        if self.ipython_notebook:
            names_vals = [['Iteration', "{:>0{l}}".format(self.iteration, l=self.len_maxiters)],
                          ['f', "{: > 05.3E}".format(self.fnow)],
                          ['||Gradient||', "{: >+05.3E}".format(float(self.current_gradient))],
                      ]
            #message = "Lik:{:5.3E} Grad:{:5.3E} Lik:{:5.3E} Len:{!s}".format(float(m.log_likelihood()), np.einsum('i,i->', grads, grads), float(m.likelihood.variance), " ".join(["{:3.2E}".format(l) for l in m.kern.lengthscale.values]))
            html_begin = """<style type="text/css">
    .tg-opt  {font-family:"Courier New", Courier, monospace !important;padding:2px 3px;word-break:normal;border-collapse:collapse;border-spacing:0;border-color:#DCDCDC;margin:0px auto;}
    .tg-opt td{font-family:"Courier New", Courier, monospace !important;font-weight:bold;color:#444;background-color:#F7FDFA;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#DCDCDC;}
    .tg-opt th{font-family:"Courier New", Courier, monospace !important;font-weight:normal;color:#fff;background-color:#26ADE4;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#DCDCDC;}
    .tg-opt .tg-left{font-family:"Courier New", Courier, monospace !important;font-weight:normal;text-align:left;}
    .tg-opt .tg-right{font-family:"Courier New", Courier, monospace !important;font-weight:normal;text-align:right;}
    </style>
    <table class="tg-opt">"""
            html_end = "</table>"
            html_body = ""
            for name, val in names_vals:
                html_body += "<tr>"
                html_body += "<td class='tg-left'>{}</td>".format(name)
                html_body += "<td class='tg-right'>{}</td>".format(val)
                html_body += "</tr>"
            self.text.value = html_begin + html_body + html_end
            self.progress.value = 100*(self.iteration+1)/self.maxiters
        else:
            n_exps = exponents(self.fnow, self.current_gradient)
            if self.iteration - self.p_iter >= 20 * np.random.rand():
                a = self.iteration >= self.p_iter * 2.78
                b = np.any(n_exps < self.exps)
                if a or b:
                    self.p_iter = self.iteration
                    print ''
                if b:
                    self.exps = n_exps
            print '\r',
            print '{0:>0{mi}g}  {1:> 12e}  {2:> 12e}'.format(self.iteration, float(self.fnow), float(self.current_gradient), mi=self.len_maxiters), # print 'Iteration:', iteration, ' Objective:', fnow, '  Scale:', beta, '\r',
            sys.stdout.flush()

    def print_status(self, me, which=None):
        self.update()

        #sys.stdout.write(" "*len(self.message))
        self.print_out()

        self.iteration += 1

    def update(self):
        self.fnow = self.model.objective_function()
        if self.model.obj_grads is not None:
            grad = self.model.obj_grads
            self.current_gradient = np.dot(grad, grad)
        else:
            self.current_gradient = np.nan

    def __exit__(self, type, value, traceback):
        if self.verbose:
            self.model.remove_observer(self)