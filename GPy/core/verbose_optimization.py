# Copyright (c) 2012-2014, Max Zwiessele.
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function
import numpy as np
import sys
import time

def exponents(fnow, current_grad):
    exps = [np.abs(np.float(fnow)), current_grad]
    return np.sign(exps) * np.log10(exps).astype(int)

class VerboseOptimization(object):
    def __init__(self, model, opt, maxiters, verbose=False, current_iteration=0, ipython_notebook=True):
        self.verbose = verbose
        if self.verbose:
            self.model = model
            self.iteration = current_iteration
            self.p_iter = self.iteration
            self.maxiters = maxiters
            self.len_maxiters = len(str(maxiters))
            self.opt_name = opt.opt_name
            self.model.add_observer(self, self.print_status)
            self.status = 'running'

            self.update()

            try:
                from IPython.display import display
                from IPython.html.widgets import FloatProgressWidget, HTMLWidget, ContainerWidget
                self.text = HTMLWidget()
                self.progress = FloatProgressWidget()
                self.model_show = HTMLWidget()
                self.ipython_notebook = ipython_notebook
            except:
                # Not in Ipython notebook
                self.ipython_notebook = False

            if self.ipython_notebook:
                self.text.set_css('width', '100%')
                #self.progress.set_css('width', '100%')

                left_col = ContainerWidget(children = [self.progress, self.text])
                right_col = ContainerWidget(children = [self.model_show])
                hor_align = ContainerWidget(children = [left_col, right_col])

                display(hor_align)

                left_col.set_css({
                         'padding': '2px',
                         'width': "100%",
                         })

                right_col.set_css({
                         'padding': '2px',
                         })

                hor_align.set_css({
                         'width': "100%",
                         })

                hor_align.remove_class('vbox')
                hor_align.add_class('hbox')

                left_col.add_class("box-flex1")
                right_col.add_class('box-flex0')

                #self.text.add_class('box-flex2')
                #self.progress.add_class('box-flex1')
            else:
                self.exps = exponents(self.fnow, self.current_gradient)
                print('Running {} Code:'.format(self.opt_name))
                print(' {3:7s}   {0:{mi}s}   {1:11s}    {2:11s}'.format("i", "f", "|g|", "secs", mi=self.len_maxiters))

    def __enter__(self):
        self.start = time.time()
        return self

    def print_out(self):
        if self.ipython_notebook:
            names_vals = [['optimizer', "{:s}".format(self.opt_name)],
                          ['runtime [s]', "{:> g}".format(time.time()-self.start)],
                          ['evaluation', "{:>0{l}}".format(self.iteration, l=self.len_maxiters)],
                          ['objective', "{: > 12.3E}".format(self.fnow)],
                          ['||gradient||', "{: >+12.3E}".format(float(self.current_gradient))],
                          ['status', "{:s}".format(self.status)],
                      ]
            #message = "Lik:{:5.3E} Grad:{:5.3E} Lik:{:5.3E} Len:{!s}".format(float(m.log_likelihood()), np.einsum('i,i->', grads, grads), float(m.likelihood.variance), " ".join(["{:3.2E}".format(l) for l in m.kern.lengthscale.values]))
            html_begin = """<style type="text/css">
    .tg-opt  {font-family:"Courier New", Courier, monospace !important;padding:2px 3px;word-break:normal;border-collapse:collapse;border-spacing:0;border-color:#DCDCDC;margin:0px auto;width:100%;}
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
            self.model_show.value = self.model._repr_html_()
        else:
            n_exps = exponents(self.fnow, self.current_gradient)
            if self.iteration - self.p_iter >= 20 * np.random.rand():
                a = self.iteration >= self.p_iter * 2.78
                b = np.any(n_exps < self.exps)
                if a or b:
                    self.p_iter = self.iteration
                    print('')
                if b:
                    self.exps = n_exps
            print('\r', end=' ')
            print('{3:> 7.2g}  {0:>0{mi}g}  {1:> 12e}  {2:> 12e}'.format(self.iteration, float(self.fnow), float(self.current_gradient), time.time()-self.start, mi=self.len_maxiters), end=' ') # print 'Iteration:', iteration, ' Objective:', fnow, '  Scale:', beta, '\r',
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

    def finish(self, opt):
        self.status = opt.status

    def __exit__(self, type, value, traceback):
        if self.verbose:
            self.stop = time.time()
            self.model.remove_observer(self)
            self.print_out()

            if not self.ipython_notebook:
                print()
                print('Optimization finished in {0:.5g} Seconds'.format(self.stop-self.start))
                print('Optimization status: {0:.5g}'.format(self.status))             
		print()
