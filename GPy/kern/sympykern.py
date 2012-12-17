import numpy as np
import sympy as sp
from sympy.utilities.codegen import codegen
from sympy.core.cache import clear_cache
from scipy import weave
import re
import os
import sys
current_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
import tempfile
import pdb
from kernpart import kernpart

class spkern(kernpart):
    """
    A kernel object, where all the hard work in done by sympy.

    :param k: the covariance function
    :type k: a positive definite sympy function of x1, z1, x2, z2...

    To construct a new sympy kernel, you'll need to define:
     - a kernel function using a sympy object. Ensure that the kernel is of the form k(x,z).
     - that's it! we'll extract the variables from the function k.

    Note:
     - to handle multiple inputs, call them x1, z1, etc
     - to handle multpile correlated outputs, you'll need to define each covariance function and 'cross' variance function. TODO
    """
    def __init__(self,D,k,param=None):
        self.name='sympykern'
        self._sp_k = k
        sp_vars = [e for e in k.atoms() if e.is_Symbol]
        self._sp_x= sorted([e for e in sp_vars if e.name[0]=='x'],key=lambda x:int(x.name[1:]))
        self._sp_z= sorted([e for e in sp_vars if e.name[0]=='z'],key=lambda z:int(z.name[1:]))
        assert all([x.name=='x%i'%i for i,x in enumerate(self._sp_x)])
        assert all([z.name=='z%i'%i for i,z in enumerate(self._sp_z)])
        assert len(self._sp_x)==len(self._sp_z)
        self.D = len(self._sp_x)
        assert self.D == D
        self._sp_theta = sorted([e for e in sp_vars if not (e.name[0]=='x' or e.name[0]=='z')],key=lambda e:e.name)
        self.Nparam = len(self._sp_theta)

        #deal with param
        if param is None:
            param = np.ones(self.Nparam)
        assert param.size==self.Nparam
        self.set_param(param)

        #Differentiate!
        self._sp_dk_dtheta = [sp.diff(k,theta) for theta in self._sp_theta]
        self._sp_dk_dx = [sp.diff(k,xi) for xi in self._sp_x]
        #self._sp_dk_dz = [sp.diff(k,zi) for zi in self._sp_z]

        #self.compute_psi_stats()
        self._gen_code()

        self.weave_kwargs = {\
            'support_code':self._function_code,\
            'include_dirs':[tempfile.gettempdir(), os.path.join(current_dir,'symbolic/')],\
            'headers':['"sympy_helpers.h"'],\
            'sources':[os.path.join(current_dir,"symbolic/sympy_helpers.cpp")],\
            #'extra_compile_args':['-ftree-vectorize', '-mssse3', '-ftree-vectorizer-verbose=5'],\
            'extra_compile_args':[],\
            'extra_link_args':['-lgomp'],\
            'verbose':True}



    def __add__(self,other):
        return spkern(self._sp_k+other._sp_k)

    def compute_psi_stats(self):
        #define some normal distributions
        mus = [sp.var('mu%i'%i,real=True) for i in range(self.D)]
        Ss = [sp.var('S%i'%i,positive=True) for i in range(self.D)]
        normals = [(2*sp.pi*Si)**(-0.5)*sp.exp(-0.5*(xi-mui)**2/Si) for xi, mui, Si in zip(self._sp_x, mus, Ss)]

        #do some integration!
        #self._sp_psi0 = ??
        self._sp_psi1 = self._sp_k
        for i in range(self.D):
            print 'perfoming integrals %i of %i'%(i+1,2*self.D)
            sys.stdout.flush()
            self._sp_psi1 *= normals[i]
            self._sp_psi1 = sp.integrate(self._sp_psi1,(self._sp_x[i],-sp.oo,sp.oo))
            clear_cache()
        self._sp_psi1 = self._sp_psi1.simplify()

        #and here's psi2 (eek!)
        zprime = [sp.Symbol('zp%i'%i) for i in range(self.D)]
        self._sp_psi2 = self._sp_k.copy()*self._sp_k.copy().subs(zip(self._sp_z,zprime))
        for i in range(self.D):
            print 'perfoming integrals %i of %i'%(self.D+i+1,2*self.D)
            sys.stdout.flush()
            self._sp_psi2 *= normals[i]
            self._sp_psi2 = sp.integrate(self._sp_psi2,(self._sp_x[i],-sp.oo,sp.oo))
            clear_cache()
        self._sp_psi2 = self._sp_psi2.simplify()


    def _gen_code(self):
        #generate c functions from sympy objects
        (foo_c,self._function_code),(foo_h,self._function_header) = \
                codegen([('k',self._sp_k)] \
                + [('dk_d%s'%x.name,dx) for x,dx in zip(self._sp_x,self._sp_dk_dx)]\
                #+ [('dk_d%s'%z.name,dz) for z,dz in zip(self._sp_z,self._sp_dk_dz)]\
                + [('dk_d%s'%theta.name,dtheta) for theta,dtheta in zip(self._sp_theta,self._sp_dk_dtheta)]\
                ,"C",'foobar',argument_sequence=self._sp_x+self._sp_z+self._sp_theta)
        #put the header file where we can find it
        f = file(os.path.join(tempfile.gettempdir(),'foobar.h'),'w')
        f.write(self._function_header)
        f.close()

        #get rid of derivatives of DiracDelta
        self._function_code = re.sub('DiracDelta\(.+?,.+?\)','0.0',self._function_code)

        #Here's some code to do the looping for K
        arglist = ", ".join(["X[i*D+%s]"%x.name[1:] for x in self._sp_x]\
                + ["Z[j*D+%s]"%z.name[1:] for z in self._sp_z]\
                + ["param[%i]"%i for i in range(self.Nparam)])

        self._K_code =\
        """
        int i;
        int j;
        int N = target_array->dimensions[0];
        int M = target_array->dimensions[1];
        int D = X_array->dimensions[1];
        //#pragma omp parallel for private(j)
        for (i=0;i<N;i++){
            for (j=0;j<M;j++){
                target[i*M+j] = k(%s);
            }
        }
        """%(arglist)

        diag_arglist = re.sub('Z','X',arglist)
        diag_arglist = re.sub('j','i',diag_arglist)
        #Here's some code to do the looping for Kdiag
        self._Kdiag_code =\
        """
        int i;
        int N = target_array->dimensions[0];
        int D = X_array->dimensions[1];
        //#pragma omp parallel for
        for (i=0;i<N;i++){
                target[i] = k(%s);
        }
        """%diag_arglist

        #here's some code to compute gradients
        funclist = '\n'.join([' '*16 + 'target[%i] += partial[i*M+j]*dk_d%s(%s);'%(i,theta.name,arglist) for i,theta in  enumerate(self._sp_theta)])
        self._dK_dtheta_code =\
        """
        int i;
        int j;
        int N = partial_array->dimensions[0];
        int M = partial_array->dimensions[1];
        int D = X_array->dimensions[1];
        //#pragma omp parallel for private(j)
        for (i=0;i<N;i++){
            for (j=0;j<M;j++){
%s
            }
        }
        """%funclist

        #here's some code to compute gradients for Kdiag TODO: thius is yucky.
        diag_funclist = re.sub('Z','X',funclist,count=0)
        diag_funclist = re.sub('j','i',diag_funclist)
        diag_funclist = re.sub('partial\[i\*M\+i\]','partial[i]',diag_funclist)
        self._dKdiag_dtheta_code =\
        """
        int i;
        int N = partial_array->dimensions[0];
        int D = X_array->dimensions[1];
        for (i=0;i<N;i++){
                %s
        }
        """%diag_funclist

        #Here's some code to do gradients wrt x
        gradient_funcs = "\n".join(["target[i*D+%i] += partial[i*M+j]*dk_dx%i(%s);"%(q,q,arglist) for q in range(self.D)])
        self._dK_dX_code = \
        """
        int i;
        int j;
        int N = partial_array->dimensions[0];
        int M = partial_array->dimensions[1];
        int D = X_array->dimensions[1];
        //#pragma omp parallel for private(j)
        for (i=0;i<N; i++){
            for (j=0; j<M; j++){
                %s
            }
        }
        """%gradient_funcs

        #Here's some code to do gradients wrt z (should be the same as for X, but this is easier
        gradient_funcs_Z = "\n".join(["target[j*D+%i] += partial[i*M+j]*dk_dz%i(%s);"%(q,q,arglist) for q in range(self.D)])
        self._dK_dZ_code = \
        """
        int i;
        int j;
        int N = partial_array->dimensions[0];
        int M = partial_array->dimensions[1];
        int D = X_array->dimensions[1];
        for (i=0;i<N; i++){
            for (j=0; j<M; j++){
                %s
            }
        }
        """%gradient_funcs_Z


        #now for gradients of Kdiag wrt X
        self._dKdiag_dX_code= \
        """
        int i;
        int j;
        int N = partial_array->dimensions[0];
        int M = 0;
        int D = X_array->dimensions[1];
        for (i=0;i<N; i++){
            j = i;
            %s
        }
        """%gradient_funcs


        #TODO: insert multiple functions here via string manipulation
        #TODO: similar functions for psi_stats

    def K(self,X,Z,target):
        param = self._param
        weave.inline(self._K_code,arg_names=['target','X','Z','param'],**self.weave_kwargs)
        return target

    def Kdiag(self,X,target):
        param = self._param
        weave.inline(self._Kdiag_code,arg_names=['target','X','param'],**self.weave_kwargs)
        return target

    def dK_dtheta(self,partial,X,Z,target):
        param = self._param
        weave.inline(self._dK_dtheta_code,arg_names=['target','X','Z','param','partial'],**self.weave_kwargs)
        return target

    def dKdiag_dtheta(self,partial,X,target):
        param = self._param
        Z = X
        weave.inline(self._dKdiag_dtheta_code,arg_names=['target','X','Z','param','partial'],**self.weave_kwargs)
        return target

    def dK_dX(self,partial,X,Z,target):
        target = np.zeros_like(X)
        param = self._param
        weave.inline(self._dK_dX_code,arg_names=['target','X','Z','param','partial'],**self.weave_kwargs)
        return target

    #def dK_dZ(self,X,Z,partial=None):
        ##TODO: this function might not be necessary
        #target = np.zeros_like(Z)
        #param = self._param
        #weave.inline(self._dK_dZ_code,arg_names=['target','X','Z','param','partial'],**self.weave_kwargs)
        #return target

    def dKdiag_dX(self,partial,X,target):
        param = self._param
        Z = X
        weave.inline(self._dKdiag_dX_code,arg_names=['target','X','Z','param','partial'],**self.weave_kwargs)
        return target

    def set_param(self,param):
        #print param.flags['C_CONTIGUOUS']
        self._param = param.copy()

    def get_param(self):
        return self._param

    def get_param_names(self):
        return [x.name for x in self._sp_theta]
