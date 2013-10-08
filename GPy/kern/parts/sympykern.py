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
import ast
from kernpart import Kernpart

class spkern(Kernpart):
    """
    A kernel object, where all the hard work in done by sympy.

    :param k: the covariance function
    :type k: a positive definite sympy function of x_0, z_0, x_1, z_1, x_2, z_2...

    To construct a new sympy kernel, you'll need to define:
     - a kernel function using a sympy object. Ensure that the kernel is of the form k(x,z).
     - that's it! we'll extract the variables from the function k.

    Note:
     - to handle multiple inputs, call them x_1, z_1, etc
     - to handle multpile correlated outputs, you'll need to add parameters with an index, such as lengthscale_i and lengthscale_j.
    """
    def __init__(self,input_dim, k=None, output_dim=1, name=None, param=None):
        if name is None:
            self.name='sympykern'
        else:
            self.name = name
        if k is None:
            raise ValueError, "You must provide an argument for the covariance function."
        self._sp_k = k
        sp_vars = [e for e in k.atoms() if e.is_Symbol]
        self._sp_x= sorted([e for e in sp_vars if e.name[0:2]=='x_'],key=lambda x:int(x.name[2:]))
        self._sp_z= sorted([e for e in sp_vars if e.name[0:2]=='z_'],key=lambda z:int(z.name[2:]))
        # Check that variable names make sense.
        assert all([x.name=='x_%i'%i for i,x in enumerate(self._sp_x)])
        assert all([z.name=='z_%i'%i for i,z in enumerate(self._sp_z)])
        assert len(self._sp_x)==len(self._sp_z)
        self.input_dim = len(self._sp_x)
        if output_dim > 1:
            self.input_dim += 1
        assert self.input_dim == input_dim
        self.output_dim = output_dim
        # extract parameter names
        thetas = sorted([e for e in sp_vars if not (e.name[0:2]=='x_' or e.name[0:2]=='z_')],key=lambda e:e.name)


        # Look for parameters with index.
        if self.output_dim>1:
            self._sp_theta_i = sorted([e for e in thetas if (e.name[-2:]=='_i')], key=lambda e:e.name)
            self._sp_theta_j = sorted([e for e in thetas if (e.name[-2:]=='_j')], key=lambda e:e.name)
            # Make sure parameter appears with both indices!
            assert len(self._sp_theta_i)==len(self._sp_theta_j)
            assert all([theta_i.name[:-2]==theta_j.name[:-2] for theta_i, theta_j in zip(self._sp_theta_i, self._sp_theta_j)])

            # Extract names of shared parameters
            self._sp_theta = [theta for theta in thetas if theta not in self._sp_theta_i and theta not in self._sp_theta_j]
            
            self.num_split_params = len(self._sp_theta_i)
            self._split_param_names = ["%s"%theta.name[:-2] for theta in self._sp_theta_i]
            for params in self._split_param_names:
                setattr(self, params, np.ones(self.output_dim))
            
            self.num_shared_params = len(self._sp_theta)
            self.num_params = self.num_shared_params+self.num_split_params*self.output_dim
            
        else:
            self.num_split_params = 0
            self._split_param_names = []
            self._sp_theta = thetas
            self.num_shared_params = len(self._sp_theta)
            self.num_params = self.num_shared_params

        #deal with param
        if param is None:
            param = np.ones(self.num_params)
            
        assert param.size==self.num_params
        self._set_params(param)

        #Differentiate!
        self._sp_dk_dtheta = [sp.diff(k,theta).simplify() for theta in self._sp_theta]
        if self.output_dim > 1:
            self._sp_dk_dtheta_i = [sp.diff(k,theta).simplify() for theta in self._sp_theta_i]
            
        self._sp_dk_dx = [sp.diff(k,xi).simplify() for xi in self._sp_x]
        #self._sp_dk_dz = [sp.diff(k,zi) for zi in self._sp_z]

        #self.compute_psi_stats()
        self._gen_code()

        self.weave_kwargs = {\
            'support_code':self._function_code,\
            'include_dirs':[tempfile.gettempdir(), os.path.join(current_dir,'parts/')],\
            'headers':['"sympy_helpers.h"'],\
            'sources':[os.path.join(current_dir,"parts/sympy_helpers.cpp")],\
            #'extra_compile_args':['-ftree-vectorize', '-mssse3', '-ftree-vectorizer-verbose=5'],\
            'extra_compile_args':[],\
            'extra_link_args':['-lgomp'],\
            'verbose':True}

    def __add__(self,other):
        return spkern(self._sp_k+other._sp_k)

    def compute_psi_stats(self):
        #define some normal distributions
        mus = [sp.var('mu_%i'%i,real=True) for i in range(self.input_dim)]
        Ss = [sp.var('S_%i'%i,positive=True) for i in range(self.input_dim)]
        normals = [(2*sp.pi*Si)**(-0.5)*sp.exp(-0.5*(xi-mui)**2/Si) for xi, mui, Si in zip(self._sp_x, mus, Ss)]

        #do some integration!
        #self._sp_psi0 = ??
        self._sp_psi1 = self._sp_k
        for i in range(self.input_dim):
            print 'perfoming integrals %i of %i'%(i+1,2*self.input_dim)
            sys.stdout.flush()
            self._sp_psi1 *= normals[i]
            self._sp_psi1 = sp.integrate(self._sp_psi1,(self._sp_x[i],-sp.oo,sp.oo))
            clear_cache()
        self._sp_psi1 = self._sp_psi1.simplify()

        #and here's psi2 (eek!)
        zprime = [sp.Symbol('zp%i'%i) for i in range(self.input_dim)]
        self._sp_psi2 = self._sp_k.copy()*self._sp_k.copy().subs(zip(self._sp_z,zprime))
        for i in range(self.input_dim):
            print 'perfoming integrals %i of %i'%(self.input_dim+i+1,2*self.input_dim)
            sys.stdout.flush()
            self._sp_psi2 *= normals[i]
            self._sp_psi2 = sp.integrate(self._sp_psi2,(self._sp_x[i],-sp.oo,sp.oo))
            clear_cache()
        self._sp_psi2 = self._sp_psi2.simplify()


    def _gen_code(self):
        #generate c functions from sympy objects        
        argument_sequence = self._sp_x+self._sp_z+self._sp_theta
        code_list = [('k',self._sp_k)]
        # gradients with respect to covariance input
        code_list += [('dk_d%s'%x.name,dx) for x,dx in zip(self._sp_x,self._sp_dk_dx)]
        # gradient with respect to parameters
        code_list += [('dk_d%s'%theta.name,dtheta) for theta,dtheta in zip(self._sp_theta,self._sp_dk_dtheta)]
        # gradient with respect to multiple output parameters
        if self.output_dim > 1:
            argument_sequence += self._sp_theta_i + self._sp_theta_j
            code_list += [('dk_d%s'%theta.name,dtheta) for theta,dtheta in zip(self._sp_theta_i,self._sp_dk_dtheta_i)]
        (foo_c,self._function_code), (foo_h,self._function_header) = \
                                     codegen(code_list, "C",'foobar',argument_sequence=argument_sequence)
        #put the header file where we can find it
        f = file(os.path.join(tempfile.gettempdir(),'foobar.h'),'w')
        f.write(self._function_header)
        f.close()

        # Substitute any known derivatives which sympy doesn't compute
        self._function_code = re.sub('DiracDelta\(.+?,.+?\)','0.0',self._function_code)

        # This is the basic argument construction for the C code.
        arg_list = (["X[i*input_dim+%s]"%x.name[2:] for x in self._sp_x]
                    + ["Z[j*input_dim+%s]"%z.name[2:] for z in self._sp_z])
        if self.output_dim>1:
            reverse_arg_list = list(arg_list)
            reverse_arg_list.reverse()

        param_arg_list = ["param[%i]"%i for i in range(self.num_shared_params)]
        arg_list += param_arg_list

        precompute_list=[]
        if self.output_dim > 1:
            reverse_arg_list+=list(param_arg_list)
            split_param_arg_list = ["%s[%s]"%(theta.name[:-2],index) for index in ['ii', 'jj'] for theta in self._sp_theta_i]
            split_param_reverse_arg_list = ["%s[%s]"%(theta.name[:-2],index) for index in ['jj', 'ii'] for theta in self._sp_theta_i]
            arg_list += split_param_arg_list
            reverse_arg_list += split_param_reverse_arg_list
            precompute_list += [' '*16+"int %s=(int)%s[%s*input_dim+output_dim];"%(index, var, index2) for index, var, index2 in zip(['ii', 'jj'], ['X', 'Z'], ['i', 'j'])]
            reverse_arg_string = ", ".join(reverse_arg_list)
        arg_string = ", ".join(arg_list)
        precompute_string = "\n".join(precompute_list)
        # Here's the code to do the looping for K
        self._K_code =\
        """
        int i;
        int j;
        int N = target_array->dimensions[0];
        int num_inducing = target_array->dimensions[1];
        int input_dim = X_array->dimensions[1];
        //#pragma omp parallel for private(j)
        for (i=0;i<N;i++){
            for (j=0;j<num_inducing;j++){
%s
                target[i*num_inducing+j] = k(%s);
            }
        }
        %s
        """%(precompute_string,arg_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed

        
        # Code to compute diagonal of covariance.
        diag_arg_string = re.sub('Z','X',arg_string)
        diag_arg_string = re.sub('j','i',diag_arg_string)
        diag_precompute_string = re.sub('Z','X',precompute_string)
        diag_precompute_string = re.sub('j','i',diag_precompute_string)
        # Code to do the looping for Kdiag
        self._Kdiag_code =\
        """
        int i;
        int N = target_array->dimensions[0];
        int input_dim = X_array->dimensions[1];
        //#pragma omp parallel for
        for (i=0;i<N;i++){
                %s
                target[i] = k(%s);
        }
        %s
        """%(diag_precompute_string,diag_arg_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed

        # Code to compute gradients
        func_list = ([' '*16 + 'target[%i] += partial[i*num_inducing+j]*dk_d%s(%s);'%(i,theta.name,arg_string) for i,theta in  enumerate(self._sp_theta)])
        if self.output_dim>1:
            func_list += [' '*16 + "int %s=(int)%s[%s*input_dim+output_dim];"%(index, var, index2) for index, var, index2 in zip(['ii', 'jj'], ['X', 'Z'], ['i', 'j'])]
            func_list += [' '*16 + 'target[%i+ii] += partial[i*num_inducing+j]*dk_d%s(%s);'%(self.num_shared_params+i*self.output_dim, theta.name, arg_string) for i, theta in enumerate(self._sp_theta_i)]
            func_list += [' '*16 + 'target[%i+jj] += partial[i*num_inducing+j]*dk_d%s(%s);'%(self.num_shared_params+i*self.output_dim, theta.name, reverse_arg_string) for i, theta in enumerate(self._sp_theta_i)]
        func_string = '\n'.join(func_list) 

        self._dK_dtheta_code =\
        """
        int i;
        int j;
        int N = partial_array->dimensions[0];
        int num_inducing = partial_array->dimensions[1];
        int input_dim = X_array->dimensions[1];
        //#pragma omp parallel for private(j)
        for (i=0;i<N;i++){
            for (j=0;j<num_inducing;j++){
%s
            }
        }
        %s
        """%(func_string,"/*"+str(self._sp_k)+"*/") # adding a string representation forces recompile when needed


        # Code to compute gradients for Kdiag TODO: needs clean up
        diag_func_string = re.sub('Z','X',func_string,count=0)
        diag_func_string = re.sub('j','i',diag_func_string)
        diag_func_string = re.sub('partial\[i\*num_inducing\+i\]','partial[i]',diag_func_string)
        self._dKdiag_dtheta_code =\
        """
        int i;
        int N = partial_array->dimensions[0];
        int input_dim = X_array->dimensions[1];
        for (i=0;i<N;i++){
                %s
        }
        %s
        """%(diag_func_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed

        # Code for gradients wrt X
        gradient_funcs = "\n".join(["target[i*input_dim+%i] += partial[i*num_inducing+j]*dk_dx%i(%s);"%(q,q,arg_string) for q in range(self.input_dim)])

        self._dK_dX_code = \
        """
        int i;
        int j;
        int N = partial_array->dimensions[0];
        int num_inducing = partial_array->dimensions[1];
        int input_dim = X_array->dimensions[1];
        //#pragma omp parallel for private(j)
        for (i=0;i<N; i++){
          for (j=0; j<num_inducing; j++){
            %s
          }
        }
        %s
        """%(gradient_funcs,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed
  

        diag_gradient_funcs = re.sub('Z','X',gradient_funcs,count=0)
        diag_gradient_funcs = re.sub('j','i',diag_gradient_funcs)
        diag_gradient_funcs = re.sub('partial\[i\*num_inducing\+i\]','2*partial[i]',diag_gradient_funcs)

        # Code for gradients of Kdiag wrt X
        self._dKdiag_dX_code= \
        """
        int N = partial_array->dimensions[0];
        int input_dim = X_array->dimensions[1];
        for (int i=0;i<N; i++){
            %s
        }
        %s
        """%(diag_gradient_funcs,"/*"+str(self._sp_k)+"*/") #adding a
        # string representation forces recompile when needed Get rid
        # of Zs in argument for diagonal. TODO: Why wasn't
        # diag_func_string called here? Need to check that.
        #self._dKdiag_dX_code = self._dKdiag_dX_code.replace('Z[j', 'X[i')

        # Code to use when only X is provided. 
        self._K_code_X = self._K_code.replace('Z[', 'X[')
        self._dK_dtheta_code_X = self._dK_dtheta_code.replace('Z[', 'X[')
        self._dK_dX_code_X = self._dK_dX_code.replace('Z[', 'X[').replace('+= partial[', '+= 2*partial[')


        #TODO: insert multiple functions here via string manipulation
        #TODO: similar functions for psi_stats
    def _get_arg_names(self, Z=None, partial=None):
        arg_names = ['target','X','param']
        if Z is not None:
            arg_names += ['Z']
        if partial is not None:
            arg_names += ['partial']
        if self.output_dim>1:
            arg_names += self._split_param_names
            arg_names += ['output_dim']
        return arg_names
        
    def _weave_inline(self, code, X, target, Z=None, partial=None):
        param, output_dim = self._shared_params, self.output_dim

        # Need to extract parameters first
        for split_params in self._split_param_names:
            locals()[split_params] = getattr(self, split_params)
        arg_names = self._get_arg_names(Z, partial)        
        weave.inline(code=code, arg_names=arg_names,**self.weave_kwargs)

    def K(self,X,Z,target):        
        if Z is None:
            self._weave_inline(self._K_code_X, X, target)
        else:
            self._weave_inline(self._K_code, X, target, Z)


    def Kdiag(self,X,target):
        self._weave_inline(self._Kdiag_code, X, target)

    def dK_dtheta(self,partial,X,Z,target):
        if Z is None:
            self._weave_inline(self._dK_dtheta_code_X, X, target, Z, partial)
        else:
            self._weave_inline(self._dK_dtheta_code, X, target, Z, partial)
            
    def dKdiag_dtheta(self,partial,X,target):
        self._weave_inline(self._dKdiag_dtheta_code, X, target, Z=None, partial=partial)
               
    def dK_dX(self,partial,X,Z,target):
        if Z is None:
            self._weave_inline(self._dK_dX_code_X, X, target, Z, partial)
        else:
            self._weave_inline(self._dK_dX_code, X, target, Z, partial)

    def dKdiag_dX(self,partial,X,target):
        self._weave.inline(self._dKdiag_dX_code, X, target, Z, partial)

    def _set_params(self,param):
        #print param.flags['C_CONTIGUOUS']
        assert param.size == (self.num_params)
        self._shared_params = param[0:self.num_shared_params]
        if self.output_dim>1:
            for i, split_params in enumerate(self._split_param_names):
                start = self.num_shared_params + i*self.output_dim
                end = self.num_shared_params + (i+1)*self.output_dim
                setattr(self, split_params, param[start:end])


    def _get_params(self):
        params = self._shared_params
        if self.output_dim>1:
            for split_params in self._split_param_names:
                params = np.hstack((params, getattr(self, split_params).flatten()))
        return params

    def _get_param_names(self):
        if self.output_dim>1:
            return [x.name for x in self._sp_theta] + [x.name[:-2] + str(i)  for x in self._sp_theta_i for i in range(self.output_dim)]
        else:
            return [x.name for x in self._sp_theta]
