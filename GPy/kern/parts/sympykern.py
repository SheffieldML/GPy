import numpy as np
import sympy as sp
from sympy.utilities.codegen import codegen
from sympy.core.cache import clear_cache
from scipy import weave
import re
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
import tempfile
import pdb
import ast
from kernpart import Kernpart
from ...util.config import config

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
    def __init__(self, input_dim, k=None, output_dim=1, name=None, param=None):
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
        self._real_input_dim = self.input_dim
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
            self._split_theta_names = ["%s"%theta.name[:-2] for theta in self._sp_theta_i]
            for theta in self._split_theta_names:
                setattr(self, theta, np.ones(self.output_dim))
            
            self.num_shared_params = len(self._sp_theta)
            self.num_params = self.num_shared_params+self.num_split_params*self.output_dim
            
        else:
            self.num_split_params = 0
            self._split_theta_names = []
            self._sp_theta = thetas
            self.num_shared_params = len(self._sp_theta)
            self.num_params = self.num_shared_params
        
        for theta in self._sp_theta:
            val = 1.0
            if param is not None:
                if param.has_key(theta):
                    val = param[theta]
            setattr(self, theta.name, val)
        #deal with param            
        self._set_params(self._get_params())

        #Differentiate!
        self._sp_dk_dtheta = [sp.diff(k,theta).simplify() for theta in self._sp_theta]
        if self.output_dim > 1:
            self._sp_dk_dtheta_i = [sp.diff(k,theta).simplify() for theta in self._sp_theta_i]
            
        self._sp_dk_dx = [sp.diff(k,xi).simplify() for xi in self._sp_x]

        if False:
            self.compute_psi_stats()

        self._gen_code()

        if False:
            extra_compile_args = ['-ftree-vectorize', '-mssse3', '-ftree-vectorizer-verbose=5']
        else:
            extra_compile_args = []
            
        self.weave_kwargs = {
            'support_code':self._function_code,
            'include_dirs':[tempfile.gettempdir(), current_dir],
            'headers':['"sympy_helpers.h"'],
            'sources':[os.path.join(current_dir,"sympy_helpers.cpp")],
            'extra_compile_args':extra_compile_args,
            'extra_link_args':[],
            'verbose':True}
        if config.getboolean('parallel', 'openmp'): self.weave_kwargs.append('-lgomp')

    def __add__(self,other):
        return spkern(self._sp_k+other._sp_k)

    def _gen_code(self):
        """Generates the C functions necessary for computing the covariance function using the sympy objects as input."""
        #TODO: maybe generate one C function only to save compile time? Also easier to take that as a basis and hand craft other covariances??

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


        ############################################################
        # This is the basic argument construction for the C code.  #
        ############################################################
        
        arg_list = (["X2(i, %s)"%x.name[2:] for x in self._sp_x]
                    + ["Z2(j, %s)"%z.name[2:] for z in self._sp_z])

        # for multiple outputs need to also provide these arguments reversed.
        if self.output_dim>1:
            reverse_arg_list = list(arg_list)
            reverse_arg_list.reverse()

        # Add in any 'shared' parameters to the list.
        param_arg_list = [shared_params.name for shared_params in self._sp_theta]
        arg_list += param_arg_list

        precompute_list=[]
        if self.output_dim > 1:
            reverse_arg_list+=list(param_arg_list)
            split_param_arg_list = ["%s1(%s)"%(theta.name[:-2].upper(),index) for index in ['ii', 'jj'] for theta in self._sp_theta_i]
            split_param_reverse_arg_list = ["%s1(%s)"%(theta.name[:-2].upper(),index) for index in ['jj', 'ii'] for theta in self._sp_theta_i]
            arg_list += split_param_arg_list
            reverse_arg_list += split_param_reverse_arg_list
            # Extract the right output indices from the inputs.
            c_define_output_indices = [' '*16 + "int %s=(int)%s(%s, %i);"%(index, var, index2, self.input_dim-1) for index, var, index2 in zip(['ii', 'jj'], ['X2', 'Z2'], ['i', 'j'])]
            precompute_list += c_define_output_indices
            reverse_arg_string = ", ".join(reverse_arg_list)
        arg_string = ", ".join(arg_list)
        precompute_string = "\n".join(precompute_list)

        # Code to compute argments string needed when only X is provided.
        X_arg_string = re.sub('Z','X',arg_string)
        # Code to compute argument string when only diagonal is required.
        diag_arg_string = re.sub('int jj','//int jj',X_arg_string)
        diag_arg_string = re.sub('j','i',diag_arg_string)
        if precompute_string == '':
            # if it's not multioutput, the precompute strings are set to zero
            diag_precompute_string = ''
            diag_precompute_replace = ''
        else:
            # for multioutput we need to extract the index of the output form the input.
            diag_precompute_string = precompute_list[0]
            diag_precompute_replace = precompute_list[1]
        

        # Here's the code to do the looping for K
        self._K_code =\
        """
        // _K_code
        // Code for computing the covariance function.
        int i;
        int j;
        int N = target_array->dimensions[0];
        int num_inducing = target_array->dimensions[1];
        int input_dim = X_array->dimensions[1];
        //#pragma omp parallel for private(j)
        for (i=0;i<N;i++){
            for (j=0;j<num_inducing;j++){
%s
                //target[i*num_inducing+j] = 
                TARGET2(i, j) += k(%s);
            }
        }
        %s
        """%(precompute_string,arg_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed

        self._K_code_X = """
        // _K_code_X
        // Code for computing the covariance function.
        int i;
        int j;
        int N = target_array->dimensions[0];
        int num_inducing = target_array->dimensions[1];
        int input_dim = X_array->dimensions[1];
        //#pragma omp parallel for private(j)
        for (i=0;i<N;i++){
            %s // int ii=(int)X2(i, 1);
            TARGET2(i, i) += k(%s);
            for (j=0;j<i;j++){
              %s //int jj=(int)X2(j, 1);
              double kval = k(%s); //double kval = k(X2(i, 0), shared_lengthscale, LENGTHSCALE1(ii), SCALE1(ii));
              TARGET2(i, j) += kval;
              TARGET2(j, i) += kval;
            }
        }
        /*%s*/
        """%(diag_precompute_string, diag_arg_string, re.sub('Z2', 'X2', diag_precompute_replace), X_arg_string,str(self._sp_k)) #adding a string representation forces recompile when needed

        # Code to do the looping for Kdiag
        self._Kdiag_code =\
        """
        // _Kdiag_code
        // Code for computing diagonal of covariance function.
        int i;
        int N = target_array->dimensions[0];
        int input_dim = X_array->dimensions[1];
        //#pragma omp parallel for
        for (i=0;i<N;i++){
                %s
                //target[i] =
                TARGET1(i)=k(%s);
        }
        %s
        """%(diag_precompute_string,diag_arg_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed

        # Code to compute gradients
        grad_func_list = []
        if self.output_dim>1:
            grad_func_list += c_define_output_indices
            grad_func_list += [' '*16 + 'TARGET1(%i+ii) += PARTIAL2(i, j)*dk_d%s(%s);'%(self.num_shared_params+i*self.output_dim, theta.name, arg_string) for i, theta in enumerate(self._sp_theta_i)]
            grad_func_list += [' '*16 + 'TARGET1(%i+jj) += PARTIAL2(i, j)*dk_d%s(%s);'%(self.num_shared_params+i*self.output_dim, theta.name, reverse_arg_string) for i, theta in enumerate(self._sp_theta_i)]
        grad_func_list += ([' '*16 + 'TARGET1(%i) += PARTIAL2(i, j)*dk_d%s(%s);'%(i,theta.name,arg_string) for i,theta in  enumerate(self._sp_theta)])
        grad_func_string = '\n'.join(grad_func_list) 

        self._dK_dtheta_code =\
        """
        // _dK_dtheta_code
        // Code for computing gradient of covariance with respect to parameters.
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
        """%(grad_func_string,"/*"+str(self._sp_k)+"*/") # adding a string representation forces recompile when needed


        # Code to compute gradients for Kdiag TODO: needs clean up
        diag_grad_func_string = re.sub('Z','X',grad_func_string,count=0)
        diag_grad_func_string = re.sub('int jj','//int jj',diag_grad_func_string)
        diag_grad_func_string = re.sub('j','i',diag_grad_func_string)
        diag_grad_func_string = re.sub('PARTIAL2\(i, i\)','PARTIAL1(i)',diag_grad_func_string)
        self._dKdiag_dtheta_code =\
        """
        // _dKdiag_dtheta_code
        // Code for computing gradient of diagonal with respect to parameters.
        int i;
        int N = partial_array->dimensions[0];
        int input_dim = X_array->dimensions[1];
        for (i=0;i<N;i++){
                %s
        }
        %s
        """%(diag_grad_func_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed

        # Code for gradients wrt X, TODO: may need to deal with special case where one input is actually an output.
        gradX_func_list = []
        if self.output_dim>1:
            gradX_func_list += c_define_output_indices
        gradX_func_list += ["TARGET2(i, %i) += PARTIAL2(i, j)*dk_dx_%i(%s);"%(q,q,arg_string) for q in range(self._real_input_dim)]
        gradX_func_string = "\n".join(gradX_func_list)

        self._dK_dX_code = \
        """
        // _dK_dX_code
        // Code for computing gradient of covariance with respect to inputs.
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
        """%(gradX_func_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed
  

        diag_gradX_func_string = re.sub('Z','X',gradX_func_string,count=0)
        diag_gradX_func_string = re.sub('int jj','//int jj',diag_gradX_func_string)
        diag_gradX_func_string = re.sub('j','i',diag_gradX_func_string)
        diag_gradX_func_string = re.sub('PARTIAL2\(i, i\)','2*PARTIAL1(i)',diag_gradX_func_string)

        # Code for gradients of Kdiag wrt X
        self._dKdiag_dX_code= \
        """
        // _dKdiag_dX_code
        // Code for computing gradient of diagonal with respect to inputs.
        int N = partial_array->dimensions[0];
        int input_dim = X_array->dimensions[1];
        for (int i=0;i<N; i++){
            %s
        }
        %s
        """%(diag_gradX_func_string,"/*"+str(self._sp_k)+"*/") #adding a
        # string representation forces recompile when needed Get rid
        # of Zs in argument for diagonal. TODO: Why wasn't
        # diag_func_string called here? Need to check that.
        #self._dKdiag_dX_code = self._dKdiag_dX_code.replace('Z[j', 'X[i')

        # Code to use when only X is provided. 
        self._dK_dtheta_code_X = self._dK_dtheta_code.replace('Z[', 'X[')
        self._dK_dX_code_X = self._dK_dX_code.replace('Z[', 'X[').replace('+= PARTIAL2(', '+= 2*PARTIAL2(') 
        self._dK_dtheta_code_X = self._dK_dtheta_code_X.replace('Z2(', 'X2(')
        self._dK_dX_code_X = self._dK_dX_code_X.replace('Z2(', 'X2(')


        #TODO: insert multiple functions here via string manipulation
        #TODO: similar functions for psi_stats
    def _get_arg_names(self, Z=None, partial=None):
        arg_names = ['target','X']
        for shared_params in self._sp_theta:
            arg_names += [shared_params.name]
        if Z is not None:
            arg_names += ['Z']
        if partial is not None:
            arg_names += ['partial']
        if self.output_dim>1:
            arg_names += self._split_theta_names
            arg_names += ['output_dim']
        return arg_names
        
    def _weave_inline(self, code, X, target, Z=None, partial=None):
        output_dim = self.output_dim
        for shared_params in self._sp_theta:
            locals()[shared_params.name] = getattr(self, shared_params.name)

        # Need to extract parameters first
        for split_params in self._split_theta_names:
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
        self._weave_inline(self._dKdiag_dX_code, X, target, Z=None, partial=partial)

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


    def _set_params(self,param):        
        assert param.size == (self.num_params)
        for i, shared_params in enumerate(self._sp_theta):
            setattr(self, shared_params.name, param[i])
            
        if self.output_dim>1:
            for i, split_params in enumerate(self._split_theta_names):
                start = self.num_shared_params + i*self.output_dim
                end = self.num_shared_params + (i+1)*self.output_dim
                setattr(self, split_params, param[start:end])


    def _get_params(self):
        params = np.zeros(0)
        for shared_params in self._sp_theta:
            params = np.hstack((params, getattr(self, shared_params.name)))
        if self.output_dim>1:
            for split_params in self._split_theta_names:
                params = np.hstack((params, getattr(self, split_params).flatten()))
        return params

    def _get_param_names(self):
        if self.output_dim>1:
            return [x.name for x in self._sp_theta] + [x.name[:-2] + str(i)  for x in self._sp_theta_i for i in range(self.output_dim)]
        else:
            return [x.name for x in self._sp_theta]
