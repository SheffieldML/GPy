# Check Matthew Rocklin's blog post.
try: 
    import sympy as sp
    sympy_available=True
except ImportError:
    sympy_available=False
    exit()

from sympy.core.cache import clear_cache
from sympy.utilities.codegen import codegen

try:
    from scipy import weave
    weave_available = True
except ImportError:
    weave_available = False

import os
current_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
import sys
import numpy as np
import re
import tempfile
import pdb
import ast

from kernpart import Kernpart
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
# TODO have this set up in a set up file!
user_code_storage = tempfile.gettempdir()

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
            name='sympykern'
        if k is None:
            raise ValueError, "You must provide an argument for the covariance function."
        super(spkern, self).__init__(input_dim, name)

        self._sp_k = k

        # pull the variable names out of the symbolic covariance function.
        sp_vars = [e for e in k.atoms() if e.is_Symbol]
        self._sp_x= sorted([e for e in sp_vars if e.name[0:2]=='x_'],key=lambda x:int(x.name[2:]))
        self._sp_z= sorted([e for e in sp_vars if e.name[0:2]=='z_'],key=lambda z:int(z.name[2:]))

        # Check that variable names make sense.
        assert all([x.name=='x_%i'%i for i,x in enumerate(self._sp_x)])
        assert all([z.name=='z_%i'%i for i,z in enumerate(self._sp_z)])
        assert len(self._sp_x)==len(self._sp_z)
        x_dim=len(self._sp_x)

        # If it is a multi-output covariance, add an input for indexing the outputs.
        self._real_input_dim = x_dim
        # Check input dim is number of xs + 1 if output_dim is >1
        assert self.input_dim == x_dim + int(output_dim > 1)
        self.output_dim = output_dim

        # extract parameter names from the covariance
        thetas = sorted([e for e in sp_vars if not (e.name[0:2]=='x_' or e.name[0:2]=='z_')],key=lambda e:e.name)


        # Look for parameters with index (subscripts), they are associated with different outputs.
        if self.output_dim>1:
            self._sp_theta_i = sorted([e for e in thetas if (e.name[-2:]=='_i')], key=lambda e:e.name)
            self._sp_theta_j = sorted([e for e in thetas if (e.name[-2:]=='_j')], key=lambda e:e.name)

            # Make sure parameter appears with both indices!
            assert len(self._sp_theta_i)==len(self._sp_theta_j)
            assert all([theta_i.name[:-2]==theta_j.name[:-2] for theta_i, theta_j in zip(self._sp_theta_i, self._sp_theta_j)])

            # Extract names of shared parameters (those without a subscript)
            self._sp_theta = [theta for theta in thetas if theta not in self._sp_theta_i and theta not in self._sp_theta_j]
            
            self.num_split_params = len(self._sp_theta_i)
            self._split_theta_names = ["%s"%theta.name[:-2] for theta in self._sp_theta_i]
            for theta in self._split_theta_names:
                setattr(self, theta, Param(theta, np.ones(self.output_dim), None))
                self.add_parameters(getattr(self, theta))

                #setattr(self, theta, np.ones(self.output_dim))
            
            self.num_shared_params = len(self._sp_theta)
            #self.num_params = self.num_shared_params+self.num_split_params*self.output_dim
            
        else:
            self.num_split_params = 0
            self._split_theta_names = []
            self._sp_theta = thetas
            self.num_shared_params = len(self._sp_theta)
            #self.num_params = self.num_shared_params

        # Add parameters to the model.
        for theta in self._sp_theta:
            val = 1.0
            if param is not None:
                if param.has_key(theta):
                    val = param[theta]
            #setattr(self, theta.name, val)
            setattr(self, theta.name, Param(theta.name, val, None))
            self.add_parameters(getattr(self, theta.name))
        #deal with param            
        #self._set_params(self._get_params())

        # Differentiate with respect to parameters.
        self._sp_dk_dtheta = [sp.diff(k,theta).simplify() for theta in self._sp_theta]
        if self.output_dim > 1:
            self._sp_dk_dtheta_i = [sp.diff(k,theta).simplify() for theta in self._sp_theta_i]

        # differentiate with respect to input variables.
        self._sp_dk_dx = [sp.diff(k,xi).simplify() for xi in self._sp_x]

        # psi_stats aren't yet implemented.
        if False:
            self.compute_psi_stats()

        self._code = {}

        # generate the code for the covariance functions
        self._gen_code()

        if weave_available:
            if False:
                extra_compile_args = ['-ftree-vectorize', '-mssse3', '-ftree-vectorizer-verbose=5']
            else:
                extra_compile_args = []
            
                self.weave_kwargs = {
                    'support_code': None, #self._function_code,
                    'include_dirs':[user_code_storage, os.path.join(current_dir,'parts/')],
                    'headers':['"sympy_helpers.h"', '"'+self.name+'.h"'],
                    'sources':[os.path.join(current_dir,"parts/sympy_helpers.cpp"), os.path.join(user_code_storage, self.name+'.cpp')],
                    'extra_compile_args':extra_compile_args,
                    'extra_link_args':['-lgomp'],
                    'verbose':True}
        self.parameters_changed() # initializes caches


    def __add__(self,other):
        return spkern(self._sp_k+other._sp_k)

    def _gen_code(self):

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
        # generate c functions from sympy objects
        if weave_available:
            code_type = "C"
        else:
            code_type = "PYTHON"
        # Need to add the sympy_helpers header in here.
        (foo_c,self._function_code), (foo_h,self._function_header) = \
                                     codegen(code_list,
                                             code_type,
                                             self.name,
                                             argument_sequence=argument_sequence)


        # Use weave to compute the underlying functions.
        if weave_available:
            # put the header file where we can find it
            f = file(os.path.join(user_code_storage, self.name + '.h'),'w')
            f.write(self._function_header)
            f.close()
    

        if weave_available:
            # Substitute any known derivatives which sympy doesn't compute
            self._function_code = re.sub('DiracDelta\(.+?,.+?\)','0.0',self._function_code)
            # put the cpp file in user code storage (defaults to temp file location)
            f = file(os.path.join(user_code_storage, self.name + '.cpp'),'w')
        else:
            # put the python file in user code storage
            f = file(os.path.join(user_code_storage, self.name + '.py'),'w')
        f.write(self._function_code)
        f.close()

        if weave_available:
            # arg_list will store the arguments required for the C code.
            input_arg_list = (["X2(i, %s)"%x.name[2:] for x in self._sp_x]
                        + ["Z2(j, %s)"%z.name[2:] for z in self._sp_z])

            # for multiple outputs reverse argument list is also required
            if self.output_dim>1:
                reverse_input_arg_list = list(input_arg_list)
                reverse_input_arg_list.reverse()

            # This gives the parameters for the arg list.
            param_arg_list = [shared_params.name for shared_params in self._sp_theta]
            arg_list = input_arg_list + param_arg_list

            precompute_list=[]
            if self.output_dim > 1:
                reverse_arg_list= reverse_input_arg_list + list(param_arg_list)
                # For multiple outputs, also need the split parameters.
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

            # Now we use the arguments in code that computes the separate parts.

            # Any precomputations will be done here eventually.
            self._precompute = \
                """
                // Precompute code would go here. It will be called when parameters are updated. 
                """

            # Here's the code to do the looping for K
            self._code['K'] =\
            """
            // _K_code
            // Code for computing the covariance function.
            int i;
            int j;
            int n = target_array->dimensions[0];
            int num_inducing = target_array->dimensions[1];
            int input_dim = X_array->dimensions[1];
            //#pragma omp parallel for private(j)
            for (i=0;i<n;i++){
                for (j=0;j<num_inducing;j++){
                    %s
                    //target[i*num_inducing+j] = 
                    TARGET2(i, j) += k(%s);
                }
            }
            %s
            """%(precompute_string,arg_string,"/*"+str(self._sp_k)+"*/")
           # adding a string representation of the function in the
           # comment forces recompile when needed
            self._code['K_X'] = self._code['K'].replace('Z2(', 'X2(')


            # Code to compute diagonal of covariance.
            diag_arg_string = re.sub('Z','X',arg_string)
            diag_arg_string = re.sub('int jj','//int jj',diag_arg_string)
            diag_arg_string = re.sub('j','i',diag_arg_string)
            diag_precompute_string = re.sub('int jj','//int jj',precompute_string)
            diag_precompute_string = re.sub('Z','X',diag_precompute_string)
            diag_precompute_string = re.sub('j','i',diag_precompute_string)
            # Code to do the looping for Kdiag
            self._code['Kdiag'] =\
            """
            // _code['Kdiag']
            // Code for computing diagonal of covariance function.
            int i;
            int n = target_array->dimensions[0];
            int input_dim = X_array->dimensions[1];
            //#pragma omp parallel for
            for (i=0;i<n;i++){
                    %s
                    //target[i] =
                    TARGET1(i)=k(%s);
            }
            %s
            """%(diag_precompute_string,diag_arg_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed

            # Code to compute gradients
            if self.output_dim>1:
                for i, theta in enumerate(self._sp_theta_i):
                    grad_func_list = [' '*26 + 'TARGET1(ii) += PARTIAL2(i, j)*dk_d%s(%s);'%(theta.name, arg_string)]
                    grad_func_list += [' '*26 + 'TARGET1(jj) += PARTIAL2(i, j)*dk_d%s(%s);'%(theta.name, reverse_arg_string)]
                    grad_func_list = c_define_output_indices+grad_func_list

                    grad_func_string = '\n'.join(grad_func_list) 
                    self._code['dK_d' + theta.name] =\
                      """
                      int i;
                      int j;
                      int n = partial_array->dimensions[0];
                      int num_inducing = partial_array->dimensions[1];
                      int input_dim = X_array->dimensions[1];
                      //#pragma omp parallel for private(j)
                      for (i=0;i<n;i++){
                        for (j=0;j<num_inducing;j++){
%s
                        }
                      }
                      %s
                      """%(grad_func_string,"/*"+str(self._sp_k)+"*/") # adding a string representation forces recompile when needed
                    self._code['dK_d' +theta.name + '_X'] = self._code['dK_d' + theta.name].replace('Z2(', 'X2(')
                    # Code to compute gradients for Kdiag TODO: needs clean up
                    diag_grad_func_string = re.sub('Z','X',grad_func_string,count=0)
                    diag_grad_func_string = re.sub('int jj','//int jj',diag_grad_func_string)
                    diag_grad_func_string = re.sub('j','i',diag_grad_func_string)
                    diag_grad_func_string = re.sub('PARTIAL2\(i, i\)','PARTIAL(i)',diag_grad_func_string)
                    self._code['dKdiag_d' + theta.name] =\
                      """
                      // _dKdiag_dtheta_code
                      // Code for computing gradient of diagonal with respect to parameters.
                      int i;
                      int n = partial_array->dimensions[0];
                      int input_dim = X_array->dimensions[1];
                      for (i=0;i<n;i++){
                        %s
                      }
                      %s
                      """%(diag_grad_func_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed

            for i, theta in enumerate(self._sp_theta):
                grad_func_list = [' '*26 + 'TARGET1(%i) += PARTIAL2(i, j)*dk_d%s(%s);'%(i,theta.name,arg_string)]
                grad_func_string = '\n'.join(grad_func_list) 

                self._code['dK_d' + theta.name] =\
                  """
                  // _dK_dtheta_code
                  // Code for computing gradient of covariance with respect to parameters.
                  int i;
                  int j;
                  int n = partial_array->dimensions[0];
                  int num_inducing = partial_array->dimensions[1];
                  int input_dim = X_array->dimensions[1];
                  //#pragma omp parallel for private(j)
                  for (i=0;i<n;i++){
                    for (j=0;j<num_inducing;j++){
                      %s
                    }
                  }
                  %s
                  """%(grad_func_string,"/*"+str(self._sp_k)+"*/") # adding a string representation forces recompile when needed
                self._code['dK_d' + theta.name +'_X'] = self._code['dK_d' + theta.name].replace('Z2(', 'X2(')
                # Code to compute gradients for Kdiag TODO: needs clean up
                diag_grad_func_string = re.sub('Z','X',grad_func_string,count=0)
                diag_grad_func_string = re.sub('int jj','//int jj',diag_grad_func_string)
                diag_grad_func_string = re.sub('j','i',diag_grad_func_string)
                diag_grad_func_string = re.sub('PARTIAL2\(i, i\)','PARTIAL(i)',diag_grad_func_string)
                self._code['dKdiag_d' + theta.name] =\
                   """
                   // _dKdiag_dtheta_code
                   // Code for computing gradient of diagonal with respect to parameters.
                   int i;
                   int n = partial_array->dimensions[0];
                   int input_dim = X_array->dimensions[1];
                   for (i=0;i<n;i++){
                     %s
                   }
                   %s
                   """%(diag_grad_func_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed



            # Code for gradients wrt X, TODO: may need to deal with special case where one input is actually an output.
            gradX_func_list = []
            if self.output_dim>1:
                gradX_func_list += c_define_output_indices
            gradX_func_list += ["TARGET2(i, %i) += partial[i*num_inducing+j]*dk_dx_%i(%s);"%(q,q,arg_string) for q in range(self._real_input_dim)]
            gradX_func_string = "\n".join(gradX_func_list)

            self._code['dK_dX'] = \
            """
            // _dK_dX_code
            // Code for computing gradient of covariance with respect to inputs.
            int i;
            int j;
            int n = partial_array->dimensions[0];
            int num_inducing = partial_array->dimensions[1];
            int input_dim = X_array->dimensions[1];
            //#pragma omp parallel for private(j)
            for (i=0;i<n; i++){
              for (j=0; j<num_inducing; j++){
                %s
              }
            }
            %s
            """%(gradX_func_string,"/*"+str(self._sp_k)+"*/") #adding a string representation forces recompile when needed
            self._code['dK_dX_X'] = self._code['dK_dX'].replace('Z2(', 'X2(')


            diag_gradX_func_string = re.sub('Z','X',gradX_func_string,count=0)
            diag_gradX_func_string = re.sub('int jj','//int jj',diag_gradX_func_string)
            diag_gradX_func_string = re.sub('j','i',diag_gradX_func_string)
            diag_gradX_func_string = re.sub('PARTIAL2\(i\, i\)','2*PARTIAL(i)',diag_gradX_func_string)

            # Code for gradients of Kdiag wrt X
            self._code['dKdiag_dX'] = \
            """
            // _dKdiag_dX_code
            // Code for computing gradient of diagonal with respect to inputs.
            int n = partial_array->dimensions[0];
            int input_dim = X_array->dimensions[1];
            for (int i=0;i<n; i++){
                %s
            }
            %s
            """%(diag_gradX_func_string,"/*"+str(self._sp_k)+"*/") #adding a
            # string representation forces recompile when needed Get rid
            # of Zs in argument for diagonal. TODO: Why wasn't
            # diag_func_string called here? Need to check that.

            


            #TODO: insert multiple functions here via string manipulation
            #TODO: similar functions for psi_stats
            #TODO: similar functions when cython available.
            #TODO: similar functions when only python available.
            
    def _get_arg_names(self, target=None, Z=None, partial=None):
        arg_names = ['X']
        if target is not None:
            arg_names += ['target']
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

    def _generate_inline(self, code, X, target=None, Z=None, partial=None):
        output_dim = self.output_dim
        # Need to extract parameters to local variables first
        for shared_params in self._sp_theta:
            locals()[shared_params.name] = getattr(self, shared_params.name)
            
        for split_params in self._split_theta_names:
            locals()[split_params] = np.asarray(getattr(self, split_params))
        arg_names = self._get_arg_names(target, Z, partial)        

        if weave_available:
            return weave.inline(code=code, arg_names=arg_names,**self.weave_kwargs)
        else:
            raise RuntimeError('Weave not available and other variants of sympy covariance not yet implemented')

    def K(self,X,Z,target):        
        if Z is None:
            self._generate_inline(self._code['K_X'], X, target)
        else:
            self._generate_inline(self._code['K'], X, target, Z)


    def Kdiag(self,X,target):
        self._generate_inline(self._code['Kdiag'], X, target)

    def _param_grad_helper(self,partial,X,Z,target):
        if Z is None:
            self._generate_inline(self._code['dK_dtheta_X'], X, target, Z, partial)
        else:
            self._generate_inline(self._code['dK_dtheta'], X, target, Z, partial)

    def dKdiag_dtheta(self,partial,X,target):
        self._generate_inline(self._code['dKdiag_dtheta'], X, target, Z=None, partial=partial).namelocals()[shared_params.name] = getattr(self, shared_params.name)
               
    def gradients_X(self,partial,X,Z,target):
        if Z is None:
            self._generate_inline(self._code['dK_dX_X'], X, target, Z, partial)
        else:
            self._generate_inline(self._code['dK_dX'], X, target, Z, partial)

    def dKdiag_dX(self,partial,X,target):
        self._generate_inline(self._code['dKdiag_dX'], X, target, Z, partial)

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

    def parameters_changed(self):
        # Reset the caches
        self._cache, self._cache2 = np.empty(shape=(2, 1))
        self._cache3, self._cache4, self._cache5 = np.empty(shape=(3, 1)) 

    def update_gradients_full(self, dL_dK, X):
        # Need to extract parameters to local variables first
        self._K_computations(X, None)
        for shared_params in self._sp_theta:
            parameter = getattr(self, shared_params.name)
            code = self._code['dK_d' + shared_params.name]
            setattr(parameter, 'gradient', self._generate_inline(code, X, target=None, Z=None, partial=dL_dK))
            
        for split_params in self._split_theta_names:
            parameter = getattr(self, split_params.name)
            code = self._code['dK_d' + split_params.name]
            setattr(parameter, 'gradient', self._generate_inline(code, X, target=None, Z=None, partial=dL_dK))
    

    # def update_gradients_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
    #     #contributions from Kdiag
    #     self.variance.gradient = np.sum(dL_dKdiag)
        
    #     #from Knm
    #     self._K_computations(X, Z)
    #     self.variance.gradient += np.sum(dL_dKnm * self._K_dvar)
    #     if self.ARD:
    #         self.lengthscale.gradient = self._dL_dlengthscales_via_K(dL_dKnm, X, Z)
            
    #     else:
    #         self.lengthscale.gradient = (self.variance / self.lengthscale) * np.sum(self._K_dvar * self._K_dist2 * dL_dKnm)
            
    #     #from Kmm
    #     self._K_computations(Z, None)
    #     self.variance.gradient += np.sum(dL_dKmm * self._K_dvar)
    #     if self.ARD:
    #         self.lengthscale.gradient += self._dL_dlengthscales_via_K(dL_dKmm, Z, None)
    #     else:
    #         self.lengthscale.gradient += (self.variance / self.lengthscale) * np.sum(self._K_dvar * self._K_dist2 * dL_dKmm)
            
            
    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    def _K_computations(self, X, Z):
        if Z is None:
            self._generate_inline(self._precompute, X)
        else:
            self._generate_inline(self._precompute, X, Z=Z)
