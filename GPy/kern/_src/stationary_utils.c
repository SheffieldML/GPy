void _grad_X(int N, int D, int M, double* X, double* X2, double* tmp, double* grad){
int n,m,d;
double retnd;
#pragma omp parallel for private(n,d, retnd, m)
for(d=0;d<D;d++){
  for(n=0;n<N;n++){
    retnd = 0.0;
    for(m=0;m<M;m++){
      retnd += tmp[n*M+m]*(X[n*D+d]-X2[m*D+d]);
    }
    grad[n*D+d] = retnd;
  }
}
} //grad_X

//#weave_options = {'headers'           : ['<omp.h>'],
         //'extra_compile_args': ['-fopenmp -O3'], # -march=native'],
         //'extra_link_args'   : ['-lgomp']}


