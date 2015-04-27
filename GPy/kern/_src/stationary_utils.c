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


void _lengthscale_grads(int N, int M, int Q, double* tmp, double* X, double* X2, double* grad){
int n,m,q;
double gradq, dist;
#pragma omp parallel for private(n,m, gradq, dist)
for(q=0; q<Q; q++){
  gradq = 0;
  for(n=0; n<N; n++){
    for(m=0; m<M; m++){
        dist = X[n*Q+q]-X2[m*Q+q];
        gradq += tmp[n*M+m]*dist*dist;
    }
  }
  grad[q] = gradq;
}
} //lengthscale_grads




