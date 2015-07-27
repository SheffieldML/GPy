void _grad_X(int N, int D, int M, double* X, double* X2, double* tmp, double* grad){
double retnd;
int n,d,nd,m;
#pragma omp parallel for private(nd,n,d, retnd, m)
for(nd=0;nd<(D*N);nd++){
  n = nd/D;
  d = nd%D;
  retnd = 0.0;
  for(m=0;m<M;m++){
    retnd += tmp[n*M+m]*(X[nd]-X2[m*D+d]);
  }
  grad[nd] = retnd;
}
} //grad_X


void _lengthscale_grads_unsafe(int N, int M, int Q, double* tmp, double* X, double* X2, double* grad){
int n,m,nm,q,nQ,mQ;
double dist;
#pragma omp parallel for private(n,m,nm,q,nQ,mQ,dist)
for(nm=0; nm<(N*M); nm++){
  n = nm/M;
  m = nm%M; 
  nQ = n*Q;
  mQ = m*Q;
  for(q=0; q<Q; q++){
    dist = X[nQ+q]-X2[mQ+q];
    grad[q] += tmp[nm]*dist*dist;
  }
}
} //lengthscale_grads


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






