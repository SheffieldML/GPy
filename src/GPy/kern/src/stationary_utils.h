#ifndef __APPLE__
#include <omp.h>
#endif
void _grad_X(int N, int D, int M, double*X, double* X2, double* tmp, double* grad);
void _lengthscale_grads(int N, int D, int M, double* X, double* X2, double* tmp, double* grad);
