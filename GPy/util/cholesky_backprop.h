#include <cblas.h>

void dsymv(int N, double*A, double*b, double*y);
double mydot(int n, double* a,  int stride_a, double* b, int stride_b);
void chol_backprop(int N, double* dL, double* L);
