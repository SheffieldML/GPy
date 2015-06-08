#include <omp.h>
double mydot(int n, double* a,  int stride_a, double* b, int stride_b){
    double ret = 0;	
    for(int i=0; i<n; i++){
	ret += a[i*stride_a]*b[i*stride_b];
    }
    return ret;
}


void chol_backprop(int N, double* dL, double* U){
    //at the input to this fn, dL is df_dL. after this fn is complet, dL is df_dK
    int iN, kN;
    for(int k=N-1;k>(-1);k--){
	kN = k*N;
        #pragma omp parallel for private(iN)
        for(int i=k+1; i<N; i++){
	    iN = i*N;
	    dL[iN+k] -= mydot(i-k, &dL[iN+k+1], 1, &U[kN+k+1], 1);
	    dL[iN+k] -= mydot(N-i, &dL[iN+i], N, &U[kN+i],  1);

	}
        for(int i=(k + 1); i<N; i++){
	    iN = i*N;
            dL[iN + k] /= U[kN + k];
            dL[kN + k] -= U[kN + i] * dL[iN + k];
	}

        dL[kN + k] /= (2. * U[kN + k]);
    }
}

