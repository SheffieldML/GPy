#include <cblas.h>
void chol_backprop(int N, double* dL, double* L){
    //at the input to this fn, dL is df_dL. after this fn is complet, dL is df_dK
    int i,k;

    dL[N*N - 1] /= (2. * L[N*N - 1]);
    for(k=N-2;k>(-1);k--){
        cblas_dsymv(CblasRowMajor, CblasLower,
	            N-k-1, -1,
		    &dL[(N*(k+1) + k+1)],N,
		    &L[k*N+k+1],1,
		    1, &dL[N*(k+1)+k], N);
        for(i=0;i<(N-k-1); i++){
		dL[N*(k+1+i)+k] -= dL[N*(k+1)+k+i*(N+1)+1] * L[k*N+k+1+i];
	}

	cblas_dscal(N-k-1, 1.0/L[k*N+k], &dL[(k+1)*N+k], N);
        dL[k*N + k] -= cblas_ddot(N-k-1, &dL[(k+1)*N+k], N, &L[k*N+k+1], 1);
        dL[k*N + k] /= (2.0 * L[k*N + k]);
    }
}

double mydot(int n, double* a,  int stride_a, double* b, int stride_b){
    double ret = 0;	
    for(int i=0; i<n; i++){
	ret += a[i*stride_a]*b[i*stride_b];
    }
    return ret;
}
void old_chol_backprop(int N, double* dL, double* U){
    //at the input to this fn, dL is df_dL. after this fn is complet, dL is df_dK
    int iN, kN,i,j,k;
    dL[N*N-1] /= (2. * U[N*N-1]);
    for(k=N-2;k>(-1);k--){
	kN = k*N;
        #pragma omp parallel for private(i,iN)
        for(i=k+1; i<N; i++){
	    iN = i*N;
	    dL[iN+k] -= mydot(i-k, &dL[iN+k+1], 1, &U[kN+k+1], 1);
	    dL[iN+k] -= mydot(N-i, &dL[iN+i], N, &U[kN+i],  1);

	}
        for(i=(k + 1); i<N; i++){
	    iN = i*N;
            dL[iN + k] /= U[kN + k];
            dL[kN + k] -= U[kN + i] * dL[iN + k];
	}
        dL[kN + k] /= (2. * U[kN + k]);
    }
}

