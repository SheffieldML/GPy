
void chol_backprop(int N, double* dL, double* L){
    //at the input to this fn, dL is df_dL. after this fn is complet, dL is df_dK
    int i,j,k;
    for(k=N-1;k>(-1);k--){
        #pragma omp parallel for private(i,j)
        for(i=k+1;i<N; i++){
            for(j=k+1;j<(i+1);j++){
                dL[i*N + k] -= dL[i *N + j] * L[j*N + k];
	    }
            for(j=i;j<N;j++){
                dL[i*N + k] -= dL[j*N + i] * L[j*N +k];
	    }
	}
        for(i=k + 1; i<N; i++){
            dL[i*N + k] /= L[k*N + k];
            dL[k*N + k] -= L[i*N + k] * dL[i*N + k];
	}
        dL[k*N + k] /= (2. * L[k*N + k]);
    }
}

