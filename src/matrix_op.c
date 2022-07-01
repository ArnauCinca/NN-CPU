#include <stdio.h>
#include "matrix_op.h"
void mat_copy(int x, int y, double **m1, double **m2){
	for(int i = 0; i < x; i++){
		for(int j = 0; j < y; j++){
			m2[i][j] = m1[i][j];
		}
	}
}

void mat_mult(int x, int y, int z, double** A, double** B, double** res){
	for(int i = 0; i < x; i++){
		for(int j = 0; j < z; j++){
			res[i][j] = 0;
			for(int k = 0; k < y; k++){
				res[i][j] += A[i][k] * B[k][j]; 
			}
		}
	}
}



void mat_transpose(int x, int y, double** A,  double** res){
	for(int i = 0; i < x; i++){
		for(int j = 0; j < y; j++){
			res[i][j] = A[j][y];
		}
	}
}
