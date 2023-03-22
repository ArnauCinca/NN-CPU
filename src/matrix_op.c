#include <stdio.h>
#include <math.h>
#include "matrix_op.h"


#define BS 32



void mat_copy(int x, int y, double **m1, double **m2){
	for(int i = 0; i < x; i++){
		for(int j = 0; j < y; j++){
			m2[i][j] = m1[i][j];
		}
	}
}

void mat_mult(int x, int y, int z, double** A, double** B, double** res){
	int xx, yy, zz;

	for(int ii = 0; ii < x; ii+=BS){
		xx = fmin(ii+BS,x);
		for(int jj = 0; jj < z; jj+=BS){
			zz = fmin(jj+BS,z);


			for(int i = ii; i < xx; i++){
				for(int j = jj; j < zz; j++){
					res[i][j] = A[i][0] * B[0][j]; 
				}
			}

			for(int kk = 1; kk < y; kk+=BS){
				yy = fmin(kk+BS,y);

				for(int i = ii; i < xx; i++){
					for(int j = jj; j < zz; j++){
						for(int k = kk; k < yy; k++){
							res[i][j] += A[i][k] * B[k][j]; 
						}
					}
				}

			}
		}
	}
}

void mat_add(int x, int y, double** A, double** B, double** res){ //A[i][j] + B[i][j] = res[i][j]
	for(int i = 0; i < x; i++){
		for(int j = 0; j < y; j++){
			res[i][j] = A[i][j] + B[i][j];
		}
	}
}



void mat_transpose(int x, int y, double** A,  double** res){
	for(int i = 0; i < x; i++){
		for(int j = 0; j < y; j++){
			res[j][i] = A[i][j];
		}
	}
}
