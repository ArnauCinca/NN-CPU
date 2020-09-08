#include <stdio.h>
#include "matrixOp.h"

void matMultOp(int i, int j, int k, double** A, double** B, double** res){
	//i->batch, j->prev shape, k->shape, A->weight, B->in
	for(int x = 0; x < i; ++x){
		for(int y = 0; y < k; ++y){
			res[y][x] = 0;
			for(int z = 0; z < j; ++z){//tree
				res[x][y] += A[x][z] * B[z][y]; 
			}
		}
	}
}



void transposeOp(int i, int j, double** A,  double** res){
	for(int x = 0; x < i; ++x){
		for(int y = 0; y < j; ++y){
			res[x][y] = A[y][x];
		}
	}
}
