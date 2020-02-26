#include <stdio.h>
#include "matrixOp.h"

void matMultOp(int i, int j, int k, double** A, double** B, double** res){
	//res[x][y] = A[x][0]*B[0][y]+ A[x][1]*B[1][y]+...+A[x][j-1]*B[j-1][y]
	for(int x = 0; x < i; ++x){
		for(int y = 0; y < k; ++y){
			res[y][x] = 0;//dotProduct(j, A[x], B[y]);
			for(int z = 0; z < j; ++z){//tree
				res[y][x] += A[x][z] * B[y][z];  //transpose B 
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
