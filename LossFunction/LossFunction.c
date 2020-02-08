#include "LossFunction.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
double mseLoss(int size, double* yhat, double* y){
	double res = 0;
  	for(int i = 0; i<size;++i){
    	res += pow((yhat[i] - y[i]),2);
  	}
	return res/size;
}
void msePLoss(int size, double* yhat, double* y, double* res){
	for(int i = 0; i<size; ++i){
		res[i] = yhat[i] - y[i];
	}
}



LossFunction* mse(){
	struct LossFunction* lf = malloc(sizeof(LossFunction));
	lf->aux = NULL;
	lf->loss = mseLoss; 
	lf->lossPrime = msePLoss;
	return lf;
}
