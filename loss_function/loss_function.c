#include "loss_function.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
double mse_loss(int size, double* yhat, double* y){
	double res = 0;
  	for(int i = 0; i<size;++i){
    	res += pow((yhat[i] - y[i]),2);
  	}
	return res/size;
}
void mse_loss_prime(int size, double* yhat, double* y, double* res){
	for(int i = 0; i<size; ++i){
		res[i] = yhat[i] - y[i];
	}
}



loss_function_t *mse(){
	struct loss_function_t* lf = malloc(sizeof(loss_function_t));
	lf->aux = NULL;
	lf->loss = mse_loss; 
	lf->loss_prime = mse_loss_prime;
	return lf;
}
