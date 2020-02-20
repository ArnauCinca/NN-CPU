#include "ActivationFunction.h"
#include "../vectorOp/vectorOp.h"
#include <math.h>
#include <stdlib.h>

void sigmoidAct(double* x, double* res){
	res[0] = 1/(1+exp(-x[0]));
}
void sigmoidPAct(double* x, double* res){
	res[0] = x[0]*(1-x[0]);
}


void ReLUAct(double* x, double* res){
	res[0] = fmax(0.0,x[0]);
}
void ReLUPAct(double* x, double* res){
	res[0] = x[0]>0.0;
}


void identityAct(double* x, double* res){
	res[0] = x[0];
}
void identityPAct(double* x, double* res){
	res[0] = 1;
}


void tanhAct(double* x, double* res){
	res[0] = (2/(1+exp(-2*x[0])))+1;
}
void tanhPAct(double* x, double* res){
	res[0] = 1-x[0]*x[0];
}


ActivationFunction* sigmoid(){
	struct ActivationFunction* af = malloc(sizeof(ActivationFunction));
	af->aux = NULL;
	af->act = sigmoidAct;
	af->actPrime = sigmoidPAct;
	return af;
}
ActivationFunction* ReLU(){
	struct ActivationFunction* af = malloc(sizeof(ActivationFunction));
	af->aux = NULL;
	af->act = ReLUAct;
	af->actPrime = ReLUPAct;
	return af;

}
ActivationFunction* identity(){
	struct ActivationFunction* af = malloc(sizeof(ActivationFunction));
	af->aux = NULL;
	af->act = identityAct;
	af->actPrime = identityPAct;
	return af;

}
ActivationFunction* tanH(){
	struct ActivationFunction* af = malloc(sizeof(ActivationFunction));
	af->aux = NULL;
	af->act = tanhAct;
	af->actPrime = tanhPAct;
	return af;

}
