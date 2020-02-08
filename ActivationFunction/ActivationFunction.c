#include "ActivationFunction.h"
#include <math.h>
#include <stdlib.h>

void sigmoidAct(int size, double* x, double* res){
  for(int i = 0; i<size;++i){
    res[i] = 1/(1+exp(-x[i]));
  }
}
void sigmoidPAct(int size, double* x, double* res){
  for(; size>0;--size){
    res[size-1] = x[size-1]*(1-x[size-1]);
  }
}


void ReLUAct(int size, double* x, double* res){
  for(int i = 0; i<size;++i){
    res[i] = fmax(0.0,x[i]);
  }
}
void ReLUPAct(int size, double* x, double* res){
  for(int i = 0; i<size;++i){
    res[i] = x[i]>0.0;
  }
}


void identityAct(int size, double* x, double* res){
  for(int i = 0; i<size;++i){
    res[i] = x[i];
  }
}
void identityPAct(int size, double* x, double* res){
  for(int i = 0; i<size;++i){
    res[i] = 1;
  }
}


void tanhAct(int size, double* x, double* res){
  for(int i = 0; i<size;++i){
    res[i] = (2/(1+exp(-2*x[i])))+1;
  }
}
void tanhPAct(int size, double* x, double* res){
  for(int i = 0; i<size;++i){
    res[i] = 1-x[i]*x[i];
  }
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
