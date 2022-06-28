#include "activation_function.h"
#include "../vector_op/vector_op.h"
#include <math.h>
#include <stdlib.h>

void sigmoid_act(double* x, double* res){
	res[0] = 1/(1+exp(-x[0]));
}
void sigmoid_act_prime(double* x, double* res){
	res[0] = (1/(1+exp(-x[0])))*(1-(1/(1+exp(-x[0])))); //TODO: use f(x) not x
}


void ReLU_act(double* x, double* res){
	res[0] = fmax(0.0,x[0]);
}
void ReLU_act_prime(double* x, double* res){
	res[0] = x[0]>0.0;
}


void identity_act(double* x, double* res){
	res[0] = x[0];
}
void identity_act_prime(double* x, double* res){
	res[0] = 1;
}


void tanh_act(double* x, double* res){
	res[0] = (2/(1+exp(-2*x[0])))+1;
}
void tanh_act_prime(double* x, double* res){
	res[0] = 1-x[0]*x[0];
}


activation_function_t *sigmoid(){
	struct activation_function_t* af = malloc(sizeof(activation_function_t));
	af->aux = NULL;
	af->act = sigmoid_act;
	af->act_prime = sigmoid_act_prime;
	return af;
}
activation_function_t* ReLU(){
	struct activation_function_t* af = malloc(sizeof(activation_function_t));
	af->aux = NULL;
	af->act = ReLU_act;
	af->act_prime = ReLU_act_prime;
	return af;

}
activation_function_t* identity(){
	struct activation_function_t* af = malloc(sizeof(activation_function_t));
	af->aux = NULL;
	af->act = identity_act;
	af->act_prime = identity_act_prime;
	return af;

}
activation_function_t* tanH(){
	struct activation_function_t* af = malloc(sizeof(activation_function_t));
	af->aux = NULL;
	af->act = tanh_act;
	af->act_prime = tanh_act_prime;
	return af;

}
