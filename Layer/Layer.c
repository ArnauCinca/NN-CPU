#include "Layer.h"
#include "../vectorOp/vectorOp.h"
#include "../matrixOp/matrixOp.h"
#include <stdlib.h>
#include <stdio.h>

Layer* getFirstLayer(Layer* me){
	if(me->prev == NULL) return me;
	Layer* l = me->prev;
	while(l->prev != NULL) l = l->prev;
	return l;
}
Layer* getILayer(Layer* me, int i){
	Layer* l = getFirstLayer(me);
	for(;i>0; --i) l = l->next;
	return l;
}
Layer* getLastLayer(Layer* me){
	if(me->next == NULL) return me;
	Layer* l = me->next;
	while(l->next != NULL) l = l->next;
	return l;
}

void inputForward(Layer* me, double* in, double* out){
	copy(me->shape[0], in, out);
}

void denseForward(Layer* me, double* in, double* out){
	int i = me->shape[0];
	int size = me->prev->shape[0];
	//matMultOp(i,size,1, me->weights, &in, &out);
	for(;i>0;--i) {
		out[i-1] = dotProduct(size, in, me->weights[i-1]) + me->weights[i-1][size];  //dot product
		//out[i-1] +=  me->weights[i-1][size];  //dot product
	}
	map(me->shape[0], me->act->act, out, out); // activation Function
}


void denseBackprop(Layer* me, double** outs, double** deltas, double* tmp){
	map(me->shape[0], me->act->actPrime, outs[me->index], tmp);
    for(int i = 0; i<me->shape[0]; ++i){  //matmul
       	deltas[me->index][i] = 0;
		for(int j = 0; j < me->next->shape[0]; ++j){
			deltas[me->index][i] += deltas[me->index+1][j] * me->next->weights[j][i];
		}
	}
	mult(me->shape[0],tmp,deltas[me->index],deltas[me->index]);
}


void denseBackpropOutput(Layer* me, double** outs, double** deltas, double* tmp, LossFunction* loss, double* realOuts){
	loss->lossPrime(me->shape[0], outs[me->index], realOuts, deltas[me->index]);//loss	
	map(me->shape[0], me->act->actPrime, outs[me->index], tmp);
	mult(me->shape[0], tmp, deltas[me->index], deltas[me->index]);// loss*actP
}
void denseGradientDescent(Layer* me, double lr,  double** outs, double** deltas){
	for (int i = 0; i< me->shape[0]; ++i){
		for(int j = 0; j<me->prev->shape[0]; ++j){
			me->weights[i][j] -= lr*deltas[me->index][i]*outs[me->index-1][j];//weights
		}
		me->weights[i][me->prev->shape[0]] -= lr*deltas[me->index][i]; //bias
	}
}

Layer* Input(int dim, int* shape){	
	struct Layer* l = malloc(sizeof(Layer));
	l->prev = NULL;
	l->next = NULL;
	l->act = NULL;
	l->dim = dim;
	l->shape = malloc(dim*sizeof(int));
	for(int i = 0; i<dim; ++i){
		l->shape[i] = shape[i];
	}
	l->weights = NULL;
	l->forward = inputForward;
	return l;
	
}

Layer* Dense(Layer* in, int size, ActivationFunction* act){
	double** wb = malloc(size*sizeof(double*));
	for(int i = 0; i<size; ++i){
		wb[i] = malloc((in->shape[0]+1)*sizeof(double));
		randomInit((in->shape[0]+1),wb[i]);
	}
   
	struct Layer* l = malloc(sizeof(Layer));
	l->prev = in;
	l->next = NULL;
	l->act = act;
	l->dim = 1;
	l->shape = malloc(sizeof(int));
	l->shape[0] = size;
	l->weights = wb;
	l->forward = denseForward;
	l->backprop = denseBackprop;
	l->backpropOutput = denseBackpropOutput;
	l->gradientDescent = denseGradientDescent;
	l->index = in->index+1;	
	in->next = l;
	return l;
}


