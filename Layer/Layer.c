#include "Layer.h"
#include "../vectorOp/vectorOp.h"
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
	copy(me->size, in, out);
}
void denseForward(Layer* me, double* in, double* out){
	int i = me->size;
	int size = me->prev->size;
	for(;i>0;--i) {
		out[i-1] = dotProduct(size-1,in,me->weights[i-1]);+me->weights[i-1][size];  //dot product
	}
	me->act->act(me->size,out,out); // activation Function
}

Layer* Input(int dim, int* kernel_shape){	
	struct Layer* l = malloc(sizeof(Layer));
	l->prev = NULL;
	l->next = NULL;
	l->act = NULL;
	l->size = dim;
	l->dim = dim;
	l->kernel_shape = kernel_shape;
	l->weights = NULL;
	l->forward = inputForward;
	return l;
	
}

Layer* Dense(Layer* in, int size, ActivationFunction* act){
	double** wb = malloc(size*sizeof(double*));
	for(int i = 0; i<size; ++i){
		wb[i] = malloc((in->size+1)*sizeof(double));
		randomInit((in->size+1),wb[i]);
	}
   
	struct Layer* l = malloc(sizeof(Layer));
	l->prev = in;
	l->next = NULL;
	l->act = act;
	l->size = size;
	l->dim = 1;
	l->kernel_shape = NULL;
	l->weights = wb;
	l->forward = denseForward;
	l->index = in->index+1;	
	in->next = l;
	return l;
}


