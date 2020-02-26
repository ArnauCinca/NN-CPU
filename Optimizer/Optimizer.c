#include "Optimizer.h"
#include "../vectorOp/vectorOp.h"
#include <stdlib.h>
#include <stdio.h>


void sgdOptimizer(struct Layer* l, struct LossFunction* loss, double lr,  double** outs, double* realOuts, double** deltas, double* tmp){
	l = getLastLayer(l);
	
	//backprop output layer
	l->backpropOutput(l, outs, deltas, tmp, loss, realOuts);
	/*loss->lossPrime(l->shape[0], outs[l->index], realOuts, deltas[l->index]);//loss	
	map(l->shape[0], l->act->actPrime, outs[l->index], tmp);
	mult(l->shape[0], tmp, deltas[l->index], deltas[l->index]);// loss*actP
*/
	l = l->prev;
	//hidden Layers
	while(l->prev != NULL){ //backprop
		l->backprop(l, outs, deltas, tmp);
		l = l->prev;
	}
	//update weights
	l = l->next; //first layer before "input layer"
	while (l != NULL){//gradientDescent
		l->gradientDescent(l, lr, outs, deltas);
		l = l->next;
	}	
}

Optimizer* sgd(){
	struct Optimizer* o = malloc(sizeof(Optimizer));
	o->optimize = sgdOptimizer;
	return o;


}
