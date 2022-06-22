#include "optimizer.h"
#include "../vector_op/vector_op.h"
#include <stdlib.h>
#include <stdio.h>


void sgd_optimizer(layer_t *l, loss_function_t *loss, double lr, int batch, double ***outs, double **realOuts, double ***deltas, double *tmp){
	
	//backprop output layer
	l->backpropagation_output(l, batch, outs[l->index], deltas[l->index], tmp, loss, realOuts);
	l = l->prev;
	//hidden Layers
	while(l->prev != NULL){ //backprop
		l->backpropagation(l, batch, outs[l->index], deltas[l->index], deltas[l->next->index], tmp);
		l = l->prev;
	}
	//update weights
	l = l->next; //first layer before "input layer"
	while (l->prev != NULL){//gradientDescent
		l->gradient_descent(l, batch, lr, outs[l->prev->index], outs[l->index], deltas[l->index]);
		l = l->next;
	}	
}

optimizer_t* sgd(){
	struct optimizer_t* o = calloc(1,sizeof(optimizer_t));
	o->optimize = sgd_optimizer;
	return o;


}
