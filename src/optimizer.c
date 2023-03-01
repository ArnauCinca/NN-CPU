#include "optimizer.h"
#include "vector_op.h"
#include <stdlib.h>
#include <stdio.h>


void sgd_optimizer(layer_t *in, layer_t *out, loss_function_t *loss, double lr, int batch, double ***outs, double ***fouts, double **realOuts, double ***deltas){
	layer_t *l = out;	
//	printf("START\n");
	//backprop output layer
	l->backpropagation_output(l, batch, outs[l->index], fouts[l->index], deltas[l->index], loss, realOuts);
//	printf("backprop_out\n");
	//hidden Layers
		l = l->prev;
	do{
		l->backpropagation(l, batch, outs[l->index], deltas[l->index], deltas[l->next->index]);
		l = l->prev;
	} while(l != in); //backprop
//	printf("backpropt\n");
	//update weights

	do{
		l = l->next; //first layer before "input layer"
		l->gradient_descent(l, batch, lr, fouts[l->prev->index], deltas[l->index]);
	} while (l != out);//gradientDescent
//	printf("END\n");
}

optimizer_t* sgd(){
	struct optimizer_t* o = calloc(1,sizeof(optimizer_t));
	o->optimize = sgd_optimizer;
	return o;


}
