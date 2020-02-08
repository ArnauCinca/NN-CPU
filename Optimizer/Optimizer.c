#include "Optimizer.h"
#include "../vectorOp/vectorOp.h"
#include <stdlib.h>
#include <stdio.h>


void OP_sgd(struct Layer* layer, struct LossFunction* loss, double lr,  double** outs, double* realOuts, double** deltas, double* tmp){
	struct Layer* l;
	l = getLastLayer(layer);
	
	//deltas
	//Output Layer
	loss->lossPrime(l->size, outs[l->index], realOuts, deltas[l->index]);//loss	
	l->act->actPrime(l->size, outs[l->index], tmp);//act
	mult(l->size, tmp, deltas[l->index], deltas[l->index]);// loss*act

	l = l->prev;
	//hidden Layers
	while(l->prev != NULL){ //untill input layer
		l->act->actPrime(l->size, outs[l->index] , tmp);//act
		for(int i = 0; i<l->size; ++i){
			deltas[l->index][i] = 0;
			for(int j = 0; j < l->next->size; ++j){
				deltas[l->index][i] += deltas[l->index+1][j]* l->next->weights[j][i];//backprop
			}
			deltas[l->index][i] *= tmp[i];  //mult
		}
		l = l->prev;
	}

	//update weights
	l = layer->next; //first layer before input layer
	while (l != NULL){
		//update
		for (int i = 0; i< l->size; ++i){
			for(int j = 0; j<l->prev->size; ++j){
				l->weights[i][j] -= lr*deltas[l->index][i]*outs[l->index-1][j];//weights
			}
			l->weights[i][l->prev->size] -= lr*deltas[l->index][i]; //bias
		}

		l = l->next;
	}	
}

Optimizer* sgd(){
	struct Optimizer* o = malloc(sizeof(Optimizer));
	o->optimize = OP_sgd;
	return o;


}
