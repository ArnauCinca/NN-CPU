#include "layer.h"
#include "../vector_op/vector_op.h"
#include "../matrix_op/matrix_op.h"
#include <stdlib.h>
#include <stdio.h>

//INPUT
void forward_input(layer_t *input, int batch, double **in, double **out){
	mat_copy(batch, input->layer_size, in, out);
}


//DENSE
void forward_dense(layer_t *layer, int batch, double **in, double **out){
	//matMultOp(i,size,1, me->weights, &in, &out);
	int size = layer->layer_size;
	for(int b = 0; b < batch; b++){
		for(int i = 0; i < size; i++) {
			out[i][b] = dotProduct(size, in[b], layer->weights[i]);
			out[i][b] += layer->weights[i][size];  //dot product
		}
		map(size, layer->act_fun->act, out[b], out[b]); // activation Function
	}
}


void backpropagation_dense(layer_t *layer, int batch, double **outs, double **deltas, double **deltas_next, double *tmp){
	int size = layer->layer_size;

	for(int b = 0; b < batch; b++){
		map(size, layer->act_fun->act_prime, outs[b], tmp);
    	for(int i = 0; i < size; i++){  //matmul
    	   	deltas[b][i] = 0;
			for(int j = 0; j < layer->next->layer_size; ++j){
				deltas[b][i] += deltas_next[b][j] * layer->next->weights[j][i];
			}
		}
		mult(size,tmp,deltas[b],deltas[b]);
	}
}


void backpropagation_output_dense(layer_t *layer, int batch, double **outs, double **deltas, double *tmp, loss_function_t *loss, double **realOuts){
	int size = layer->layer_size;
	for(int b = 0; b < batch; b++){
		loss->loss_prime(size, outs[b], realOuts[b], deltas[b]);//loss	
		map(size, layer->act_fun->act_prime, outs[b], tmp);
		mult(size, tmp, deltas[b], deltas[b]);// loss*actP
	}
}

void gradient_descent_dense(layer_t *layer, int batch, double lr, double **outs_prev, double **outs, double **deltas){
	for(int b = 0; b < batch; b++){
		for (int i = 0; i < layer->layer_size; i++){
			for(int j = 0; j < layer->prev->layer_size; j++){
				layer->weights[i][j] -= lr * deltas[b][i]*outs_prev[b][j];//weights
			}
			layer->weights[i][layer->prev->layer_size] -= lr*deltas[b][i]; //bias
		}
	}
}

layer_t* InputDense(int input_size){	
	layer_t *l = calloc(1, sizeof(layer_t));

	l->prev = NULL;
	l->next = NULL;


	l->layer_size = input_size;

	l->weights = NULL;
	l->act_fun = NULL;

	l->forward = forward_input;
	l->backpropagation = NULL;
	l->backpropagation_output = NULL;
	l->gradient_descent = NULL;

	l->index = 0;


	return l;
	
}

layer_t* Dense(int layer_size, activation_function_t *act, layer_t *input){
	
	layer_t *l = calloc(1, sizeof(layer_t));

	l->prev = input;
	l->next = NULL;
	l->index = input->index + 1;	

	l->layer_size = layer_size;

	l->weights = calloc(layer_size,sizeof(double*));
	for(int i = 0; i<layer_size; ++i){
		l->weights[i] = calloc(input->layer_size + 1,sizeof(double));
		randomInit(input->layer_size + 1, l->weights[i]);
	}
   
	l->act_fun = act;



	l->forward = forward_dense;
	l->backpropagation = backpropagation_dense;
	l->backpropagation_output = backpropagation_output_dense;
	l->gradient_descent = gradient_descent_dense;


	input->next = l;
	return l;
}


