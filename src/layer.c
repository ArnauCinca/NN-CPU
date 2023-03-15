#include "layer.h"
#include "vector_op.h"
#include "matrix_op.h"
#include <stdlib.h>
#include <stdio.h>

//INPUT
void forward_input(layer_t *input, int batch, double **in, double **out, double **fout){
	mat_copy(batch, input->layer_size, in, fout);
}


//DENSE
void forward_dense(layer_t *layer, int batch, double **in, double **out, double **fout){
	int size = layer->layer_size;
	int prev_size = layer->prev->layer_size;


	//WEIGHTS & BIAS
	mat_mult(batch, prev_size+1, size, in, layer->weights, out);
	//Activation Function
	for(int b = 0; b < batch; b++){
		map(size, layer->act_fun->act, out[b], fout[b]); // activation Function
	}
}


void backpropagation_dense(layer_t *layer, int batch, double **outs,  double **deltas, double **deltas_next){
	int size = layer->layer_size;
	int next_size = layer->next->layer_size;
	double tmp[size];
	for(int b = 0; b < batch; b++){
		map(size, layer->act_fun->act_prime, outs[b], tmp);
	}

	mat_transpose(size+1, next_size, layer->next->weights, layer->next->weightsT);

	mat_mult(batch, next_size, size, deltas_next, layer->next->weightsT, deltas);

	for(int b = 0; b < batch; b++){
		mult(size,tmp,deltas[b],deltas[b]);
	}
}


void backpropagation_output_dense(layer_t *layer, int batch, double **outs, double **fouts, double **deltas, loss_function_t *loss, double **realOuts){
	int size = layer->layer_size;
	double tmp[size];
	for(int b = 0; b < batch; b++){
		loss->loss_prime(loss, size, fouts[b], realOuts[b], deltas[b]);//loss	
		map(size, layer->act_fun->act_prime, outs[b], tmp); //TODO:Reuse outs as they are not needed
		mult(size, tmp, deltas[b], deltas[b]);// loss*actP
	}
}

void gradient_descent_dense(layer_t *layer, int batch, double lr, double **fouts_prev, double **deltas){
	int size = layer->layer_size;
	int prev_size = layer->prev->layer_size;
	for(int b = 0; b < batch; b++){
		for (int i = 0; i < size; i++){
			for(int j = 0; j < prev_size; j++){
				layer->weights[j][i] -= lr * deltas[b][i]*fouts_prev[b][j];//weights
			}
			layer->weights[prev_size][i] -= lr * deltas[b][i]; //bias
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

	l->weights = calloc(input->layer_size + 1,sizeof(double*));
	for(int i = 0; i < input->layer_size + 1 ; ++i){
		l->weights[i] = calloc(layer_size,sizeof(double));
		randomInit(layer_size, l->weights[i]);
	}
	l->weightsT = calloc(layer_size,sizeof(double*));
	for(int i = 0; i < layer_size; ++i){
		l->weightsT[i] = calloc(input->layer_size + 1 ,sizeof(double));
		randomInit(input->layer_size +1, l->weightsT[i]);
	}
   
	l->act_fun = act;

	l->forward = forward_dense;
	l->backpropagation = backpropagation_dense;
	l->backpropagation_output = backpropagation_output_dense;
	l->gradient_descent = gradient_descent_dense;


	input->next = l;
	return l;
}


