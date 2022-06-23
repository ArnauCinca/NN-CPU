#include "model.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "../vector_op/vector_op.h"
#include "../matrix_op/matrix_op.h"


void fit(model_t* m, double learningRate, int size, double** data, double** res, int epoch, int batch){
	layer_t *in = m->input;
	layer_t *out = m->output;
	double ***outs = calloc(out->index - in->index, sizeof(double**));
	double ***delt = calloc(out->index - in->index, sizeof(double**));
	layer_t *l = in;
	for(int x = 0; x <= out->index - in->index; x++){
		outs[x] = calloc(batch, sizeof(double*));
		delt[x] = calloc(batch, sizeof(double*));
		for(int y = 0; y < batch; y++){
			outs[x][y] = calloc(l->layer_size, sizeof(double));
			delt[x][y] = calloc(l->layer_size, sizeof(double));
		}
		l = l->next;
	}


	double lr;
	double loss;
	int batch_size;
	//-----------------------------------------------------------------
	for(int epo = 0; epo < epoch; epo++){
		loss = 0;
		lr = learningRate; //TODO
		//Forward
		for(int nb = 0; nb <= size/batch; nb++){
			batch_size = fmin(batch,size - nb*batch);

			in->forward(in, batch_size, &data[nb*batch], outs[0]);
			l = in;
		       	while(l != out){
				l = l->next;
				l->forward(l, batch_size, outs[l->index-1], outs[l->index]);
			}
			for(int b = 0; b < batch_size; b++){
				//m->optimizer->optimize(in, m->loss, lr);   //train
				loss += m->loss_fun->loss(m->loss_fun, out->layer_size, outs[out->index][b], res[nb*batch + b]);
			}
		}
		//Backprop
		//PRINT MEAN EPOCH LOSS
		printf("Epoch %d:  loss: %f\n", epo, loss/size);
	}
/*
	free(realOuts);
	free(tmp2);
	free(tmp);
	l = me->firstLayer;
    while(l != NULL){
        free(outs[l->index]);
        free(deltas[l->index]);
        l = l->next;
    }
	free(deltas);
	free(outs);
	*/
}

void test(model_t *m, int size, double **input, double **output){
	layer_t *in = m->input;
	layer_t *out = m->output;
	layer_t *l = in;
	double ***outs = calloc(out->index - in->index, sizeof(double**));
	for(int x = 0; x <= out->index - in->index; x++){
		outs[x] = calloc(size, sizeof(double*));
		for(int y = 0; y < size; y++){
			outs[x][y] = calloc(l->layer_size, sizeof(double));
		}
		l = l->next;
	}
	double loss = 0;
	int s = 0;
	in->forward(in, size, input, outs[0]);
	l = in;
	while(l != out){
		l = l->next;
		l->forward(l, size, outs[l->index -1], outs[l->index]);
	}
	for(int x = 0; x < size; x++){
		loss += m->loss_fun->loss(m->loss_fun, out->layer_size, outs[out->layer_size][x], output[x]);
	}
	printf("Loss: %f\n", loss/size);

}

void predict(model_t *m, int size, double **input, double **res){
    layer_t *in = m->input;
    layer_t *out = m->output;
    layer_t *l = in;
    double ***outs = calloc(out->index - in->index, sizeof(double**));
	for(int x = 0; x <= out->index - in->index; x++){
		outs[x] = calloc(size, sizeof(double*));
		for(int y = 0; y < size; y++){
			outs[x][y] = calloc(l->layer_size, sizeof(double));
		}
		l=l->next;
	}
	in->forward(in, size, input, outs[0]);
	l = in;
	while(l != out){
		l = l->next;
		l->forward(l, size, outs[l->index-1], outs[l->index]);
	}
	mat_copy(size, out->layer_size, outs[out->index], res);
    free(outs);
}
/*
//TODO
void read (struct Model* me, int sizeName, char* name){




}
void save (struct Model* me, int sizeName, char* name){
//save layers (type_id,act_id,size,dim.kernel_shape,weights)



}
*/

model_t* model(layer_t *input, layer_t *output, loss_function_t *loss_fun, optimizer_t* optim){
	model_t *m = calloc(1,sizeof(model_t));
	m->input = input;
	m->output = output;

	m->loss_fun = loss_fun;
	m->optimizer = optim;

	m->fit = fit;
	m->test = test;
	m->predict = predict;
	//m->read = read;
	//m->save = save;
	return m;
}
