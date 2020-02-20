#include "Model.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "../vectorOp/vectorOp.h"


void fit(struct Model* me, double learningRate, int size, double** data, double** out, int epoch, int batchSize){
	//all variables needed-------------------------------------------
	//batch
	int batch;
	double nBatch = (double)size/batchSize;
	nBatch += (nBatch > (int)nBatch)?1:0;
	nBatch = (double)(int)(nBatch); //rm decimal part
	//outs & deltas
	struct Layer* l;
	l = getLastLayer(me->firstLayer);
	int outSize = l->size;
	int s = l->index+1;	
    double** outs = malloc(s*sizeof(double*));
    double** deltas = malloc(s*sizeof(double*));
    l = me->firstLayer;
    while(l != NULL){
        outs[l->index] = malloc(l->size*sizeof(double));
        deltas[l->index] = malloc(l->size*sizeof(double));
        l = l->next;
    }
	
	//tmps
	double* tmp = malloc(me->maxLayerSize*sizeof(double));
	double* tmp2 = malloc(me->maxLayerSize*sizeof(double));
	double* realOuts = malloc(outSize*sizeof(double));
	
	double lr;
	double loss;
	//-----------------------------------------------------------------
	for(int epo = 1; epo<=epoch; ++epo){
		loss = 0;
		lr = learningRate/epo;
		for(int i = 0; i<size; i+=batchSize){
			l = me->firstLayer;
			while(l != NULL){
				to0(l->size, outs[l->index]);
				l = l->next;
			}
			for(int a = 0; a<outSize; ++a){
				realOuts[a] = 0; 
			}
			batch = fmin(batchSize,size-i);
			for(int j = 0; j < batch; ++j){
				//input	
				l = me->firstLayer;
				l->forward(l,data[i+j],tmp);
				sum(l->size, tmp, outs[0], outs[0]); //save the result for train
				l = l->next;
				while(l != NULL){
					l->forward(l, tmp, tmp2);  //forward layer
					sum(l->size, tmp2, outs[l->index], outs[l->index]); //save the result for train
					copy(l->size, tmp2, tmp);
					l = l->next;
				}
				for(int a = 0; a<outSize; ++a){
					realOuts[a] += out[i+j][a]; 
				}
			}
			l = me->firstLayer;
			while(l != NULL){	
				diviVal(l->size, outs[l->index], batch, outs[l->index]);	
				l = l->next;
			}
			diviVal(outSize, realOuts, batch, realOuts);
			me->optimizer->optimize(me->firstLayer, me->loss, lr, outs, realOuts, deltas, tmp);   //train
			loss += me->loss->loss(outSize, outs[getLastLayer(me->firstLayer)->index], realOuts);
		}	
		//PRINT MEAN EPOCH LOSS
		printf("Epoch %d:  loss: %f\n", epo, loss/nBatch);
	}

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
}


void test(struct Model* me, int size, double** data, double** out){
    double* tmp = malloc(me->maxLayerSize*sizeof(double));
    double* tmp2 = malloc(me->maxLayerSize*sizeof(double));
    struct Layer* l;
	double loss = 0;
  	for(int i = 0; i<size; ++i){
		l = me->firstLayer; 	
		l->forward(l,data[i],tmp);
    	l = l->next;
    	while(l != NULL){
        	l->forward(l, tmp, tmp2);  //forward layer
        	copy(l->size, tmp2, tmp);
        	l = l->next;
    	}
		loss += me->loss->loss(getLastLayer(me->firstLayer)->size, tmp, out[i]);
	}
	printf("Loss: %f\n", loss/size);
    free(tmp2);
    free(tmp);



}

void predict(struct Model* me, int size, double** data, double** res){
	double* tmp = malloc(me->maxLayerSize*sizeof(double));
	double* tmp2 = malloc(me->maxLayerSize*sizeof(double));
	struct Layer* l = me->firstLayer;
	l->forward(l,data[0],tmp);
    l = l->next;
	int s = 0;
    while(l != NULL){
    	l->forward(l, tmp, tmp2);  //forward layer
        copy(l->size, tmp2, tmp);
		s = l->size;//save size for copy the out to res
        l = l->next;
    }
	copy(s,tmp,res[0]);
	free(tmp2);
	free(tmp);
}

void read (struct Model* me, int sizeName, char* name){




}
void save (struct Model* me, int sizeName, char* name){
//save layers (type_id,act_id,size,dim.kernel_shape,weights)



}


Model* model(Layer* layer, LossFunction* loss, Optimizer* op){
	struct Model* m = malloc(sizeof(Model));
	m->firstLayer = getFirstLayer(layer);
	m->loss = loss;
	m->optimizer = op;
	struct Layer* l = m->firstLayer;
	int max = 0;
	while(l != NULL){
		max = (max>l->size)?max:l->size;
		l = l->next;
	}
	m->maxLayerSize = max;
	m->fit = fit;
	m->test = test;
	m->predict = predict;
	m->read = read;
	m->save = save;
	return m;
}
