#include "../Model/Model.h"
#include "../vectorOp/vectorOp.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
int main(){
	//vector op
	srand(time(NULL));
	int size = 3;
	double* a = malloc(size*sizeof(double));
	printf("vectorOp\n");
        randomInit(size,a);
	printf("a: ");
	for(int i = 0; i<size; ++i){
		printf("%x: %f ",i,a[i]);
	}
	double* b = malloc(size*sizeof(double));
        randomInit(size,b);
	printf("\nb: ");
	for(int i = 0; i<size; ++i){
		printf("%x: %f ",i,b[i]);
	}
	printf("\ndot product: ");
	printf("%f",dotProduct(size,a,b));
	
	double* res = malloc(size*sizeof(double));
	printf("\nsum: ");
	sum(size,a,b,res);
	for(int i = 0; i<size; ++i){
		printf("%x: %f ",i,res[i]);
	}
	
	printf("\nmult: ");
	mult(size,a,b,res);
	for(int i = 0; i<size; ++i){
		printf("%x: %f ",i,res[i]);
	}
	printf("\nsumRng (1,2,a): ");
	printf("%f", sumRng(1,2,a));
	
	
	printf("\n");
	printf("\n");
	// Activation Function
	printf("ActivationFunction sigmoid (a)\n");
	struct ActivationFunction* af = sigmoid();
	map(size, af->act, a,res);
	for(int i = 0; i<size; ++i){
		printf("%x: %f ",i,res[i]);
	}

	printf("\n");
	printf("\n");

	//Loss Function
	printf("\nLossFunction mse (a,b)\n");
	struct LossFunction* lf = mse();
	printf("Loss: %f ",lf->loss(size,a,b));
	
	
	printf("\n");
	printf("\n");
	//Layers
	int* dim = malloc(sizeof(int));
	dim[0]=1;
	struct Layer* in = Input(size,dim); // add srand in Input


	int layer_size = 4;
	printf("\nLayer1: (input: a)\n");
	struct Layer* x = Dense(in,layer_size,sigmoid());
	for(int i = 0; i<x->size; ++i){
		printf("Neuron %x\n", i);
		for(int j = 0; j<x->prev->size+1; ++j){
			printf("%x: %f ",j,x->weights[i][j]);
		}
		printf("\n");
	}
	x->forward(x,a,res);
	printf("Out (sigmoid):\n");
	for(int i = 0; i<x->size; ++i){
		printf("%x: %f ",i,res[i]);
	}
	printf("\n");
	printf("\nLayer2: \n");
	struct Layer* x2 = Dense(x,size,sigmoid());
	for(int i = 0; i<x2->size; ++i){
		printf("Neuron %x\n", i);
		for(int j = 0; j<x2->prev->size+1; ++j){
			printf("%x: %f ",j,x2->weights[i][j]);
		}
		printf("\n");
	}

	double* res2 = malloc(size*sizeof(double));
	x2->forward(x2,res,res2);
	printf("Out (sigmoid):\n");
	for(int i = 0; i<x2->size; ++i){
		printf("%x: %f ",i,res2[i]);
	}
	printf("\n");


	double** input = malloc(1*sizeof(double*));
	double** output = malloc(1*sizeof(double*));
	input[0] = a;
	sumVal(size,b,1,b);
	diviVal(size,b,2,b);
	output[0] = b;
	printf("\n");
	printf("\nModel\n");
	struct Model* m = model(x,lf,sgd());
	for(int i = 1; i<=1000; ++i){
		m->fit(m,0.1,1,input,output,10,1);
		m->predict(m,1,&a, &res);
		if(i%100 == 0){
			for(int j = 0; j<x2->size; ++j){
				printf("%d: %f ",j, res[j]);
			}
			printf("\n");
		}
	}
	
	printf("objective: ");
	for(int i = 0; i<x2->size; ++i){
		printf("%x: %f ",i,b[i]);
	}
	printf("\n");
}
