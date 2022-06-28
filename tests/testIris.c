#include "../model/model.h"
#include "../vector_op/vector_op.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
int main(){
	srand(time(NULL));

	int inSize;
	int outSize;
	int size;



	FILE *fp;
	char buff[255];
    
   	fp = fopen("tests/iris.txt", "r");
	fscanf(fp, "%s", buff);
	inSize = atoi(buff);
	fscanf(fp, "%s", buff);
	outSize = atoi(buff);
	fscanf(fp, "%s", buff);
	size = atoi(buff);



	double** input = malloc(size*sizeof(double*));
	double** output = malloc(size*sizeof(double*));
	for(int i = 0; i<size;++i){
		input[i] = malloc(inSize*sizeof(double));		
		output[i] = malloc(outSize*sizeof(double));		
		for(int j = 0; j<inSize; ++j){
			fscanf(fp, "%s", buff);
			input[i][j] = atoi(buff);
		}
		for(int j = 0; j<outSize; ++j){
			fscanf(fp, "%s", buff);
			output[i][j] = atoi(buff);
		}
	}

	layer_t *in = InputDense(inSize);

	layer_t *x = Dense(1024,sigmoid(),in);
	x = Dense(512,sigmoid(),x);
	x = Dense(256,sigmoid(),x);
	x = Dense(128,sigmoid(),x);
	x = Dense(64,sigmoid(),x);
	x = Dense(32,sigmoid(),x);
	x = Dense(16,sigmoid(),x);
	x = Dense(outSize,sigmoid(),x);

	model_t *m = model(in, x, mse(), sgd());
	

	double* res = malloc(outSize*sizeof(double));


	printf("Pre-train:\n");
	m->predict(m,1,&input[6], &res);
	printf("Input:\n");
	for(int i = 0; i< inSize; ++i)	printf("%f ", input[6][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", output[6][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", res[i]);	
	printf("\n");	

	m->predict(m,1,input, &res);
	printf("Input:\n");
	for(int i = 0; i< inSize; ++i) printf("%f ", input[0][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", output[0][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", res[i]);	
	printf("\n");	

	m->predict(m,1,&input[2], &res);
	printf("Input:\n");
	for(int i = 0; i< inSize; ++i) printf("%f ", input[2][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", output[2][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", res[i]);	
	printf("\n");	
	
	m->predict(m,1,&input[10], &res);
	printf("Input:\n");
	for(int i = 0; i< inSize; ++i) printf("%f ", input[10][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", output[10][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", res[i]);	
	printf("\n");	
	printf("\n");	
	
	m->fit(m,0.1,size,input,output,1000,1);
	
	printf("\n");	
	printf("\n");	
	printf("Post-train:\n");
	m->predict(m,1,&input[6], &res);
	printf("Input:\n");
	for(int i = 0; i< inSize; ++i) printf("%f ", input[6][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", output[6][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", res[i]);	
	printf("\n");	

	m->predict(m,1,input, &res);
	printf("Input:\n");
	for(int i = 0; i< inSize; ++i) printf("%f ", input[0][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", output[0][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< outSize; ++i) printf("%f ", res[i]);	
	printf("\n");	

	m->predict(m,1,&input[2], &res);
	printf("Input:\n");
	for(int i = 0; i< inSize; ++i) printf("%f ", input[2][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", output[2][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", res[i]);	
	printf("\n");	
	
	m->predict(m,1,&input[10], &res);
	printf("Input:\n");
	for(int i = 0; i< inSize; ++i) printf("%f ", input[10][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< outSize; ++i)	printf("%f ", output[10][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< outSize; ++i) printf("%f ", res[i]);	
	printf("\n");	
	

	int** CM = malloc(outSize*sizeof(int*));
	for(int i = 0; i<outSize;++i){
		CM[i] = malloc(outSize*sizeof(int));
		for(int j = 0; j<outSize; ++j) CM[i][j] = 0;
	}	
	
	for(int i = 0; i<size; ++i){		
		m->predict(m,1,&input[i], &res);
		int maxi = 0;
		double max;
		for(int j = 0; j<outSize; ++j){
			max = res[maxi];
			if(max<res[j]) maxi = j;
		}
		for(int j = 0; j<outSize; ++j){
			if(output[i][j] == 1) ++CM[j][maxi];	
		}
	}

	printf("Confusion Matrix:\n");
	for(int i = 0; i<outSize; ++i){
		for(int j = 0; j<outSize; ++j){
			printf("%d ", CM[i][j]);
		}
		printf("\n");
	}	
}
