#include "../Model/Model.h"
#include "../vectorOp/vectorOp.h"
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
    
   	fp = fopen("tests/oax.txt", "r");
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

	struct Layer* in = Input(inSize, NULL);

	struct Layer* x = Dense(in,8,sigmoid());
	x = Dense(x,4,sigmoid());
	x = Dense(x,outSize,sigmoid());

	struct Model* m = model(x,mse(),sgd());
	

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
	
	m->fit(m,0.01,size/2,input,output,500,1);
	
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
	printf("\n");	
   
	printf("Test ");	
	m->test(m,size/2,input+size/2,output+size/2);
	
}
