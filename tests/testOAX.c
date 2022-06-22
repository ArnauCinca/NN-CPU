#include "../model/model.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
int main(){
	srand(time(NULL));

	int input_size;
	int output_size;
	int size;



	FILE *fp;
	char buff[255];
    
   	fp = fopen("tests/oax.txt", "r");
	fscanf(fp, "%s", buff);
	input_size = atoi(buff);
	fscanf(fp, "%s", buff);
	output_size = atoi(buff);
	fscanf(fp, "%s", buff);
	size = atoi(buff);



	double** input = malloc(size*sizeof(double*));
	double** output = malloc(size*sizeof(double*));
	for(int i = 0; i<size;++i){
		input[i] = malloc(input_size*sizeof(double));		
		output[i] = malloc(output_size*sizeof(double));		
		for(int j = 0; j<input_size; ++j){
			fscanf(fp, "%s", buff);
			input[i][j] = atoi(buff);
		}
		for(int j = 0; j<output_size; ++j){
			fscanf(fp, "%s", buff);
			output[i][j] = atoi(buff);
		}
	}

	layer_t *i = InputDense(input_size);

	layer_t *x = Dense(8, sigmoid(), i);
	x = Dense(4,sigmoid(),x);
	x = Dense(output_size,sigmoid(),x);

	struct model_t *m = model(i, x, mse(), sgd());
	

	double* res = malloc(output_size*sizeof(double));


	printf("Pre-train:\n");
	m->predict(m,1,&input[6], &res);
	printf("Input:\n");
	for(int i = 0; i< input_size; ++i)	printf("%f ", input[6][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", output[6][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", res[i]);	
	printf("\n");	

	m->predict(m,1,input, &res);
	printf("Input:\n");
	for(int i = 0; i< input_size; ++i) printf("%f ", input[0][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", output[0][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", res[i]);	
	printf("\n");	

	m->predict(m,1,&input[2], &res);
	printf("Input:\n");
	for(int i = 0; i< input_size; ++i) printf("%f ", input[2][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", output[2][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", res[i]);	
	printf("\n");	
	
	m->predict(m,1,&input[10], &res);
	printf("Input:\n");
	for(int i = 0; i< input_size; ++i) printf("%f ", input[10][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", output[10][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", res[i]);	
	printf("\n");	
	printf("\n");	
	
	m->fit(m,0.01,size/2,input,output,500,1);
	
	printf("\n");	
	printf("\n");	
	printf("Post-train:\n");
	m->predict(m,1,&input[6], &res);
	printf("Input:\n");
	for(int i = 0; i< input_size; ++i) printf("%f ", input[6][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", output[6][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", res[i]);	
	printf("\n");	

	m->predict(m,1,input, &res);
	printf("Input:\n");
	for(int i = 0; i< input_size; ++i) printf("%f ", input[0][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", output[0][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< output_size; ++i) printf("%f ", res[i]);	
	printf("\n");	

	m->predict(m,1,&input[2], &res);
	printf("Input:\n");
	for(int i = 0; i< input_size; ++i) printf("%f ", input[2][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", output[2][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", res[i]);	
	printf("\n");	
	
	m->predict(m,1,&input[10], &res);
	printf("Input:\n");
	for(int i = 0; i< input_size; ++i) printf("%f ", input[10][i]);
	printf("\n");	
	printf("Real Output:\n");
	for(int i = 0; i< output_size; ++i)	printf("%f ", output[10][i]);	
	printf("\n");	
	printf("Net Output:\n");
	for(int i = 0; i< output_size; ++i) printf("%f ", res[i]);	
	printf("\n");	
	printf("\n");	
   
	printf("Test ");	
	m->test(m,size/2,input+size/2,output+size/2);
	
}
