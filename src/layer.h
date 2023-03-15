#include "activation_function.h"
#include "loss_function.h"

typedef struct layer_t{
	struct layer_t *prev;
	struct layer_t *next;
	int index;



	int layer_size;
	//TODO: CNNs
	//int input_dim;
	//int *input_shape; //0: dim1, 1: dim2, ...

	
	double** weights;
	double** weightsT;
	activation_function_t *act_fun;



	void (*forward)(struct layer_t *layer, int batch, double **in, double **out, double **fout); 
	void (*backpropagation)(struct layer_t *layer, int batch, double **outs, double **deltas, double **deltas_next); 
	void (*backpropagation_output)(struct layer_t *layer, int batch, double **outs, double **fouts, double **deltas, loss_function_t *lf, double **realOuts); 
	void (*gradient_descent)(struct layer_t *layer, int batch, double lr, double **fouts_prev, double **deltas); 
} layer_t;


layer_t* InputDense(int input_size);
layer_t* Dense(int layer_size, activation_function_t* act_fun, layer_t *input);





