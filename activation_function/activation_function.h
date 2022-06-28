typedef struct activation_function_t{
	double* aux;
	void (*act)(double* x, double* res);
	void (*act_prime)(double* x, double* res);
} activation_function_t;

activation_function_t *sigmoid();
activation_function_t *ReLU();
activation_function_t *identity();
activation_function_t *tanH();

