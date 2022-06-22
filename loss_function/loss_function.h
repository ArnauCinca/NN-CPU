typedef struct loss_function_t {
	double* aux;
	double (*loss)(int size, double* yhat, double* y);
	void (*loss_prime)(int size, double* yhat, double* y, double* res);
} loss_function_t;

loss_function_t *mse();

