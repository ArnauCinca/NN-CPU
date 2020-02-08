typedef struct LossFunction {
	double* aux;
	double (*loss)(int size, double* yhat, double* y);
	void (*lossPrime)(int size, double* yhat, double* y, double* res);
} LossFunction;

LossFunction* mse();

