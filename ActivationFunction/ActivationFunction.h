typedef struct ActivationFunction{
	double* aux;
	void (*act)(double* x, double* res);
	void (*actPrime)(double* x, double* res);
} ActivationFunction;

ActivationFunction* sigmoid();
ActivationFunction* ReLU();
ActivationFunction* identity();
ActivationFunction* tanH();

