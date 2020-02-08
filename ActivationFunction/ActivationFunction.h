typedef struct ActivationFunction{
	double* aux;
	void (*act)(int size, double* x, double* res);
	void (*actPrime)(int size, double* x, double* res);
} ActivationFunction;

ActivationFunction* sigmoid();
ActivationFunction* ReLU();
ActivationFunction* identity();
ActivationFunction* tanH();

