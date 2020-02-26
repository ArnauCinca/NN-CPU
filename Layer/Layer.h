#include "../ActivationFunction/ActivationFunction.h"
#include "../LossFunction/LossFunction.h"
typedef struct Layer{
  struct Layer* prev;
  struct Layer* next;
  ActivationFunction* act;
  int dim;
  int* shape; //0: layer size, 1: dim1, 2: dim2, ...
  double** weights;
  void (*forward)(struct Layer* me, double* input, double* res); 
  void (*backprop)(struct Layer* me, double** outs, double** deltas, double* tmp); 
  void (*backpropOutput)(struct Layer* me, double** outs, double** deltas, double* tmp, LossFunction* lf, double* realOuts); 
  void (*gradientDescent)(struct Layer* me, double lr, double** outs, double** deltas); 
  int index;
}Layer;


Layer* getFirstLayer(Layer* me);
Layer* getILayer(Layer* me, int i);
Layer* getLastLayer(Layer* me);

Layer* Input(int dim, int* kernel_shape);
Layer* Dense(Layer* in, int size, ActivationFunction* act);





