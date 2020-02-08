#include "../ActivationFunction/ActivationFunction.h"
typedef struct Layer{
  struct Layer* prev;
  struct Layer* next;
  ActivationFunction* act;
  int size;
  int dim;
  int* kernel_shape;
  double** weights;
  void (*forward)(struct Layer* me, double* input, double* res); 
  int index;
}Layer;


Layer* getFirstLayer(Layer* me);
Layer* getILayer(Layer* me, int i);
Layer* getLastLayer(Layer* me);

Layer* Input(int dim, int* kernel_shape);
Layer* Dense(Layer* in, int size, ActivationFunction* act);





