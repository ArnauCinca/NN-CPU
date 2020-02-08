#include "../Layer/Layer.h"
#include "../LossFunction/LossFunction.h"
typedef struct Optimizer{
  void (*optimize)(struct Layer* l, struct LossFunction* loss, double lr,  double** outs, double* realOut, double** deltas, double* tmp);
} Optimizer;

Optimizer* sgd();

