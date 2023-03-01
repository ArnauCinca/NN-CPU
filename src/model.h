#include "optimizer.h"
typedef struct model_t{
  layer_t *input;
  layer_t *output;//TODO: need?

  loss_function_t *loss_fun;
  optimizer_t *optimizer;

  void (*fit)(struct model_t *m, double learningRate, int size, double **data, double **out, int epoch, int batchSize);
  void (*test)(struct model_t *m, int size, double **data, double **out);
  void (*predict)(struct model_t *m, int size, double** data, double** res);
  //TODO
  //void (*read)(struct model_t *m, char* file_name);
  //void (*save)(struct model_t *m, char* file_name);
} model_t;

model_t* model(layer_t *input, layer_t *output, loss_function_t *loss_func, optimizer_t *optim);

