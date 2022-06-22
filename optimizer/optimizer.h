#include "../layer/layer.h"
typedef struct optimizer_t{
  void (*optimize)(layer_t *l, loss_function_t *loss, double lr, int batch, double ***outs, double **realOuts, double ***deltas, double *tmp);
} optimizer_t;

optimizer_t *sgd();

