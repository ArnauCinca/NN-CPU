#include "layer.h"
typedef struct optimizer_t{
  void (*optimize)(layer_t *in, layer_t *out, loss_function_t *loss, double lr, int batch, double ***outs, double ***fouts, double **real_outs, double ***deltas);
} optimizer_t;

optimizer_t *sgd();

