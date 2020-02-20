#include "../Optimizer/Optimizer.h"
typedef struct Model{
  Layer* firstLayer;
  LossFunction* loss;
  Optimizer* optimizer;
  int maxLayerSize;
  void (*fit)(struct Model* me, double learningRate, int size, double** data, double** out, int epoch, int batchSize);
  void (*test)(struct Model* me, int size, double** data, double** out);
  void (*predict)(struct Model* me, int size, double** data, double** res);
  void (*read)(struct Model* me, int sizeName, char* name);
  void (*save)(struct Model* me, int sizeName, char* name);
} Model;

Model* model(Layer* layer, LossFunction* loss, Optimizer* op);

