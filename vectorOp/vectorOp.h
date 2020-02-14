void randomInit(int size, double* a);
void to0(int size, double* a);

double dotProduct(int size, double* a, double* b);

void copy(int size, double* a,  double* out);

void mult(int size, double* a, double* b, double* out);
void multVal(int size, double* a, double b, double* out);

void divi(int size, double* a, double* b, double* out);
void diviVal(int size, double* a, double b, double* out);

void sum(int size, double* a, double* b, double* out);
void sumVal(int size, double* a, double b, double* out);

void map(int size, double (*f) (double), double* x, double* res);

double sumRng(int r1, int r2, double* a);
