void mat_copy(int x, int y, double **m1, double **m2);
void mat_mult(int x, int y, int z, double** A, double** B, double** res); //A[i][j] * B[j][k] = res[i][k]
void mat_add(int x, int y, double** A, double** B, double** res); //A[i][j] + B[i][j] = res[i][j]
void mat_transpose(int x, int y, double** A,  double** res);
