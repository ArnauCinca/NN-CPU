#include "vectorOp.h"
#include <stdlib.h>

#include <stdio.h>

void randomInit(int size, double* a){
	for(; size>0; --size) a[size-1] = 0.01*(rand()%99+1)*(rand()%2?1:-1);
}
void to0(int size, double* a){
	for(; size>0; --size) a[size-1] = 0;
}

double dotProductTree(int i, int e, double* a, double* b){
	if(i==e) return a[i]*b[i];
	else{
		int m = (i+e)/2;
		return dotProductTree(i,m,a,b)+dotProductTree(m+1,e,a,b);
	}

}

double dotProduct(int size, double* a, double* b){
	return dotProductTree(0,size,a,b);
}


void copy(int size, double* a,  double* out){
	for(; size>0; --size) out[size-1] = a[size-1];
}

void mult(int size, double* a, double* b, double* out){
	for(; size>0; --size) out[size-1] = a[size-1]*b[size-1];
}
void multVal(int size, double* a, double b, double* out){
	for(; size>0; --size) out[size-1] = a[size-1]*b;
}

void divi(int size, double* a, double* b, double* out){
	for(; size>0; --size) out[size-1] = a[size-1]/b[size-1];
}
void diviVal(int size, double* a, double b, double* out){
	for(; size>0; --size) out[size-1] = a[size-1]/b;
}

void sum(int size, double* a, double* b, double* out){
	for(; size>0; --size) out[size-1] = a[size-1]+b[size-1];
}
void sumVal(int size, double* a, double b, double* out){
	for(; size>0; --size) out[size-1] = a[size-1]+b;
}

void map(int size, void (*f) (double*, double*), double* x, double* out){
	for(; size>0; --size) f(&x[size-1],&out[size-1]);
}


double sumRng(int r1, int r2, double* a){
	if(r1==r2) return a[r1];
	else{
		int m = (r1+r2)/2;
		return sumRng(r1,m,a)+sumRng(m+1,r2,a);
	}

}
