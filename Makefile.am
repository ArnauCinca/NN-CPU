ACLOCAL_AMFLAGS=-I m4
AUTOMAKE_OPTIONS = foreign


AM_LDFLAGS=-lm

include_HEADERS=								\
src/activation_function.h	\
src/loss_function.h				\
src/layer.h								\
src/optimizer.h						\
src/model.h								\
src/vector_op.h						\
src/matrix_op.h


api_sources=									\
src/activation_function.c	\
src/loss_function.c				\
src/layer.c								\
src/optimizer.c						\
src/model.c								\
src/vector_op.c						\
src/matrix_op.c




lib_LTLIBRARIES=libnn.la
libnn_la_SOURCES=$(api_sources)

