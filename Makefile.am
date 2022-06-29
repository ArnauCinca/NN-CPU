ACLOCAL_AMFLAGS=-I m4
AUTOMAKE_OPTIONS = foreign


AM_LDFLAGS=-lm

include_HEADERS=								\
src/activation_function/activation_function.h	\
src/loss_function/loss_function.h				\
src/layer/layer.h								\
src/optimizer/optimizer.h						\
src/model/model.h								\
src/vector_op/vector_op.h						\
src/matrix_op/matrix_op.h


api_sources=									\
src/activation_function/activation_function.c	\
src/loss_function/loss_function.c				\
src/layer/layer.c								\
src/optimizer/optimizer.c						\
src/model/model.c								\
src/vector_op/vector_op.c						\
src/matrix_op/matrix_op.c


lib_LTLIBRARIES=libnn.la
libnn_la_SOURCES=$(api_sources)
