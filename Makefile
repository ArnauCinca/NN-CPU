all:
	mkdir -p makes/
	gcc -o makes/vectorOp.o -c vectorOp/vectorOp.c -O3
	gcc -o makes/ActivationFunction.o -c ActivationFunction/ActivationFunction.c -O3
	gcc -o makes/LossFunction.o -c LossFunction/LossFunction.c -O3
	gcc -o makes/Layer.o -c Layer/Layer.c -O3
	gcc -o makes/Optimizer.o -c Optimizer/Optimizer.c -O3
	gcc -o makes/Model.o -c Model/Model.c -O3

test: tests/test.c
	mkdir -p makes/tests
	gcc -o makes/tests/test.o -c tests/test.c
	gcc makes/*.o makes/tests/test.o -o test -lm

testOAX: tests/testOAX.c
	mkdir -p makes/tests
	gcc -o makes/tests/testOAX.o -c tests/testOAX.c
	gcc makes/*.o makes/tests/testOAX.o -o testOAX -lm

testIris: tests/testIris.c
	mkdir -p makes/tests
	gcc -o makes/tests/testIris.o -c tests/testIris.c
	gcc makes/*.o makes/tests/testIris.o -o testIris -lm

clean:
	rm -r makes
