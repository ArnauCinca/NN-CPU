all:
	mkdir -p makes/
	gcc -o makes/vector_op.o -c vector_op/vector_op.c -O3 -std=c99
	gcc -o makes/matrix_op.o -c matrix_op/matrix_op.c -O3 -std=c99
	gcc -o makes/activation_function.o -c activation_function/activation_function.c -O3 -std=c99
	gcc -o makes/loss_function.o -c loss_function/loss_function.c -O3 -std=c99
	gcc -o makes/layer.o -c layer/layer.c -O3 -std=c99
	gcc -o makes/optimizer.o -c optimizer/optimizer.c -O3 -std=c99
	gcc -o makes/model.o -c model/model.c -O3  -std=c99
	rm -f test
	rm -f testOAX
	rm -f testIris

testOAX: tests/testOAX.c
	mkdir -p makes/tests
	gcc -o makes/tests/testOAX.o -c tests/testOAX.c -std=c99
	gcc makes/*.o makes/tests/testOAX.o -o testOAX -lm

testIris: tests/testIris.c
	mkdir -p makes/tests
	gcc -o makes/tests/testIris.o -c tests/testIris.c
	gcc makes/*.o makes/tests/testIris.o -o testIris -lm

clean:
	rm -r makes
