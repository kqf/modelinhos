all: run.bin
	./run.bin

run.bin: infer.cpp
	g++ -o run.bin infer.cpp `pkg-config --cflags --libs opencv4` -std=c++11

clean:
	rm -rf $(model).onnx

.PRECIOUS: *.onnx
