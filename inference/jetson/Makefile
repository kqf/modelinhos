CXX = g++
CXXFLAGS = -std=c++14 -O2
MODELS := $(shell find models -name '*.onnx' 2>/dev/null | sort)

# Runs on the Jetson Nano. Same contract as vonnx/Makefile:
# models/<recipe>/<height>-<width>.onnx, MODE selects fp32|fp16|int8.
MODE ?= fp32

all: run.bin
	@for model in $(MODELS); do \
		res=$$(basename $$model .onnx); \
		./run.bin $$model $${res%-*} $${res#*-} $(MODE) || echo "FAILED $$model"; \
	done

run.bin: infer.cpp
	$(CXX) $(CXXFLAGS) -o run.bin infer.cpp

clean:
	rm -f run.bin

.PHONY: all clean
