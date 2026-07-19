CXX = g++
CXXFLAGS = -std=c++17 -O2
OPENCV := $(shell pkg-config --cflags --libs opencv4)
MODELS := $(shell find models -name '*.onnx' 2>/dev/null | sort)

# Resolution comes from the file name: models/<recipe>/<height>-<width>.onnx
# MODE selects the precision (fp32|fp16|int8), e.g.:  make MODE=int8
MODE ?= fp32

all: run.bin
	@for model in $(MODELS); do \
		res=$$(basename $$model .onnx); \
		./run.bin $$model $${res%-*} $${res#*-} $(MODE) || echo "FAILED $$model"; \
	done

run.bin: infer.cpp
	$(CXX) $(CXXFLAGS) -o run.bin infer.cpp $(OPENCV)

models:
	python export.py

clean:
	rm -f run.bin

.PHONY: all clean
