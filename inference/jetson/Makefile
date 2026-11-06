CXX = g++
CXXFLAGS = -std=c++14 -O2 -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib64 -lnvinfer -lcudart
MODE = fp16

# Runs on the Jetson Nano. Models come as models/<recipe>/<height>-<width>.onnx,
# engines are built next to them under engines/ (models/ is rsynced with
# --delete, which would wipe engines kept there).
MODELS := $(shell find models -name '*.onnx' 2>/dev/null | sort)
ENGINES := $(MODELS:models/%.onnx=engines/%.$(MODE).engine)

all: run.bin $(ENGINES)
	@for engine in $(ENGINES); do \
		res=$$(basename $$engine .$(MODE).engine); \
		./run.bin $$engine $${res%-*} $${res#*-} $(MODE) || echo "FAILED $$engine"; \
	done

engines/%.$(MODE).engine: models/%.onnx
	mkdir -p $(dir $@)
	/usr/src/tensorrt/bin/trtexec \
	--onnx=$< \
	--saveEngine=$@ \
	--fp16

run.bin: infer.cpp
	$(CXX) $(CXXFLAGS) -o run.bin infer.cpp $(LIBS)

clean:
	rm -f run.bin

.PHONY: all clean
