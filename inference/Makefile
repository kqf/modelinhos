MODE ?= fp32
REMOTE = rpi5:~/modelinhos/inference/vonnx/
JETSON = jetson-nano:~/modelinhos/inference/jetson/
EXPORTED = vonnx/models/.exported

rpi5: $(EXPORTED)
	ssh rpi5 'mkdir -p ~/modelinhos/inference/vonnx'
	rsync -az --delete --progress vonnx/Makefile vonnx/infer.cpp $(REMOTE)
	rsync -az --delete --progress vonnx/models $(REMOTE)
	ssh rpi5 'make -C ~/modelinhos/inference/vonnx'

jetson-nano: $(EXPORTED)
	ssh jetson-nano 'mkdir -p ~/modelinhos/inference/jetson'
	rsync -az --delete --progress jetson/Makefile jetson/infer.cpp $(JETSON)
	rsync -az --delete --progress vonnx/models $(JETSON)
	ssh jetson-nano 'make -C ~/modelinhos/inference/jetson MODE=$(MODE)'

$(EXPORTED):
	python vonnx/export.py
	touch $@

models:
	rm -f $(EXPORTED)
	$(MAKE) $(EXPORTED)

.PHONY: rpi5 jetson-nano models
