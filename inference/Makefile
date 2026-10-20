MODE ?= fp32
REMOTE = rpi5:~/modelinhos/inference/vonnx/
EXPORTED = vonnx/models/.exported

rpi5: $(EXPORTED)
	ssh rpi5 'mkdir -p ~/modelinhos/inference/vonnx'
	rsync -az --delete --progress vonnx/Makefile vonnx/infer.cpp $(REMOTE)
	rsync -az --delete --progress vonnx/models $(REMOTE)
	ssh rpi5 'make -C ~/modelinhos/inference/vonnx'

$(EXPORTED):
	python vonnx/export.py
	touch $@

models:
	rm -f $(EXPORTED)
	$(MAKE) $(EXPORTED)

.PHONY: rpi5 models
