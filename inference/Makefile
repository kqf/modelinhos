MODE ?= fp32
REMOTE = rpi5:~/modelinhos/inference/vonnx/

rpi5: vonnx/models
	ssh rpi5 'mkdir -p ~/modelinhos/inference/vonnx'
	rsync -az --delete vonnx/Makefile vonnx/infer.cpp vonnx/models $(REMOTE)
	ssh rpi5 'make -C ~/modelinhos/inference/vonnx MODE=$(MODE)'

# Exported once; `make models` re-exports after the recipes change.
vonnx/models:
	$(MAKE) models

models:
	python vonnx/export.py

.PHONY: rpi5 models
