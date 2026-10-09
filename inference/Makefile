rpi5:
	ssh rpi5 'mkdir -p ~/modelinhos/inference/onnx'
	rsync -az --exclude run.bin --exclude models onnx/ rpi5:~/modelinhos/inference/onnx/
	ssh rpi5 'cd ~/modelinhos/inference/onnx && make models && make MODE=fp32'

.PHONY: rpi5
