# export.py needs the modelinhos package, so the sources go along with
# the benchmark; the copy has no .git, hence the pinned version.
rpi5:
	ssh rpi5 'mkdir -p ~/modelinhos'
	cd .. && rsync -azR pyproject.toml README.rst modelinhos inference/vonnx rpi5:~/modelinhos/
	ssh rpi5 'cd ~/modelinhos/inference/vonnx && python3 -m venv .venv \
		&& SETUPTOOLS_SCM_PRETEND_VERSION=0 make deps models PYTHON=.venv/bin/python \
		&& make MODE=fp32'

.PHONY: rpi5
