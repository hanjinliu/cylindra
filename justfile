set shell := ["powershell.exe", "-c"]

doc:
	sphinx-apidoc -f -o ./rst/apidoc ./cylindra
	sphinx-build -b html ./rst ./docs

run:
	python ./cylindra/__main__.py

dev:
	maturin develop --release
