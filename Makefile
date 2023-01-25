doc:
	sphinx-apidoc -f -o ./rst/apidoc ./cylindra
	sphinx-build -b html ./rst ./docs

release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload --repository testpypi dist/*
	twine upload --repository pypi dist/*

build-ext:
	python setup.py build_ext --inplace
