PYFC = f2py3
FC = gfortran

all: models params

models: params.f90 lorenz96.f90
	$(PYFC) --overwrite-signature -h models.pyf $^ models.f90 -m $@
	$(PYFC) -c models.pyf $^ models.f90
	mv $@.cpython-34m.so models.so

params:
	$(PYFC) -c -m params params.f90
	mv $@.cpython-34m.so params.so

.PHONY: clean
clean:
	rm -f *.o *.mod *.so *.pyf *.c