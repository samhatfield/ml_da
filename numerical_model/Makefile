PYFC = f2py
FC = gfortran

subroutines: qg_constants.f90 calc_pv.f90 pv_operator.f90 laplacian_2d.f90
	$(PYFC) --overwrite-signature -h subroutines.pyf $^ subroutines.f90 -m $@
	$(PYFC) -c subroutines.pyf $^ subroutines.f90
	mv $@.*.so subroutines.so

qg_constants:
	$(PYFC) -c -m qg_constants qg_constants.f90
	mv $@.*.so qg_constants.so

.PHONY: clean
clean:
	rm -f *.o *.mod *.so *.pyf *.c
