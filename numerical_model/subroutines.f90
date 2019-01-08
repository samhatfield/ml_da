! (C) Copyright 2009-2016 ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Prepare for an integration of the QG model.

!> At the start of a timestep of the QG model, the state must contain
!! streamfunction, potential vorticity and wind components. The control
!! variable for the analysis, however, contains only streamfunction.
!! This routine calculates potential vorticity and wind from the
!! streamfunction, and is called before an integration of the QG model.

subroutine prepare_integration(q_in, x_in, x_north_in, x_south_in, rs,&
    & q_out, x_out, x_north_out, x_south_out)

use qg_constants, only: nx, ny, f1, f2, deltax, deltay, bet

implicit none

!f2py integer, intent(aux) :: nx, ny
real(8), intent(in) :: q_in(nx,ny,2)
real(8), intent(in) :: x_in(nx,ny,2)
real(8), intent(in) :: x_north_in(2)
real(8), intent(in) :: x_south_in(2)
real(8), intent(in) :: rs(nx,ny)

real(8), intent(out) :: q_out(nx,ny,2)
real(8), intent(out) :: x_out(nx,ny,2)
real(8), intent(out) :: x_north_out(2)
real(8), intent(out) :: x_south_out(2)

! -- set up output variables
q_out = q_in; x_out = x_in; x_north_out = x_north_in; x_south_out = x_south_in

! -- calculate potential vorticity and wind components

call calc_pv(nx, ny, q_out, x_out, x_north_out, x_south_out, f1, f2, deltax, deltay, bet, rs)
! call zonal_wind(flds%u,flds%x,flds%x_north,flds%x_south,flds%nx,flds%ny, &
!               & conf%deltay)
! call meridional_wind(flds%v,flds%x,flds%nx,flds%ny,conf%deltax)

end subroutine prepare_integration
