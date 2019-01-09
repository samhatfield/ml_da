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

subroutine prepare_integration(x, x_north, x_south, rs, q_out, u_out, v_out)

use qg_constants, only: nx, ny, f1, f2, deltax, deltay, bet

implicit none

!f2py integer, intent(aux) :: nx, ny
real(8), intent(in) :: x(nx,ny,2)
real(8), intent(in) :: x_north(2)
real(8), intent(in) :: x_south(2)
real(8), intent(in) :: rs(nx,ny)

real(8), intent(out) :: q_out(nx,ny,2)
real(8), intent(out) :: u_out(nx,ny,2)
real(8), intent(out) :: v_out(nx,ny,2)

! -- calculate potential vorticity and wind components

call calc_pv(nx, ny, x, x_north, x_south, f1, f2, deltax, deltay, bet, rs, q_out)
call zonal_wind(u_out, x, x_north, x_south, nx, ny, deltay)
call meridional_wind(v_out, x, nx, ny, deltax)

end subroutine prepare_integration

!> Perform a timestep of the QG model

!> This routine is called from C++ to propagate the state.
!!
!!
!! The timestep starts by advecting the potential vorticity using
!! a semi-Lagrangian method. The potential vorticity is then inverted
!! to determine the streamfunction. Velocity components are then
!! calculated, which are used at the next timestep to advect the
!! potential vorticity.
!!
!! Note that the state visible to C++ contains potential vorticity and
!! wind components, in addition to streamfunction. This makes these
!! fields available for use in the observation operators.

subroutine propagate(q_in, q_north, q_south, x_north, x_south, u_in, v_in, rs,&
    & q_out, x_out, u_out, v_out)

use qg_constants, only: nx, ny, deltax, deltay, dt, f1, f2, bet

implicit none

real(8), intent(in) :: q_in(nx,ny,2)
real(8), intent(in) :: q_north(nx,2)
real(8), intent(in) :: q_south(nx,2)
real(8), intent(in) :: x_north(2)
real(8), intent(in) :: x_south(2)
real(8), intent(in) :: u_in(nx,ny,2)
real(8), intent(in) :: v_in(nx,ny,2)
real(8), intent(in) :: rs(nx,ny)

real(8), intent(out) :: q_out(nx,ny,2)
real(8), intent(out) :: x_out(nx,ny,2)
real(8), intent(out) :: u_out(nx,ny,2)
real(8), intent(out) :: v_out(nx,ny,2)

! ------------------------------------------------------------------------------

!--- advect the potential vorticity

call advect_pv(q_out, q_in, q_north, q_south, u_in, v_in, nx, ny, deltax, deltay, dt)

!--- invert the potential vorticity to determine streamfunction

call invert_pv(x_out, q_out, x_north, x_south, rs, nx, ny, deltax, deltay, f1, f2, bet)

! -- calculate potential vorticity and wind components

call zonal_wind(u_out, x_out, x_north, x_south, nx, ny, deltay)
call meridional_wind(v_out, x_out, nx, ny, deltax)

! ------------------------------------------------------------------------------
return
end subroutine propagate

!> Calculate potential vorticity from streamfunction

!> Potential vorticity is defined as
!! \f{eqnarray*}{
!! q_1 &=& \nabla^2 \psi_1 - F_1 (\psi_1 -\psi_2 ) + \beta y \\\\
!! q_2 &=& \nabla^2 \psi_2 - F_2 (\psi_2 -\psi_1 ) + \beta y + R_s
!! \f}

subroutine calc_pv(kx,ky,x,x_north,x_south,f1,f2,deltax,deltay,bet,rs,pv)

!--- calculate potential vorticity from streamfunction

implicit none
integer, intent(in) :: kx           !< Zonal grid dimension
integer, intent(in) :: ky           !< Meridional grid dimension
real(8), intent(in) :: x(kx,ky,2)   !< Streamfunction
real(8), intent(in) :: x_north(2)   !< Streamfunction on northern wall
real(8), intent(in) :: x_south(2)   !< Streamfunction on southern wall
real(8), intent(in) :: f1           !< Coefficient in PV operator
real(8), intent(in) :: f2           !< Coefficient in PV operator
real(8), intent(in) :: deltax       !< Zonal grid spacing (non-dimensional)
real(8), intent(in) :: deltay       !< Meridional grid spacing (non-dimensional)
real(8), intent(in) :: bet          !< NS Gradient of Coriolis parameter
real(8), intent(in) :: rs(kx,ky)    !< Orography

real(8), intent(out)   :: pv(kx,ky,2)  !< Potential vorticity

integer :: jj
real(8) :: y

!--- apply the linear operator

call pv_operator(x,pv,kx,ky,f1,f2,deltax,deltay)

!--- add the contribution from the boundaries

pv(:,1 ,1) = pv(:,1 ,1) + (1.0_8/(deltay*deltay))*x_south(1)
pv(:,1 ,2) = pv(:,1 ,2) + (1.0_8/(deltay*deltay))*x_south(2)
pv(:,ky,1) = pv(:,ky,1) + (1.0_8/(deltay*deltay))*x_north(1)
pv(:,ky,2) = pv(:,ky,2) + (1.0_8/(deltay*deltay))*x_north(2)

!--- add the beta term

do jj=1,ky
  y = real(jj-(ky+1)/2,8)*deltay;
  pv(:,jj,:) = pv(:,jj,:) + bet*y
enddo

!--- add the orography/heating term

pv(:,:,2) = pv(:,:,2) + rs(:,:)

end subroutine calc_pv