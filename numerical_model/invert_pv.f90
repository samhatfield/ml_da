! (C) Copyright 2009-2016 ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Invert potential vorticity, returning streamfunction

!> Streamfunction is determined from potential vorticity by subtracting
!! the beta and orography terms and then solving the resulting elliptic
!! equation.

subroutine invert_pv (x,pv,x_north,x_south,rs,nx,ny,deltax,deltay,F1,F2,bet)

!--- invert potential vorticity to get streamfunction

implicit none
integer, intent(in) :: nx          !< Zonal grid dimension
integer, intent(in) :: ny          !< Meridional grid dimension
real(8), intent(out) :: x(nx,ny,2)    !< Streamfunction
real(8), intent(in)  :: pv(nx,ny,2)   !< Potential vorticity
real(8), intent(in)  :: x_north(2)    !< Streamfunction on northern wall
real(8), intent(in)  :: x_south(2)    !< Streamfunction on southern wall
real(8), intent(in)  :: rs(nx,ny)     !< Orography
real(8), intent(in) :: deltax         !< Zonal grid spacing (non-dimensional)
real(8), intent(in) :: deltay         !< Meridional grid spacing (non-dimensional)
real(8), intent(in) :: F1             !< Coefficient in PV operator
real(8), intent(in) :: F2             !< Coefficient in PV operator
real(8), intent(in) :: bet            !< NS Gradient of Coriolis parameter

real(8) :: y
real(8) :: pv_nobc(nx,ny,2)
integer :: jj

!--- subtract the beta term and the orography/heating term

do jj=1,ny
  y = real(jj-(ny+1)/2,8)*deltay;
  pv_nobc(:,jj,1) = pv(:,jj,1) - bet*y
  pv_nobc(:,jj,2) = pv(:,jj,2) - bet*y -rs(:,jj)
enddo

!--- subtract the contribution from the boundaries

pv_nobc(:,1 ,1) = pv_nobc(:,1 ,1) - (1.0_8/(deltay*deltay))*x_south(1)
pv_nobc(:,1 ,2) = pv_nobc(:,1 ,2) - (1.0_8/(deltay*deltay))*x_south(2)
pv_nobc(:,ny,1) = pv_nobc(:,ny,1) - (1.0_8/(deltay*deltay))*x_north(1)
pv_nobc(:,ny,2) = pv_nobc(:,ny,2) - (1.0_8/(deltay*deltay))*x_north(2)

!--- Solve the elliptic system

call solve_elliptic_system (x,pv_nobc,nx,ny,deltax,deltay,F1,F2)

end subroutine invert_pv
