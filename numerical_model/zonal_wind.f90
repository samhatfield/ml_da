! (C) Copyright 2009-2016 ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Calculate zonal wind component from the streamfunction

!> Nothing fancy.
!! It's just a standard centred finite difference.

subroutine zonal_wind (u,x,x_north,x_south,nx,ny,deltay)

!--- calculate zonal wind component

implicit none

integer, intent(in) :: nx        !< Zonal grid dimension
integer, intent(in) :: ny        !< Meridional grid dimension
real(8), intent(out) :: u(nx,ny,2)  !< Zonal wind
real(8), intent(in)  :: x(nx,ny,2)  !< Streamfunction
real(8), intent(in)  :: x_north(2)  !< Streamfunction on northern wall
real(8), intent(in)  :: x_south(2)  !< Streamfunction on southern wall
real(8), intent(in) :: deltay       !< Meridional grid spacing (non-dimensional)

u(:,2:ny  ,:) =                 (0.5_8/deltay)*x(:,1:ny-1,:)
u(:,1     ,1) =                 (0.5_8/deltay)*x_south(1)
u(:,1     ,2) =                 (0.5_8/deltay)*x_south(2)
u(:,1:ny-1,:) = u(:,1:ny-1,:) - (0.5_8/deltay)*x(:,2:ny  ,:)
u(:,ny    ,1) = u(:,ny    ,1) - (0.5_8/deltay)*x_north(1)
u(:,ny    ,2) = u(:,ny    ,2) - (0.5_8/deltay)*x_north(2)

end subroutine zonal_wind
