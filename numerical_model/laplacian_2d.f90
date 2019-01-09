! (C) Copyright 2009-2016 ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Horizontal Laplacian operator

!> Nothing fancy here.
!! It's just the standard 5-point finite-difference Laplacian.

subroutine laplacian_2d (x,del2x,nx,ny,deltax,deltay)

!--- The 2d Laplacian

implicit none
integer, intent(in) :: nx         !< Zonal grid dimension
integer, intent(in) :: ny         !< Meridional grid dimension
real(8), intent(in)  :: x(nx,ny)     !< Streamfunction
real(8), intent(out) :: del2x(nx,ny) !< Result of applying Laplacian to x
real(8), intent(in) :: deltax        !< Zonal grid spacing (non-dimensional)
real(8), intent(in) :: deltay        !< Meridional grid spacing (non-dimensional)

!--- del-squared of the streamfunction (5-point laplacian)

del2x(:,:) = -2.0_8*( 1.0_8/(deltax*deltax) + 1.0_8/(deltay*deltay))*x(:,:)

del2x(1:nx-1,:) = del2x(1:nx-1,:) + (1.0_8/(deltax*deltax))*x(2:nx  ,:)
del2x(nx    ,:) = del2x(nx    ,:) + (1.0_8/(deltax*deltax))*x(1     ,:)
del2x(2:nx  ,:) = del2x(2:nx  ,:) + (1.0_8/(deltax*deltax))*x(1:nx-1,:)
del2x(1     ,:) = del2x(1     ,:) + (1.0_8/(deltax*deltax))*x(nx    ,:)

del2x(:,1:ny-1) = del2x(:,1:ny-1) + (1.0_8/(deltay*deltay))*x(:,2:ny  )
del2x(:,2:ny  ) = del2x(:,2:ny  ) + (1.0_8/(deltay*deltay))*x(:,1:ny-1)

end subroutine laplacian_2d
