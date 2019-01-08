! (C) Copyright 2009-2016 ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Solve a Helmholz equation.

!> This routine solves the Helmholz equation:
!! \f$ \nabla^2 \psi + c \psi = b \f$,
!! where \f$ c \f$ is a constant, and where \f$ \nabla^2 \f$ is the
!! two-dimensional, 5-point finite-difference Laplacian.
!!
!! The solution method is to apply an FFT in the zonal direction. This
!! reduces the problem to a set of un-coupled tri-diagonal systems that
!! are solved with the standard back-substitution algorithm.

subroutine solve_helmholz (x,b,c,nx,ny,deltax,deltay)

implicit none

integer, intent(in) :: nx     !< Zonal grid dimension
integer, intent(in) :: ny     !< Meridional grid dimension
real(8), intent(out) :: x(nx,ny) !< Solution
real(8), intent(in) :: b(nx,ny)  !< Right hand side
real(8), intent(in) :: c         !< Coefficient in the linear operator
real(8), intent(in) :: deltax    !< Zonal grid spacing (non-dimensional)
real(8), intent(in) :: deltay    !< Meridional grid spacing (non-dimensional)

integer, dimension(13) :: ifax
real(8) :: trigs(3*nx/2+1)
real(8) :: work(ny*(nx+2))
real(8) :: v(ny), z(ny), bext(nx+2,ny), xext(nx+2,ny)
real(8) :: pi, am, bm, w
integer k, i, iri

x(:,:) = 0.0_8

if (maxval(abs(b))>0.0_8) then

  !--- initialise the FFT
  call set99 (trigs,ifax,nx)

  !--- transform

  bext(1:nx,:)=b(:,:)
  bext(nx+1:nx+2,:)=0.0_8
  call fft991 (bext,work,trigs,ifax,1,nx+2,nx,ny,-1)

  pi = 4.0_8*atan(1.0_8)

  !--- loop over wavenumber

  do k=0,nx/2

    !--- solve the tri-diagonal systems

    am = c + 2.0_8*( cos( 2.0_8*real(k,8)*pi &
                                 /real(nx,8)) &
                            -1.0_8)/(deltax*deltax) &
           - 2.0_8/(deltay*deltay)

    bm = 1.0_8/(deltay*deltay)

    do iri=1,2
      v(1) = bm/am
      z(1) = bext(2*k+iri,1)/am

      do i=2,ny
        w    = am-bm*v(i-1)
        v(i) = bm/w
        z(i) = (bext(2*k+iri,i)-bm*z(i-1))/w
      enddo

      xext(2*k+iri,ny) = z(ny)
      do i=ny-1,1,-1
         xext(2*k+iri,i) = z(i) - v(i)*xext(2*k+iri,i+1)
      enddo
    enddo
  enddo

  !--- transform back
  call fft991 (xext,work,trigs,ifax,1,nx+2,nx,ny,+1)
  x(:,:) = xext(1:nx,:)

endif

end subroutine solve_helmholz
