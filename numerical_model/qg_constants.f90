! (C) Copyright 2009-2016 ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Constants for the QG model

module qg_constants

implicit none

!--- Model dimensions
integer, parameter :: nx = 120
integer, parameter :: ny = 20

!--- Dimensional parameters

real(8),parameter :: domain_zonal=12e6_8  !< model domain (m) in zonal direction
real(8),parameter :: domain_meridional=6.3e6_8  !< meridional model domain (m)
real(8),parameter :: scale_length = 1e6_8 !< horizontal length scale (m)
real(8),parameter :: ubar = 10.0_8       !< typical verlocity (m/s)
real(8),parameter :: ubar1 = 40.0_8      !< mean zonal wind in the top layer (m/s)
real(8),parameter :: ubar2 = 10.0_8      !< mean zonal wind in the bottom layer (m/s)
real(8),parameter :: dlogtheta = 0.1_8   !< difference in log(pot. T) across boundary
real(8),parameter :: g=10.0_8            !< gravity (m^2 s^{-2})
real(8),parameter :: f0 = 1e-4_8         !< Coriolis parameter at southern boundary
real(8),parameter :: bet0 = 1.5e-11_8    !< Meridional gradient of f (s^{-1} m^{-1})
real(8),parameter :: horog = 2000.0_8    !< height of orography (m)
real(8),parameter :: worog = 1000e3_8    !< e-folding width of orography (m)
real(8),parameter :: d1 = 5500.0_8
real(8),parameter :: d2 = 4500.0_8

!--- Non-dimensional parameters

real(8),parameter :: u1 = ubar1/ubar
real(8),parameter :: u2 = ubar2/ubar
real(8),parameter :: bet = bet0*scale_length*scale_length/ubar
real(8),parameter :: rossby_number = ubar/(f0*scale_length)

!--- Additional parameters

real(8),parameter :: f1 = f0*f0*scale_length*scale_length/(g*dlogtheta*d1)
real(8),parameter :: f2 = f0*f0*scale_length*scale_length/(g*dlogtheta*d2)
real(8),parameter :: rsmax = horog/(rossby_number*d2)
real(8),parameter :: deltax0 = domain_zonal/real(nx,8)
real(8),parameter :: deltay0 = domain_meridional/real(ny+1,8)
real(8),parameter :: deltax = deltax0/scale_length
real(8),parameter :: deltay = deltay0/scale_length

end module qg_constants
