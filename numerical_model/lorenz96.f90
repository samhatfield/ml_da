module lorenz96
    use params

    implicit none

contains
    pure function ode_one_layer(state) result(tend)
        real(8), intent(in) :: state(nx)
        real(8) :: tend(nx)

        ! Find derivative of each component separately
        tend = cshift(state,-1)*(cshift(state,1)-cshift(state,-2)) - state + f
    end function

    pure function ode_two_layer(state) result(tend)
        real(8), intent(in) :: state(nx+nx*ny)
        real(8) :: x(nx)
        real(8) :: y(nx*ny)
        real(8) :: tend(nx+nx*ny)

        ! Break up state vector into components
        x = state(:nx)
        y = state(nx+1:nx+nx*ny)

        ! Find derivative of each component separately
        tend(:nx) = dXdT(x, y)
        tend(nx+1:nx+nx*ny) = dYdT_two_layer(x, y)
    end function

    pure function ode_three_layer(state) result(tend)
        real(8), intent(in) :: state(n)
        real(8) :: x(nx)
        real(8) :: y(nx*ny)
        real(8) :: z(nx*ny*nz)
        real(8) :: tend(n)

        ! Break up state vector into components
        x = state(:nx)
        y = state(nx+1:nx+nx*ny)
        z = state(nx+nx*ny+1:)

        ! Find derivative of each component separately
        tend(:nx) = dXdT(x, y)
        tend(nx+1:nx+nx*ny) = dYdT_three_layer(x, y, z)
        tend(nx+nx*ny+1:) = dZdT(y, z)
    end function

    pure function dXdT(x, y)
         real(8), intent(in) :: x(nx)
         real(8), intent(in) :: y(nx*ny)
         real(8) :: dXdT(nx)
         real(8) :: sum_y(nx)

         ! Sum all y's for each x, making an nx length vector of y sums
         sum_y = sum_2d(reshape(y, (/ny,nx/)))

         dXdT = cshift(x,-1)*(cshift(x,1)-cshift(x,-2)) - x + f &
             & - (h*c/b)*sum_y
    end function

    pure function dYdT_two_layer(x, y) result(dYdT)
        real(8), intent(in) :: x(nx)
        real(8), intent(in) :: y(nx*ny)
        real(8) :: dYdT(nx*ny)
        real(8) :: x_rpt(nx*ny)
        integer :: k

        ! Repeat elements of x ny times
        x_rpt = (/ (x(1+(k-1)/ny), k = 1, nx*ny) /)

        dYdT = c*b*cshift(y,1)*(cshift(y,-1)-cshift(y,2)) - c*y
        dYdT = dYdT + (h*c/b)*x_rpt
    end function

    pure function dYdT_three_layer(x, y, z) result(dYdT)
        real(8), intent(in) :: x(nx)
        real(8), intent(in) :: y(nx*ny)
        real(8), intent(in) :: z(nx*ny*nz)
        real(8) :: dYdT(nx*ny)
        real(8) :: x_rpt(nx*ny)
        real(8) :: sum_z(nx*ny)
        integer :: k

        ! Repeat elements of x ny times
        x_rpt = (/ (x(1+(k-1)/ny), k = 1, nx*ny) /)

        ! Sum all z's for each y, making an nx*ny length vector of z sums
        sum_z = sum(reshape(z, (/nz,nx*ny/)), dim=1)

        dYdT = c*b*cshift(y,1)*(cshift(y,-1)-cshift(y,2)) - c*y
        dYdT = dYdT + (h*c/b)*x_rpt
        dYdT = dYdT - (h*e/d)*sum_z
    end function

    pure function dZdT(y, z)
        real(8), intent(in) :: y(nx*ny)
        real(8), intent(in) :: z(nx*ny*nz)
        real(8) :: dZdT(nx*ny*nz)
        real(8) :: y_rpt(nx*ny*nz)
        integer :: k

        ! Repeat elements of y n_z times
        y_rpt = (/ (y(1+(k-1)/nz), k = 1, nx*ny*nz) /)

        dZdT = e*d*cshift(z,-1)*(cshift(z,1)-cshift(z,-2)) - g_Z*e*Z
        dZdT = dZdT + (h*e/d)*y_rpt
    end function

    pure function sum_2d(array)
        real(8), intent(in) :: array(:,:)
        real(8) :: sum_2d(size(array, 2))
        integer :: i, n

        n = size(array, 1)

        sum_2d(:) = 0.0_8

        do i = 1, n
            sum_2d = sum_2d + array(i, :)
        end do
    end function
end module