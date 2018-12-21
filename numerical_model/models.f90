subroutine step_one_layer(input, output)
    use params
    use lorenz96, only: ode_one_layer

    implicit none

    !f2py integer, intent(aux) :: nx
    real(8), dimension(nx), intent(in) :: input
    real(8), dimension(nx), intent(out) :: output
    real(8), dimension(nx) :: k1, k2, k3, k4

    ! 4th order Runge-Kutta
    k1 = ode_one_layer(input)
    k2 = ode_one_layer(input+0.5_8*dt*k1)
    k3 = ode_one_layer(input+0.5_8*dt*k2)
    k4 = ode_one_layer(input+dt*k3)

    output = input + (dt/6.0_8)*(k1 + 2.0_8*k2 + 2.0_8*k3 + k4)
end subroutine

subroutine step_two_layer(input, output)
    use params
    use lorenz96, only: ode_two_layer

    implicit none

    !f2py integer, intent(aux) :: n
    real(8), dimension(n), intent(in) :: input
    real(8), dimension(n), intent(out) :: output
    real(8), dimension(n) :: k1, k2, k3, k4

    ! 4th order Runge-Kutta
    k1 = ode_two_layer(input)
    k2 = ode_two_layer(input+0.5_8*dt*k1)
    k3 = ode_two_layer(input+0.5_8*dt*k2)
    k4 = ode_two_layer(input+dt*k3)

    output = input + (dt/6.0_8)*(k1 + 2.0_8*k2 + 2.0_8*k3 + k4)
end subroutine