import numpy as np
import matplotlib.pyplot as plt

def RK2(f, g, x, y1, y2, h):    
    f0 = f(x, y1, y2)
    g0 = g(x, y1, y2)
    f1 = f(x + h / 4, y1 + f0 * h / 4, y2 + g0 * h / 4)
    g1 = g(x + h / 4, y1 + f0 * h / 4, y2 + g0 * h / 4)
    f2 = f(x + 3 * h / 8, y1 + h * f0 * 3 / 32 + h * f1 * 9 / 32, y2 + h * g0 * 3 / 32 + h * g1 * 9 / 32)
    g2 = g(x + 3 * h / 8, y1 + h * f0 * 3 / 32 + h * f1 * 9 / 32, y2 + h * g0 * 3 / 32 + h * g1 * 9 / 32)
    f3 = f(x + 12 * h / 13, y1 + h * f0 * 1932 / 2197 - h * f1 * 7200 / 2197 + h * f2 * 7296 / 2197,
           y2 + h * g0 * 1932 / 2197 - h * g1 * 7200 / 2197 + h * g2 * 7296 / 2197)
    g3 = g(x + 12 * h / 13, y1 + h * f0 * 1932 / 2197 - h * f1 * 7200 / 2197 + h * f2 * 7296 / 2197,
           y2 + h * g0 * 1932 / 2197 - h * g1 * 7200 / 2197 + h * g2 * 7296 / 2197)
    f4 = f(x + h, y1 + h * f0 * 439 / 216 - h * f1 * 8 + h * f2 * 3680 / 513 - h * f3 * 845 / 4104,
           y2 + h * g0 * 439 / 216 - h * g1 * 8 + h * g2 * 3680 / 513 - h * g3 * 845 / 4104)
    g4 = g(x + h, y1 + h * f0 * 439 / 216 - h * f1 * 8 + h * f2 * 3680 / 513 - h * f3 * 845 / 4104,
           y2 + h * g0 * 439 / 216 - h * g1 * 8 + h * g2 * 3680 / 513 - h * g3 * 845 / 4104)
    f5 = f(x + h / 2, y1 - h * f0 * 8 / 27 + h * f1 * 2 - h * f2 * 3544 / 2565 + h * f3 * 1859 / 4104 - h * f4 * 11 / 40,
           y2 - h * g0 * 8 / 27 + h * g1 * 2 - h * g2 * 3544 / 2565 + h * g3 * 1859 / 4104 - h * g4 * 11 / 40)
    g5 = g(x + h / 2, y1 - h * f0 * 8 / 27 + h * f1 * 2 - h * f2 * 3544 / 2565 + h * f3 * 1859 / 4104 - h * f4 * 11 / 40,
           y2 - h * g0 * 8 / 27 + h * g1 * 2 - h * g2 * 3544 / 2565 + h * g3 * 1859 / 4104 - h * g4 * 11 / 40)
    y1_next = y1 + h * (f0 * 16 / 135 + f2 * 6656 / 12825 + f3 * 28561 / 56430 - f4 * 9 / 50 + f5 * 2 / 55)
    y2_next = y2 + h * (g0 * 16 / 135 + g2 * 6656 / 12825 + g3 * 28561 / 56430 - g4 * 9 / 50 + g5 * 2 / 55)
    err1 = h * abs(1 / 360 * f0 - 128 / 4275 * f2 - 2197 / 75240 * f3 + 1 / 50 * f4 + 2 / 55 * f5)
    err2 = h * abs(1 / 360 * g0 - 128 / 4275 * g2 - 2197 / 75240 * g3 + 1 / 50 * g4 + 2 / 55 * g5)
    return y1_next, y2_next, err1, err2

def solve_van_der_pol(x0, dx0, t_end, h_init, tol, mu=1.0, adaptatiu=True):
    t_vals = [0]
    x_vals = [x0]
    dx_vals = [dx0]
    h_vals = [h_init]
    h = h_init
    while t_vals[-1] < t_end:
        x, dx, err_x, err_dx = RK2(
            lambda t, x, dx: dx,
            lambda t, x, dx: mu * (1 - x**2) * dx - x,
            t_vals[-1], x_vals[-1], dx_vals[-1], h
        )
        if adaptatiu and max(err_x, err_dx) > tol:
            h = 0.9 * h * (tol / max(err_x, err_dx))**0.25
        else:
            t_vals.append(t_vals[-1] + h)
            x_vals.append(x)
            dx_vals.append(dx)
            h_vals.append(h)
    return t_vals, x_vals, dx_vals, h_vals

def plot_results(t_vals, x_vals, dx_vals, h_vals, title_suffix=""):
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, x_vals, label="x(t) (Posició)")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title(f"Oscil·lador de Van der Pol - Posició {title_suffix}")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, dx_vals, label="dx(t) (Velocitat)")
    plt.xlabel("t")
    plt.ylabel("dx(t)")
    plt.title(f"Oscil·lador de Van der Pol - Velocitat {title_suffix}")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, dx_vals, label="Diagrama de fases")
    plt.xlabel("x")
    plt.ylabel("dx")
    plt.title(f"Oscil·lador de Van der Pol - Diagrama de fases {title_suffix}")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Cas 1: RKF adaptatiu
    x0 = 2.0
    dx0 = 0.0
    t_end = 25
    h_init = 0.1
    tol = 1e-10
    mu = 4.0
    t_vals, x_vals, dx_vals, h_vals = solve_van_der_pol(x0, dx0, t_end, h_init, tol, mu, adaptatiu=True)
    plot_results(t_vals, x_vals, dx_vals, h_vals, title_suffix="(RKF adaptatiu)")

    # Cas 2: RKF amb pas fix
    x0 = 0.01
    dx0 = 0.01
    t_end = 25
    h_init = 0.5
    tol = 1e-10
    mu = 1.0
    t_vals, x_vals, dx_vals, h_vals = solve_van_der_pol(x0, dx0, t_end, h_init, tol, mu, adaptatiu=False)
    plot_results(t_vals, x_vals, dx_vals, h_vals, title_suffix="(Pas fix)")