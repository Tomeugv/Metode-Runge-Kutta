import numpy as np
import matplotlib.pyplot as plt
from rk import runge_kutta_45, y_analitic

def f(x, y):
    return - (1 / x**2) + 4 * (x - 6) * np.exp(-2 * (x - 6)**2)

def fehlberg_adaptatiu(x0, y0, x_end, h_init, tol):
    x_vals = [x0]
    y_vals = [y0]
    h_vals = [h_init]
    err_vals = []
    x = x0
    y = y0
    h = h_init
    while x < x_end:
        if x + h > x_end:
            h = x_end - x
        y_rk4, y_rk5, err, *_ = runge_kutta_45(x, y, h)
        err_abs = abs(err)
        # Calcula el nou pas (sense limitar pel valor inicial)
        if err_abs != 0:
            h_nou = 0.9 * h * (tol / err_abs)**0.25
        else:
            h_nou = h
        # Evita valors massa petits
        if h_nou < 1e-8:
            h_nou = 1e-8
        # Accepta el pas si l'error és prou petit
        if err_abs <= tol:
            x += h
            y = y_rk4
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)
            err_vals.append(err_abs)
        h = h_nou
    return x_vals, y_vals, h_vals, err_vals

def plot_rkf_adaptatiu(x0=1, y0=1, x_end=10, h_init=0.1, tol=1e-6):
    x_vals, y_vals, h_vals, err_vals = fehlberg_adaptatiu(x0, y0, x_end, h_init, tol)
    plt.plot(x_vals, y_vals, '.', label='RKF Fehlberg adaptatiu')
    plt.plot(x_vals, [y_analitic(x) for x in x_vals], '--', label='Solució analítica')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RKF Fehlberg adaptatiu')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.plot(x_vals, h_vals, marker='o', label='h en funció de x')
    plt.xlabel('x')
    plt.ylabel('h')
    plt.title('Evolució del pas d\'integració')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.plot(x_vals[:-1], err_vals, marker='o', label='Error local')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Evolució de l\'error local')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_rkf_adaptatiu()