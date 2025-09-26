import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return - (1 / x**2) - 4 * (x - 6) * np.exp(-2 * (x - 6)**2)

def runge_kutta_45(x0, y0, h):
    f0 = f(x0, y0)
    f1 = f(x0 + h/4, y0 + h/4 * f0)
    f2 = f(x0 + 3*h/8, y0 + 3*h/32 * f0 + 9*h/32 * f1)
    f3 = f(x0 + 12*h/13, y0 + 1932*h/2197 * f0 + 7200*h/2197 * f1 + 7296*h/2197 * f2)
    f4 = f(x0 + h, y0 + 439*h/216 * f0 - 8*h * f1 + 3680*h/513 * f2 - 845*h/4104 * f3)
    f5 = f(x0 + h/2, y0 + 8*h/27 * f0 + 2*h * f1 + 3544*h/2565 * f2 + 1859*h/4104 * f3 - 11*h/40 * f4)

    # RK4
    y_rk4 = y0 + h * (25*f0/216 + 1408*f2/2565 + 2197*f3/4104 - f4/5)
    # RK5
    y_rk5 = y0 + h * (16*f0/135 + 6656*f2/12825 + 28561*f3/56430 - 9*f4/50 + 2*f5/55)
    # Fehlberg error
    err = h * (1/360*f0 - 128/4275*f2 - 2197/75240*f3 + 1/50*f4 + 2/55*f5)
    return y_rk4, y_rk5, err, f0, f2, f3, f4, f5

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
        y_rk4, y_rk5, err, f0, f2, f3, f4, f5 = runge_kutta_45(x, y, h)
        err_abs = abs(err)
        err_vals.append(err_abs)
        # Calcula h_nou segons la fórmula de la captura
        if err_abs != 0:
            h_nou = 0.9 * h * (tol / err_abs)**0.25
        else:
            h_nou = h
        # Si h_nou > h, empra h pel següent pas
        if h_nou > h:
            h_nou = h
        # Accepta el pas si l'error és prou petit
        if err_abs <= tol:
            x += h
            y = y_rk4
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)
        # Actualitza h pel següent pas
        h = h_nou
    return x_vals, y_vals, h_vals, err_vals

def y_analitic(x):
    return 1/x + np.exp(-2*(x-6)**2)

def menu():
    print("Escull el mètode:")
    print("1. RKF amb pas fix")
    print("2. RKF amb pas adaptatiu (Fehlberg)")
    opcio = input("Opció (1/2): ")
    return opcio

# Paràmetres
x0 = 1
y0 = 1
x_end = 10
h_init = 0.1
tol = 1e-6

opcio = menu()

if opcio == "1":
    x_vals = [x0]
    y_rk4_vals = [y0]
    y_rk5_vals = [y0]
    x = x0
    y_rk4 = y0
    y_rk5 = y0
    h = h_init
    while x < x_end:
        if x + h > x_end:
            h = x_end - x
        y_rk4, y_rk5, _, _, _, _, _, _ = runge_kutta_45(x, y_rk4, h)
        x += h
        x_vals.append(x)
        y_rk4_vals.append(y_rk4)
        y_rk5_vals.append(y_rk5)
    plt.plot(x_vals, y_rk4_vals, label='RK4 pas fix')
    plt.plot(x_vals, y_rk5_vals, label='RK5 pas fix')
    plt.plot(x_vals, [y_analitic(x) for x in x_vals], '--', label='Solució analítica')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RKF pas fix')
    plt.legend()
    plt.grid(True)
    plt.show()
elif opcio == "2":
    x_vals, y_vals, h_vals, err_vals = fehlberg_adaptatiu(x0, y0, x_end, h_init, tol)
    plt.plot(x_vals, y_vals, ',' label='RKF Fehlberg adaptatiu')
    plt.plot(x_vals, [y_analitic(x) for x in x_vals], '--', label='Solució analítica')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RKF Fehlberg adaptatiu')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Gràfic de l'evolució del pas d'integració
    plt.plot(x_vals, h_vals, label='h (pas d\'integració)')
    plt.xlabel('x')
    plt.ylabel('h')
    plt.title('Evolució del pas d\'integració')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Gràfic de l'error
    plt.plot(x_vals[:-1], err_vals, label='Error local')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Evolució de l\'error local')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()
else:
    print("Opció no vàlida.")