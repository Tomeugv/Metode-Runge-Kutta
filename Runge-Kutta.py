import matplotlib.pyplot as plt
import numpy as np

# Definim la funció de la captura (canvi de signe per y')
def f(x, y):
    return - (1 / x**2) - 4 * (x - 6) * np.exp(-2 * (x - 6)**2)

def runge_kutta_45(x0, y0, h):
    f0 = f(x0, y0)
    f1 = f(x0 + h/4, y0 + h/4 * f0)
    f2 = f(x0 + 3*h/8, y0 + 3*h/32 * f0 + 9*h/32 * f1)
    f3 = f(x0 + 12*h/13, y0 + 1932*h/2197 * f0 + 7200*h/2197 * f1 + 7296*h/2197 * f2)
    f4 = f(x0 + h, y0 + 439*h/216 * f0 - 8*h * f1 + 3680*h/513 * f2 - 845*h/4104 * f3)
    f5 = f(x0 + h/2, y0 + 8*h/27 * f0 + 2*h * f1 + 3544*h/2565 * f2 + 1859*h/4104 * f3 - 11*h/40 * f4)

    y_rk4 = y0 + h * (25*f0/216 + 1408*f2/2565 + 2197*f3/4104 - f4/5)
    y_rk5 = y0 + h * (16*f0/135 + 6656*f2/12825 + 28561*f3/56430 - 9*f4/50 + 2*f5/55)

    return y_rk4, y_rk5

# Paràmetres
x0 = 1
y0 = 1
x_end = 10
h = 0.1

x_vals = [x0]
y_rk4_vals = [y0]
y_rk5_vals = [y0]

x = x0
y_rk4 = y0
y_rk5 = y0

while x < x_end:
    y_rk4, y_rk5 = runge_kutta_45(x, y_rk4, h)
    x += h
    x_vals.append(x)
    y_rk4_vals.append(y_rk4)
    y_rk5_vals.append(y_rk5)

# Solució analítica (donada per l'enunciat, si la tens)
def y_analitic(x):
    # Si no la tens, pots deixar-ho en blanc o posar una aproximació si la coneixes
    return 1/x + np.exp(-2*(x-6)**2)

plt.plot(x_vals, y_rk4_vals, label='RK4')
plt.plot(x_vals, y_rk5_vals, label='RK5')
plt.plot(x_vals, [y_analitic(x) for x in x_vals], '--', label='Solució analítica')
plt.xlabel('x')
plt.ylabel('y')
plt.title('RKF per la funció de la captura')
plt.legend()
plt.grid(True)
plt.show()