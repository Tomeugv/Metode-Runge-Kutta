import matplotlib.pyplot as plt

def f(x, y):
    return x  # f_0 = x

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
y0 = 0
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

plt.plot(x_vals, y_rk4_vals, label='RK4')
plt.plot(x_vals, y_rk5_vals, label='RK5')
plt.plot(x_vals, [0.5*x**2 - 0.5 for x in x_vals], '--', label='Solució exacta (y=0.5x²-0.5)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Runge-Kutta Ordre 4 i 5 per f(x, y) = x (rang 1 a 10)')
plt.legend()
plt.grid(True)
plt.show()