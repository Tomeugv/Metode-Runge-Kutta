import numpy as np
from rk import runge_kutta_45

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
        if err_abs != 0:
            h_nou = 0.9 * h * (tol / err_abs)**0.25
        else:
            h_nou = h
        if h_nou > h:
            h_nou = h
        if err_abs <= tol:
            x += h
            y = y_rk4
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)
        h = h_nou
    return x_vals, y_vals, h_vals, err_vals

def y_analitic(x):
    return 1/x + np.exp(-2*(x-6)**2)