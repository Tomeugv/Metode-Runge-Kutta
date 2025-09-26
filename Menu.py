from rk import runge_kutta_45, y_analitic
from rkf import fehlberg_adaptatiu, y_analitic as y_analitic_rkf
import matplotlib.pyplot as plt

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
    plt.plot(x_vals, y_rk4_vals, '.', label='RK4 pas fix')
    plt.plot(x_vals, y_rk5_vals, '.', label='RK5 pas fix')
    plt.plot(x_vals, [y_analitic(x) for x in x_vals], '--', label='Solució analítica')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RKF pas fix')
    plt.legend()
    plt.grid(True)
    plt.show()
elif opcio == "2":
    x_vals, y_vals, h_vals, err_vals = fehlberg_adaptatiu(x0, y0, x_end, h_init, tol)
    plt.plot(x_vals, y_vals, '.', label='RKF Fehlberg adaptatiu')
    plt.plot(x_vals, [y_analitic_rkf(x) for x in x_vals], '--', label='Solució analítica')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RKF Fehlberg adaptatiu')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.plot(x_vals, h_vals, label='h (pas d\'integració)')
    plt.xlabel('x')
    plt.ylabel('h')
    plt.title('Evolució del pas d\'integració')
    plt.legend()
    plt.grid(True)
    plt.show()
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