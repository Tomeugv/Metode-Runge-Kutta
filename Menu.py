from rk import plot_rk_pas_fix
from rkf import plot_rkf_adaptatiu
from Rfk_sistema import plot_results

def menu():
    print("Escull el mètode:")
    print("1. RKF amb pas fix")
    print("2. RKF amb pas adaptatiu (Fehlberg)")
    print("3. Oscil·lador de Van der Pol")
    opcio = input("Opció (1/2): ")
    return opcio

opcio = menu()

if opcio == "1":
    plot_rk_pas_fix()
elif opcio == "2":
    plot_rkf_adaptatiu()
elif opcio == "3":
    plot_results()
else:
    print("Opció no vàlida.")