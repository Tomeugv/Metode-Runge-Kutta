from rk import plot_rk_pas_fix
from rkf import plot_rkf_adaptatiu

def menu():
    print("Escull el mètode:")
    print("1. RKF amb pas fix")
    print("2. RKF amb pas adaptatiu (Fehlberg)")
    opcio = input("Opció (1/2): ")
    return opcio

opcio = menu()

if opcio == "1":
    plot_rk_pas_fix()
elif opcio == "2":
    plot_rkf_adaptatiu()
else:
    print("Opció no vàlida.")