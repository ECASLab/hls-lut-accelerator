import numpy as np
import matplotlib.pyplot as plt


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def relu(x):
    return np.maximum(0, x)

def exponential(x):
    return np.exp(x)

FUNCIONES = {
    "gelu": gelu, 
    "relu": relu,
    "exp": exponential
}

funcion = "exp"
funcion_activa = FUNCIONES['exp']

def pwl_no_uniforme(func, x_inicio, x_fin, error_max):
    lut_m = []          # Lista para las pendientes
    lut_b = []          # Lista para los intercepts
    puntos_corte = [x_inicio]
    
    x_actual = x_inicio
    print(f"Función no-lineal aproximada: {func.__name__}")
    print(f"{'Tramo':<8} | {'Inicio':<8} | {'Fin':<8} | {'Ancho':<8} | {'Error':<10}")
    print("-" * 55)

    while x_actual < x_fin: # While cubre todo el rango
        # Se inicia con ancho 2.0 - Se quiere empezar con un ancho grande para usar la menor cantidad de memoria (greedy)
        ancho = 2.0 
        encontrado = False
        
        while not encontrado:
            x0 = x_actual
            x1 = x_actual + ancho
            
            # m = (y1 - y0) / ancho
            m = (func(x1) - func(x0)) / ancho
            b = func(x0) #
            
            # Verificar el error en el punto medio (donde suele ser máximo)
            x_medio = x0 + (ancho / 2)
            y_real = func(x_medio)
            y_aprox = m * (x_medio - x0) + b
            error_calculado = abs(y_real - y_aprox)
            
            """"
            x_test = np.linspace(x0,x1,20)
            y_real = func(x_test)
            y_aprox = m * (x_test - x0) + b
            error_calculado = np.max(np.abs(y_real - y_aprox))
            """
            
            # Criterio de decisión
            # Si el error es aceptable, el ancho ya es el mínimo permitido
            if error_calculado <= error_max or ancho <= 0.015625:
                lut_m.append(m)
                lut_b.append(b)
                
                print(f"{len(lut_m):<8} | {x0:<8.3f} | {x1:<8.3f} | {ancho:<8.4f} | {error_calculado:<10.6f}")
                
                x_actual = x1
                puntos_corte.append(x_actual)
                encontrado = True
            else:
                # Si el error es mucho, se reduce el ancho a la mitad (mantiene potencia de 2)
                ancho /= 2.0
                
    return lut_m, lut_b, puntos_corte


def lut_hls_coeffs_header(cortes, m_list, b_list, filename="lut_coeffs.h"):
    x0_list = cortes[:-1]
    num_seg = len(m_list)

    with open(filename, "w") as f:
        f.write("#ifndef LUT_COEFFS_H\n#define LUT_COEFFS_H\n\n")
        f.write(f"// Generado para: {funcion.upper()} \n")
        f.write(f"#define MAX_SEG 128\n")
        f.write(f"#define ACTUAL_SEG {num_seg}\n") 
        f.write(f"#define X_MIN {RANGO_MIN}f\n")
        f.write(f"#define X_MAX {RANGO_MAX}f\n")
        is_exp = 1 if funcion == "exp" else 0
        f.write(f"#define EXP_FUNC {is_exp}\n\n")
        f.write("const float lut_x0[MAX_SEG] = {" + ", ".join([f"{x:.6f}f" for x in x0_list]) + ", " + ", ".join(["0.0f"]*(128-num_seg)) + "};\n")
        f.write("const float lut_m[MAX_SEG] = {" + ", ".join([f"{x:.6f}f" for x in m_list]) + ", " + ", ".join(["0.0f"]*(128-num_seg)) + "};\n")
        f.write("const float lut_b[MAX_SEG] = {" + ", ".join([f"{x:.6f}f" for x in b_list]) + ", " + ", ".join(["0.0f"]*(128-num_seg)) + "};\n")
        f.write("\n#endif")

# Parámetros
ERROR_ADMITIDO = 0.01 # Precisión deseada 
# Rango 
RANGO_MIN = -4.0        
RANGO_MAX = 4.0         

# Ejecución del algoritmo
pendientes, interceptos, cortes = pwl_no_uniforme(funcion_activa, RANGO_MIN, RANGO_MAX, ERROR_ADMITIDO)

#Generar header (.h) de los coeficientes de interpolación
lut_hls_coeffs_header(cortes, pendientes, interceptos)

print(f"\n=== TABLA DE MEMORIA (Look-Up Table)")
print(f"{'Segmento':<10} | {'Breakpoint (x0)':<18} | {'Pendiente (m)':<18} | {'Intercepto (b)':<18}")
print("-" * 70)

for i in range(len(pendientes)):
    # cortes[i] es x0, pendientes[i] es m, interceptos[i] es b
    print(f"Seg {i:<6} | {cortes[i]:<18.6f} | {pendientes[i]:<18.6f} | {interceptos[i]:<18.6f}")


y_cortes = [funcion_activa(x) for x in cortes]

plt.figure(figsize=(10, 6))

plt.plot(cortes, y_cortes, 
         color='blue', 
         linestyle='-', 
         linewidth=2, 
         marker='o',             
         markersize=6, 
         markerfacecolor='red', 
         label='Aproximación PWL (Segmentos Rectos)')

# Dibujar la curva real de fondo en gris claro para comparar
x_real = np.linspace(RANGO_MIN, RANGO_MAX, 500)
plt.plot(x_real, funcion_activa(x_real), color='black', alpha=0.2, linewidth=4, 
         label=f'Función {funcion.upper()} Real', zorder=0)

plt.title(f"Aproximación PWL no uniforme: {funcion.upper()}")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Mostrar texto con la cantidad de segmentos
plt.text(-3.8, max(y_cortes)*0.9, f"Total Segmentos: {len(pendientes)}", 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.show()

print("\n" + "="*30)
print(f"RESULTADO FINAL:")
print(f"Total de segmentos generados: {len(pendientes)}")
print("="*30)

