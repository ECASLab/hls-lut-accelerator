""""
Autor: Patrick Hugo Nepveu Nelson <patrick.cr1405@gmail.com>
Año: 2026 ECASLab
"""

import numpy as np
import matplotlib.pyplot as plt

#Funciones no-lineales
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def relu(x):
    return np.maximum(0, x)

def exponential(x):
    return np.exp(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanhiper(x):
    return np.tanh(x)

def swish(x,beta=1):
    return x / (1 + np.exp(-beta * x))



FUNCIONES = {
    "gelu": gelu, 
    "relu": relu,
    "exp": exponential,
    "sig": sigmoid,
    "tanh": tanhiper,
    "swish": swish
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
            b = func(x0) #intercept local
            
            
            # Verificar el error en el punto medio (donde suele ser máximo)  
            """"
            x_medio = x0 + (ancho / 2)
            y_real = func(x_medio)
            y_aprox = m * (x_medio - x0) + b
            error_calculado = abs(y_real - y_aprox)
            """
            
            x_test = np.linspace(x0,x1,20)
            y_real = func(x_test)
            y_aprox = m * (x_test - x0) + b   #point-slope
            error_calculado = np.max(np.abs(y_real - y_aprox))
            
            
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





def lut_hls_coeffs_header(cortes, m_list, b_list, bit_width=12, int_width=6, filename="lut_coeffs.h"):

    

    frac_bits = bit_width - int_width
    
    #paso mínimo, equivale a 2^-6
    STEP_MIN = 0.015625  
    #cantidad de slots uniformes que caben en el rango 
    NUM_SLOTS = int((RANGO_MAX - RANGO_MIN) / STEP_MIN)

    #LUT uniforme 
    segment_map = []
    for i in range(NUM_SLOTS):
        x_val = RANGO_MIN + (i * STEP_MIN) + (STEP_MIN / 2)
        found = False
        for seg_idx in range(len(cortes) - 1):
            if x_val >= cortes[seg_idx] and x_val < cortes[seg_idx+1]:
                segment_map.append(seg_idx)
                found = True
                break
        if not found: segment_map.append(len(m_list) - 1)

    def to_fixed_hex(val):
        # Multiplicar por 2^frac_bits para eliminar parte fraccionaria
        scaled = int(round(val * (2**frac_bits)))
        # Two's Complement con base a bit_width
        if scaled < 0:
            scaled = (1 << bit_width) + scaled
        
        #división por exceso (A+B-1)//B
        hex_len = (bit_width + 3) // 4
        #formato HEX de python - 0:{hex_len} es para padding con 0s y X para la conversión
        return f"0x{scaled & ((1 << bit_width) - 1):0{hex_len}X}"
    
    x_min_raw = to_fixed_hex(RANGO_MIN)
    x_max_raw = to_fixed_hex(RANGO_MAX)

    with open(filename, "w") as f:
        f.write(f"// Autor: Patrick Hugo Nepveu Nelson <patrick.cr1405@gmail.com>\n")
        f.write(f"// Año: 2026 ECASLab\n")
        f.write(f"// Generado para <{bit_width},{int_width}> \n \n")
        f.write("#ifndef LUT_COEFFS_H\n#define LUT_COEFFS_H\n\n")
        

        f.write(f"#define EXP_FUNC  {1 if funcion == 'exp' else 0}\n")
        f.write(f"#define SIG_FUNC  {1 if funcion == 'sig' else 0}\n")
        f.write(f"#define TANH_FUNC {1 if funcion == 'tanh' else 0}\n")
        f.write(f"#define GELU_FUNC {1 if funcion == 'gelu' else 0}\n")
        f.write(f"#define RELU_FUNC {1 if funcion == 'relu' else 0}\n\n")
        f.write(f"#define SWISH_FUNC {1 if funcion == 'swish' else 0}\n\n")


        f.write(f"#define X_MIN {x_min_raw}\n") 
        f.write(f"#define X_MAX {x_max_raw}\n")
        f.write(f"#define ACTUAL_SEG {len(m_list)}\n")
        f.write(f"#define MAP_SIZE {NUM_SLOTS}\n\n")
        f.write(f"#define MAX_SEG 128 \n\n")

 
        f.write("const uint8_t segment_map[MAP_SIZE] = {\n    ") # con MAX_SEG en 128, con uint8_t se puede guardar hasta 255
        f.write(", ".join([str(x) for x in segment_map]) + "\n};\n\n")
        storage_type = "uint16_t" if bit_width <= 16 else "uint32_t" if bit_width <= 32 else "uint64_t"
        f.write(f"const {storage_type} lut_x0_raw[MAX_SEG] = {{\n    ")
        f.write(", ".join([to_fixed_hex(x) for x in cortes[:-1]]) + "};\n\n")  
        f.write(f"const {storage_type} lut_m_raw[MAX_SEG] = {{\n    ")
        f.write(", ".join([to_fixed_hex(m) for m in m_list]) + "};\n\n")    
        f.write(f"const {storage_type} lut_b_raw[MAX_SEG] = {{\n    ")
        f.write(", ".join([to_fixed_hex(b) for b in b_list]) + "};\n\n")   
        f.write("#endif\n")



# Parámetros
ERROR_ADMITIDO = 0.005# Precisión deseada 
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

