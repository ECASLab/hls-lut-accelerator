/*
Autor: Patrick Hugo Nepveu Nelson <patrick.cr1405@gmail.com>
Año: 2026 ECASLab
*/

#include "pwl_no_uniforme.h"
#include "lut_coeffs.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>

// Número de vectores de prueba
#ifndef NUM_TEST_VECTORS
#define NUM_TEST_VECTORS 50
#endif

// Error máximo permitido
#ifndef ERROR_ADMITIDO
#define ERROR_ADMITIDO 0.005 
#endif

// Interpretación de bits crudos a fixed-point
float raw_to_float(uint32_t raw_val) {
    DataT temp;
    temp.V = raw_val;
    return (float)temp;
}

// Golden Reference para las funciones no-lineales
float get_pointwise_golden(float x, float f_min, float f_max) {

    // Clamping para mantener a x entre f_min y f_max
    // condition ? value_if_true : value_if_false;
    float x_clamped = (x > f_max) ? f_max : (x < f_min) ? f_min : x;
    
    if (SIG_FUNC)   return 1.0f / (1.0f + std::exp(-x_clamped));
    if (TANH_FUNC)  return std::tanh(x_clamped);
    if (GELU_FUNC)  return 0.5f * x_clamped * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x_clamped + 0.044715f * std::pow(x_clamped, 3))));
    if (RELU_FUNC)  return (x_clamped > 0.0f) ? x_clamped : 0.0f;
    if (SWISH_FUNC) return x_clamped / (1.0f + std::exp(-x_clamped));
    if (EXP_FUNC)   return std::exp(x_clamped); // se calcula exp solamente si mode = 0
    return 0.0f;
}

// Golden Reference para softmax
void get_softmax_golden(const std::vector<float>& input, std::vector<float>& output) {
    // Encuentra máximo - max_element retorna un puntero a la posición del valor máximo, por lo que hay que hacer dereference
    float max_val = *std::max_element(input.begin(), input.end());
    double sum = 0.0;
    std::vector<double> exps(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        // Safe softmax - al restar max_val se asegura un rango entre 0 y 1 
        exps[i] = std::exp((double)input[i] - (double)max_val);
        sum += exps[i];
    }
    for (size_t i = 0; i < input.size(); ++i) {
        // Normalización final
        output[i] = (float)(exps[i] / sum);
    }
}


int main() {
    // Interpretación de limites a float
    float f_min = raw_to_float(X_MIN);
    float f_max = raw_to_float(X_MAX);
    // Tamaño total del vector
    const uint64_t vector_length = KROWS * KCOLS; 
    // Número necesarios de paquetes de 512 bits
    const uint64_t n_packets = vector_length / kPackets;

    // Mode 1 si se está usando exp
    int test_mode = (EXP_FUNC) ? 1 : 0;

    std::cout << "==================================================" << std::endl;
    std::cout << " TESTBENCH PWL" << std::endl;
    std::cout << " Rango detectado:  [" << f_min << ", " << f_max << "]" << std::endl;
    std::cout << " Modo:   " << (test_mode ? "Softmax" : "Pointwise") << std::endl;
    std::cout << " Vector de:  " << vector_length << " elementos" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Buffers para la simulación - se usa new para utilizar heap y no stack
    RawDataT* in_hw  = new RawDataT[n_packets];
    RawDataT* out_hw = new RawDataT[n_packets];

    // Generación de números entre f_min y f_max
    std::mt19937 gen(42); // semilla constante para reproducibilidad - Mersenne Twister
    std::uniform_real_distribution<float> dist(f_min, f_max);

    // Carga de los coeficientes con reload = true. No se procesa datos, solo cargar coeficientes
    pwl_no_uniforme(in_hw, out_hw, vector_length, segment_map, lut_x0_raw, 
                    lut_m_raw, lut_b_raw, ACTUAL_SEG, X_MIN, X_MAX, true, test_mode);


    // Prueba para testear la funcionalidad de reload - se crean variables dummy g_map y g_lut de puros 0s
    uint8_t g_map[MAP_SIZE] = {0};
    RawStorageT g_lut[MAX_SEG] = {0};

    double global_mse = 0.0;
    float max_abs_err = 0.0f;

    // Loop principal - ejecutando múltiples pruebas
    for (int iter = 0; iter < NUM_TEST_VECTORS; ++iter) {

        if(iter % 10 == 0 && test_mode == 1){
            std::cout << "Iteración: " << iter << " ---" << std::endl;
        }
        

        // Buffers de prueba 
        std::vector<float> input_floats(vector_length);
        std::vector<float> gold_results(vector_length);
        
        // Generación de los números aleatorios entre f_min y f_max
        for (uint64_t i = 0; i < vector_length; i++) input_floats[i] = dist(gen);

        // Golden references
        if (test_mode == 1) {
            get_softmax_golden(input_floats, gold_results);
        } else {
            for (uint64_t i = 0; i < vector_length; i++) 
                gold_results[i] = get_pointwise_golden(input_floats[i], f_min, f_max);
        }

        // Empaquetado para hardware
        for (uint64_t p = 0; p < n_packets; p++) {
            RawDataT packet = 0;
            for (int i = 0; i < kPackets; i++) {
                DataT val_fixed = (DataT)input_floats[p * kPackets + i];
                int start = i * kPaddedWidth;
                packet.range(start + kDataWidth - 1, start) = val_fixed.range();
            }
            in_hw[p] = packet;
        }

        // Ejecución del kernel con reload = false - usa los valores ya cargados
        pwl_no_uniforme(in_hw, out_hw, vector_length, g_map, g_lut, g_lut, g_lut, 
                        ACTUAL_SEG, X_MIN, X_MAX, false, test_mode);

        // Análisis de resultados
        float iter_sum_hw = 0.0f;
        for (uint64_t p = 0; p < n_packets; p++) {
            RawDataT packet = out_hw[p];
            for (int i = 0; i < kPackets; i++) {
                int start = i * kPaddedWidth;
                DataT hw_fixed_val;
                hw_fixed_val.V = packet.range(start + kDataWidth - 1, start);
                float hw_val = (float)hw_fixed_val;
                float gold_val = gold_results[p * kPackets + i];

                // Error absoluto
                float diff = std::abs(hw_val - gold_val);
                // Guarda el peor error
                if (diff > max_abs_err) max_abs_err = diff;
                global_mse += (double)(diff * diff);
                iter_sum_hw += hw_val;
            }
        }
        
        if(iter % 10 == 0 && test_mode == 1) {
            std::cout << ">> Suma HW Softmax: " << iter_sum_hw << " (Valor esperado ~1.0)" << std::endl;
        }
    }
    std::cout << "Iteraciones realizadas: " << NUM_TEST_VECTORS <<  std::endl;

    // Métricas finales
    double total_elements = (double)NUM_TEST_VECTORS * vector_length;
    // RMSE
    double final_rmse = std::sqrt(global_mse / total_elements);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << ">> RMSE:             " << final_rmse << std::endl;
    std::cout << ">> Max Abs Error:    " << max_abs_err << std::endl;
    std::cout << ">> Status:           " << (final_rmse < ERROR_ADMITIDO ? "[PASS]" : "[FAIL]") << std::endl;
    std::cout << "==================================================" << std::endl;

    delete[] in_hw; delete[] out_hw;
    return (final_rmse < ERROR_ADMITIDO) ? 0 : 1;
}