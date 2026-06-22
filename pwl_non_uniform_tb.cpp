// Author: Patrick Hugo Nepveu Nelson <patrick.nepveu.ecaslab@gmail.com>
// Year: 2026 
// ECASLab

#include "pwl_non_uniform.h"
#include "lut_coeffs.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>

// Number of simulation test vectors
#ifndef NUM_TEST_VECTORS
#define NUM_TEST_VECTORS 50
#endif

// Maximum permitted target error tolerance
#ifndef ALLOWED_ERROR
#define ALLOWED_ERROR 0.005 
#endif

// Interpret raw bit-patterns as a floating point representation
float raw_to_float(uint32_t raw_val) {
    DataT temp;
    temp.V = raw_val;
    return (float)temp;
}

// Golden Reference math evaluations for point-wise non-linear functions
float get_pointwise_golden(float x, float f_min, float f_max) {
    // Clamping to isolate x within the [f_min, f_max] boundary
    // condition ? value_if_true : value_if_false;
    float x_clamped = (x > f_max) ? f_max : (x < f_min) ? f_min : x;
    
    if (SIG_FUNC)   return 1.0f / (1.0f + std::exp(-x_clamped));
    if (TANH_FUNC)  return std::tanh(x_clamped);
    if (GELU_FUNC)  return 0.5f * x_clamped * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x_clamped + 0.044715f * std::pow(x_clamped, 3))));
    if (RELU_FUNC)  return (x_clamped > 0.0f) ? x_clamped : 0.0f;
    if (SWISH_FUNC) return x_clamped / (1.0f + std::exp(-x_clamped));
    if (EXP_FUNC)   return std::exp(x_clamped); // exp is evaluated directly only if mode = 0
    return 0.0f;
}

// Golden Reference tracking for Softmax functionality
void get_softmax_golden(const std::vector<float>& input, std::vector<float>& output) {
    // Locate peak value - max_element returns an iterator; dereferencing yields the actual value
    float max_val = *std::max_element(input.begin(), input.end());
    double sum = 0.0;
    std::vector<double> exps(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        // Safe Softmax transformation - subtracting max_val stabilizes ranges between 0 and 1 
        exps[i] = std::exp((double)input[i] - (double)max_val);
        sum += exps[i];
    }
    for (size_t i = 0; i < input.size(); ++i) {
        // Final normalization normalization scale assignment
        output[i] = (float)(exps[i] / sum);
    }
}


int main() {
    // Convert configuration limits to floats
    float f_min = raw_to_float(X_MIN);
    float f_max = raw_to_float(X_MAX);
    // Combined total length of data elements within vectors
    const uint64_t vector_length = KROWS * KCOLS; 
    // Total required 512-bit transactional packets
    const uint64_t n_packets = vector_length / kPackets;

    // Set test_mode to 1 when evaluating exponential layers for Softmax routing
    int test_mode = 1;

    std::cout << " Function: ";
    if (SIG_FUNC)  std::cout << "Sigmoid";
    else if (TANH_FUNC) std::cout << "Tanh";
    else if (GELU_FUNC) std::cout << "GELU";
    else if (RELU_FUNC) std::cout << "ReLU";
    else if (SWISH_FUNC) std::cout << "Swish";
    else if (EXP_FUNC)   std::cout << "Exponential";
    else std::cout << "Unknown";
    std::cout << std::endl;


    std::cout << "==================================================" << std::endl;
    std::cout << " PWL TESTBENCH" << std::endl;
    std::cout << " Data Width: " << KDATAWIDTH_FIXED << " with " << KFXPDATAINT << " integer bits" << std::endl;
    std::cout << " Range Detected:  [" << f_min << ", " << f_max << "]" << std::endl;
    std::cout << " Mode:   " << (test_mode ? "Softmax" : "Pointwise") << std::endl;
    std::cout << " Vector Dimensions:  " << vector_length << " elements" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Simulation mapping memory space - allocation targets heap memory rather than bounded stacks
    RawDataT* in_hw  = new RawDataT[n_packets];
    RawDataT* out_hw = new RawDataT[n_packets];

    // Seed logic generating random numbers strictly bounded within [f_min, f_max]
    std::mt19937 gen(42); // Consistent static seed for deterministic reproducibility - Mersenne Twister
    std::uniform_real_distribution<float> dist(f_min, f_max);

    // Initial coefficients loading stage with reload = true. Processes coefficients configuration only.
    pwl_non_uniform(in_hw, out_hw, vector_length, segment_map, lut_x0_raw, 
                    lut_m_raw, lut_b_raw, ACTUAL_SEG, X_MIN, X_MAX, true, test_mode);


    // Verification block mimicking actual reload triggers - initializes zeroed dummy arrays
    uint8_t g_map[MAP_SIZE] = {0};
    RawStorageT g_lut[MAX_SEG] = {0};

    double global_mse = 0.0;
    float max_abs_err = 0.0f;

    // Core Loop - execution of verification batches
    for (int iter = 0; iter < NUM_TEST_VECTORS; ++iter) {

        if(iter % 10 == 0 && test_mode == 1){
            std::cout << "Iteration: " << iter << " ---" << std::endl;
        }
        
        // Simulation processing arrays 
        std::vector<float> input_floats(vector_length);
        std::vector<float> gold_results(vector_length);
        
        // Populate random numbers falling inside [f_min, f_max] limits
        for (uint64_t i = 0; i < vector_length; i++) input_floats[i] = dist(gen);

        // Golden References evaluation paths
        if (test_mode == 1) {
            get_softmax_golden(input_floats, gold_results);
        } else {
            for (uint64_t i = 0; i < vector_length; i++) 
                gold_results[i] = get_pointwise_golden(input_floats[i], f_min, f_max);
        }

        // Bit packing format structure compilation targeting IP pipeline expectations
        for (uint64_t p = 0; p < n_packets; p++) {
            RawDataT packet = 0;
            for (int i = 0; i < kPackets; i++) {
                DataT val_fixed = (DataT)input_floats[p * kPackets + i];
                int start = i * kPaddedWidth;
                packet.range(start + kDataWidth - 1, start) = val_fixed.range();
            }
            in_hw[p] = packet;
        }

        // Execution of the kernel core loop with reload = false - leverages existing local memory parameters
        pwl_non_uniform(in_hw, out_hw, vector_length, g_map, g_lut, g_lut, g_lut, 
                        ACTUAL_SEG, X_MIN, X_MAX, false, test_mode);

        // Metrics assessment phase
        float iter_sum_hw = 0.0f;
        for (uint64_t p = 0; p < n_packets; p++) {
            RawDataT packet = out_hw[p];
            for (int i = 0; i < kPackets; i++) {
                int start = i * kPaddedWidth;
                DataT hw_fixed_val;
                hw_fixed_val.V = packet.range(start + kDataWidth - 1, start);
                float hw_val = (float)hw_fixed_val;
                float gold_val = gold_results[p * kPackets + i];

                // Absolute Error quantification
                float diff = std::abs(hw_val - gold_val);
                // Log and trace worst-case discrepancy peaks
                if (diff > max_abs_err) max_abs_err = diff;
                global_mse += (double)(diff * diff);
                iter_sum_hw += hw_val;
            }
        }

        if (iter == 0) {
            std::cout << "\n>> Sample Preview of the first 15 entries (Iteration 0):" << std::endl;
            std::cout << std::setw(12) << "Input" << " | "
                      << std::setw(12) << "HW Out" << " | "
                      << std::setw(12) << "Golden" << " | "
                      << std::setw(12) << "Diff" << std::endl;
            std::cout << std::string(55, '-') << std::endl;

            for (uint64_t i = 0; i < 15 && i < vector_length; i++) {
                // Read processed hardware data from the out_hw buffer (unpacking packet segments)
                uint64_t p = i / kPackets;
                int offset = i % kPackets;
                int start = offset * kPaddedWidth;
                
                DataT hw_fixed_val;
                hw_fixed_val.V = out_hw[p].range(start + kDataWidth - 1, start);
                
                float hw_val = (float)hw_fixed_val;
                float gold_val = gold_results[i];
                float diff = std::abs(hw_val - gold_val);

                std::cout << std::fixed << std::setprecision(6)
                          << std::setw(12) << input_floats[i] << " | "
                          << std::setw(12) << hw_val << " | "
                          << std::setw(12) << gold_val << " | "
                          << std::setw(12) << diff << std::endl;
            }
            std::cout << "--------------------------------------------------\n" << std::endl;
        }
        
        if(iter % 10 == 0 && test_mode == 1) {
            std::cout << ">> HW Softmax Summation Output: " << iter_sum_hw << " (Expected Reference Value ~1.0)" << std::endl;
        }
    }
    std::cout << "Completed Iterations Count: " << NUM_TEST_VECTORS <<  std::endl;

    // Summary Analytics Reporting
    double total_elements = (double)NUM_TEST_VECTORS * vector_length;
    // RMSE
    double final_rmse = std::sqrt(global_mse / total_elements);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << ">> RMSE:             " << final_rmse << std::endl;
    std::cout << ">> Max Abs Error:    " << max_abs_err << std::endl;
    std::cout << ">> Verification:     " << (final_rmse < ALLOWED_ERROR ? "[PASS]" : "[FAIL]") << std::endl;
    std::cout << "==================================================" << std::endl;

    delete[] in_hw; delete[] out_hw;
    return 0;
}