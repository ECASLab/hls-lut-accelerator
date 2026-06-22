"""
Author: Patrick Hugo Nepveu Nelson <patrick.nepveu.ecaslab@gmail.com>
Year: 2026 
ECASLab
"""

import numpy as np
import matplotlib.pyplot as plt

# Non-linear functions
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def relu(x):
    return np.maximum(0, x)

def exponential(x):
    return np.exp(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanhiper(x):
    return np.tanh(x)

def swish(x, beta=1):
    return x / (1 + np.exp(-beta * x))


FUNCTIONS = {
    "gelu": gelu, 
    "relu": relu,
    "exp": exponential,
    "sig": sigmoid,
    "tanh": tanhiper,
    "swish": swish
}

function_name = "exp"
active_function = FUNCTIONS['exp']


def non_uniform_pwl(func, x_start, x_end, max_error):
    lut_m = []          # List for slopes
    lut_b = []          # List for intercepts
    breakpoints = [x_start]
    
    x_current = x_start
    print(f"Approximated non-linear function: {func.__name__}")
    print(f"{'Segment':<8} | {'Start':<8} | {'End':<8} | {'Width':<8} | {'Error':<10}")
    print("-" * 55)

    while x_current < x_end: # Loop covers the entire range
        # Start with a width of 2.0 - start wide to use the least amount of memory (greedy)
        width = 2.0 
        found = False
        
        while not found:
            x0 = x_current
            x1 = x_current + width
            
            # m = (y1 - y0) / width
            m = (func(x1) - func(x0)) / width
            b = func(x0) # local intercept
            
            # Verify the error across the segment using test points
            x_test = np.linspace(x0, x1, 20)
            y_real = func(x_test)
            y_approx = m * (x_test - x0) + b   # point-slope form
            calculated_error = np.max(np.abs(y_real - y_approx))
            
            # Decision criteria
            # If the error is acceptable, or the width reaches the minimum allowed step
            if calculated_error <= max_error or width <= 0.015625:
                lut_m.append(m)
                lut_b.append(b)
                
                print(f"{len(lut_m):<8} | {x0:<8.3f} | {x1:<8.3f} | {width:<8.4f} | {calculated_error:<10.6f}")
                
                x_current = x1
                breakpoints.append(x_current)
                found = True
            else:
                # If the error is too high, halve the width (maintains power of 2)
                width /= 2.0
            
    return lut_m, lut_b, breakpoints


def lut_hls_coeffs_header(breakpoints, m_list, b_list, bit_width=12, int_width=6, filename="lut_coeffs.h"):

    frac_bits = bit_width - int_width
    
    # Minimum step, equivalent to 2^-6
    STEP_MIN = 0.015625  
    # Amount of uniform slots that fit in the range 
    NUM_SLOTS = int((RANGE_MAX - RANGE_MIN) / STEP_MIN)

    # Uniform LUT mapping
    segment_map = []
    for i in range(NUM_SLOTS):
        x_val = RANGE_MIN + (i * STEP_MIN) + (STEP_MIN / 2)
        found = False
        for seg_idx in range(len(breakpoints) - 1):
            if x_val >= breakpoints[seg_idx] and x_val < breakpoints[seg_idx+1]:
                segment_map.append(seg_idx)
                found = True
                break
        if not found: 
            segment_map.append(len(m_list) - 1)

    def to_fixed_hex(val):
        # Multiply by 2^frac_bits to eliminate fractional part
        scaled = int(round(val * (2**frac_bits)))
        # Two's Complement based on bit_width
        if scaled < 0:
            scaled = (1 << bit_width) + scaled
        
        # Round up division (A+B-1)//B
        hex_len = (bit_width + 3) // 4
        # Python HEX format - 0:{hex_len} handles padding with 0s and X converts to uppercase Hex
        return f"0x{scaled & ((1 << bit_width) - 1):0{hex_len}X}"
    
    x_min_raw = to_fixed_hex(RANGE_MIN)
    x_max_raw = to_fixed_hex(RANGE_MAX)

    with open(filename, "w") as f:
        f.write(f"// Author: Patrick Hugo Nepveu Nelson <patrick.nepveu.ecaslab@gmail.com>\n")
        f.write(f"// Year: 2026 ECASLab\n")
        f.write(f"// Generated for <{bit_width},{int_width}> \n \n")
        f.write("#ifndef LUT_COEFFS_H\n#define LUT_COEFFS_H\n\n")
        
        f.write(f"#define EXP_FUNC  {1 if function_name == 'exp' else 0}\n")
        f.write(f"#define SIG_FUNC  {1 if function_name == 'sig' else 0}\n")
        f.write(f"#define TANH_FUNC {1 if function_name == 'tanh' else 0}\n")
        f.write(f"#define GELU_FUNC {1 if function_name == 'gelu' else 0}\n")
        f.write(f"#define RELU_FUNC {1 if function_name == 'relu' else 0}\n\n")
        f.write(f"#define SWISH_FUNC {1 if function_name == 'swish' else 0}\n\n")

        f.write(f"#define X_MIN {x_min_raw}\n") 
        f.write(f"#define X_MAX {x_max_raw}\n")
        f.write(f"#define ACTUAL_SEG {len(m_list)}\n")
        f.write(f"#define MAP_SIZE {NUM_SLOTS}\n\n")
        f.write(f"#define MAX_SEG 128 \n\n")

        # With MAX_SEG at 128, uint8_t can safely hold up to 255 elements
        f.write("const uint8_t segment_map[MAP_SIZE] = {\n    ") 
        f.write(", ".join([str(x) for x in segment_map]) + "\n};\n\n")
        storage_type = "uint16_t" if bit_width <= 16 else "uint32_t" if bit_width <= 32 else "uint64_t"
        
        f.write(f"const {storage_type} lut_x0_raw[MAX_SEG] = {{\n    ")
        f.write(", ".join([to_fixed_hex(x) for x in breakpoints[:-1]]) + "};\n\n")  
        f.write(f"const {storage_type} lut_m_raw[MAX_SEG] = {{\n    ")
        f.write(", ".join([to_fixed_hex(m) for m in m_list]) + "};\n\n")    
        f.write(f"const {storage_type} lut_b_raw[MAX_SEG] = {{\n    ")
        f.write(", ".join([to_fixed_hex(b) for b in b_list]) + "};\n\n")   
        f.write("#endif\n")


# Parameters
ALLOWED_ERROR = 0.005 # Target precision 
# Target Range
RANGE_MIN = -4.0        
RANGE_MAX = 4.0    

# Algorithm execution
slopes, intercepts, breakpoints = non_uniform_pwl(active_function, RANGE_MIN, RANGE_MAX, ALLOWED_ERROR)

# Generate C header (.h) for the interpolation coefficients
lut_hls_coeffs_header(breakpoints, slopes, intercepts)

print(f"\n=== MEMORY TABLE (Look-Up Table)")
print(f"{'Segment':<10} | {'Breakpoint (x0)':<18} | {'Slope (m)':<18} | {'Intercept (b)':<18}")
print("-" * 70)

for i in range(len(slopes)):
    # breakpoints[i] is x0, slopes[i] is m, intercepts[i] is b
    print(f"Seg {i:<6} | {breakpoints[i]:<18.6f} | {slopes[i]:<18.6f} | {intercepts[i]:<18.6f}")


y_breakpoints = [active_function(x) for x in breakpoints]

plt.figure(figsize=(10, 6))

plt.plot(breakpoints, y_breakpoints, 
         color='blue', 
         linestyle='-', 
         linewidth=2, 
         marker='o',             
         markersize=6, 
         markerfacecolor='red', 
         label='PWL Approximation (Straight Segments)')

# Plot the real underlying curve in light gray for comparison
x_real = np.linspace(RANGE_MIN, RANGE_MAX, 500)
plt.plot(x_real, active_function(x_real), color='black', alpha=0.2, linewidth=4, 
         label=f'Real {function_name.upper()} Function', zorder=0)

plt.title(f"Non-Uniform PWL Approximation: {function_name.upper()}")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Display total number of segments on the plot
plt.text(-3.8, max(y_breakpoints) * 0.9, f"Total Segments: {len(slopes)}", 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.show()

print("\n" + "="*30)
print(f"FINAL RESULT:")
print(f"Total segments generated: {len(slopes)}")
print("="*30)