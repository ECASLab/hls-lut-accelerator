/*
Autor: Patrick Hugo Nepveu Nelson
Año: 2026 ECASLab
*/

#include "pwl_no_uniforme.h"


// Reduction tree para suma
static DataT sum_reduction_tree(DataT elements[kPackets]) { //Recibe vector de tamaño kPackets donde kPackets = kBusWidth / kPaddedWidth;
    #pragma HLS INLINE
    DataT tree[kPackets];
    // Cada elemento se convierte en un registro independiente para leer todos los valores en paralelo
    #pragma HLS ARRAY_PARTITION variable=tree complete

    // Copia el array elements en tree, todo en el mismo ciclo para preparar el árbol de reducción
    init: for(int i=0; i<kPackets; i++) {
        #pragma HLS UNROLL
        tree[i] = elements[i];

    }

    // Aquí se implementa el árbol binario de reducción - iteraciones log2(kPackets)
    levels: for (int curr_size = kPackets / 2; curr_size > 0; curr_size /= 2) {
        #pragma HLS UNROLL
        reduce: for (int i = 0; i < curr_size; i++) {
            #pragma HLS UNROLL
            tree[i] = tree[i*2] + tree[i*2+1];
        }
    }
    return tree[0];
}

// Reduction tree para max
static DataT max_reduction_tree(DataT elements[kPackets]) { //Recibe vector de tamaño kPackets donde kPackets = kBusWidth / kPaddedWidth;
    #pragma HLS INLINE
    DataT tree[kPackets];
    // Cada elemento se convierte en un registro independiente para leer todos los valores en paralelo
    #pragma HLS ARRAY_PARTITION variable=tree complete

    // Copia el array elements en tree, todo en el mismo ciclo para preparar el árbol de reducción  
    init: for(int i=0; i<kPackets; i++) {
        #pragma HLS UNROLL
        tree[i] = elements[i];

    }

    levels: for (int curr_size = kPackets / 2; curr_size > 0; curr_size /= 2) {
        #pragma HLS UNROLL
        reduce: for (int i = 0; i < curr_size; i++) {
            #pragma HLS UNROLL
            // operador ternario para encontrar el valor máximo entre dos elementos contiguos del árbol y guardarlo en la posición actual
            tree[i] = (tree[i*2] > tree[i*2+1]) ? tree[i*2] : tree[i*2+1];

        }
    }
    return tree[0];
}

// Motor matemático
static DataT compute_pwl(DataT x,
                         const uint8_t segment_map[MAP_SIZE],
                         const RawStorageT x0_l[MAX_SEG],
                         const RawStorageT m_l[MAX_SEG],
                         const RawStorageT b_l[MAX_SEG],
                         RawStorageT X_MIN_val, RawStorageT X_MAX_val) {
    #pragma HLS INLINE

    // Convertir valores crudos (bits) a tipo DataT
    DataT x_min; x_min.V = X_MIN_val;
    DataT x_max; x_max.V = X_MAX_val;
    
    // Clamping: limita x al rango permitido [x_min, x_max]
    DataT x_clamped = (x > x_max) ? x_max : (x < x_min) ? x_min : x;

    // Calcular offset respecto al mínimo
    DataT offset = x_clamped - x_min;

    // Index Lookup (Bit Slicing)
    constexpr int shift_val = (KDATAWIDTH_FIXED - KFXPDATAINT) - 6;
    // Extraer bits relevantes para generar índice
    uint16_t map_idx = (uint16_t)(offset.range(KDATAWIDTH_FIXED-1, (shift_val < 0 ? 0 : shift_val)));
    if (map_idx > (MAP_SIZE - 1)) map_idx = MAP_SIZE - 1;
    // Obtener ID del segmento correspondiente
    uint8_t seg_id = segment_map[map_idx];

    // Motor PWL: y = m*(x-x0) + b
    DataT x0_v, m_v, b_v;
    // Convertir coeficientes de formato raw a DataT
    x0_v.V = x0_l[seg_id];
    m_v.V  = m_l[seg_id];
    b_v.V  = b_l[seg_id];

    DataT diff = x_clamped - x0_v;
    DataT prod;
    // Fuerzar uso de DSP y pipeline interno
    #pragma HLS BIND_OP variable=prod op=mul impl=dsp latency=2 
    prod = m_v * diff;
    
    // Resultado final del segmento
    return prod + b_v;
}

static void load_input(RawDataT *in, StreamT &inStream, uint64_t size) {
    #pragma HLS INLINE off
     // Número de paquetes de 512 bits
    const uint64_t n_packets = size / kPackets;
    for (uint64_t i = 0; i < n_packets; i++) {
        #pragma HLS PIPELINE II=1
        // Lee desde memoria global (AXI) y lo empuja al stream
        inStream << in[i];
    }
}

static void compute_engine(StreamT &inStream, StreamT &outStream, uint64_t size,
                           const uint8_t segment_map[MAP_SIZE],
                           const RawStorageT x0_l[MAX_SEG],
                           const RawStorageT m_l[MAX_SEG],
                           const RawStorageT b_l[MAX_SEG],
                           RawStorageT X_MIN, RawStorageT X_MAX,
                           int mode) {
    #pragma HLS INLINE off
    const uint64_t n_packets = size / kPackets;

    // Buffer de 512 bits para guardar paquetes completos en 1 ciclo
    static RawDataT internal_buffer[2048]; 
    #pragma HLS BIND_STORAGE variable=internal_buffer type=ram_2p impl=uram

    if (mode == 1) { // Modo softmax
        // Pass 1: Máximo global
        DataT global_max;
        global_max.V = (RawStorageT)(1U << (KDATAWIDTH_FIXED - 1)); // Minimo posible

        p1_max: for (uint64_t p = 0; p < n_packets; p++) {
            #pragma HLS PIPELINE II=1
            RawDataT raw = inStream.read();
            internal_buffer[p] = raw; // Buffer para Pass 2
            
            DataT lanes[kPackets];
            #pragma HLS ARRAY_PARTITION variable=lanes complete
            for(int i=0; i<kPackets; i++) {
                #pragma HLS UNROLL
                lanes[i] = GET_NUMBER<DataT>(raw.range((i+1)*kPaddedWidth-1, i*kPaddedWidth));
            }
            DataT p_max = max_reduction_tree(lanes);
            if (p_max > global_max) global_max = p_max;
        }

        // Pass 2: Exponencial y suma
        DataT global_sum = 0;
        p2_exp_sum: for (uint64_t p = 0; p < n_packets; p++) {
            #pragma HLS PIPELINE II=1
            RawDataT raw = internal_buffer[p];
            RawDataT raw_exp;
            DataT lanes_exp[kPackets];
            #pragma HLS ARRAY_PARTITION variable=lanes_exp complete

            for(int i=0; i<kPackets; i++) {
                #pragma HLS UNROLL
                DataT v = GET_NUMBER<DataT>(raw.range((i+1)*kPaddedWidth-1, i*kPaddedWidth));
                DataT v_exp = compute_pwl(v - global_max, segment_map, x0_l, m_l, b_l, X_MIN, X_MAX);
                lanes_exp[i] = v_exp;
                raw_exp.range((i+1)*kPaddedWidth-1, i*kPaddedWidth) = GET_RAW(v_exp);
            }
            internal_buffer[p] = raw_exp; // Guardar resultados exp para Pass 3
            global_sum += sum_reduction_tree(lanes_exp);
        }

        // Pass 3: Normalización
        DataT inv_sum = (DataT)1.0 / global_sum;
        p3_norm: for (uint64_t p = 0; p < n_packets; p++) {
            #pragma HLS PIPELINE II=1
            RawDataT raw = internal_buffer[p];
            RawDataT raw_out;
            for(int i=0; i<kPackets; i++) {
                #pragma HLS UNROLL
                DataT v = GET_NUMBER<DataT>(raw.range((i+1)*kPaddedWidth-1, i*kPaddedWidth));
                DataT v_norm;
                #pragma HLS BIND_OP variable=v_norm op=mul impl=dsp latency=3
                v_norm = v * inv_sum;
                raw_out.range((i+1)*kPaddedWidth-1, i*kPaddedWidth) = GET_RAW(v_norm);
            }
            outStream << raw_out;
        }
    } else { // Modo pointwise
        p0_direct: for (uint64_t p = 0; p < n_packets; p++) {
            #pragma HLS PIPELINE II=1
            RawDataT raw_in = inStream.read();
            RawDataT raw_out;
            for (int i = 0; i < kPackets; i++) {
                #pragma HLS UNROLL
                DataT v = GET_NUMBER<DataT>(raw_in.range((i+1)*kPaddedWidth-1, i*kPaddedWidth));
                DataT v_out = compute_pwl(v, segment_map, x0_l, m_l, b_l, X_MIN, X_MAX);
                raw_out.range((i+1)*kPaddedWidth-1, i*kPaddedWidth) = GET_RAW(v_out);
            }
            outStream << raw_out;
        }
    }
}

static void store_result(RawDataT *out, StreamT &outStream, uint64_t size) {
    #pragma HLS INLINE off
    const uint64_t n_packets = size / kPackets;
    for (uint64_t i = 0; i < n_packets; i++) {
        #pragma HLS PIPELINE II=1
        // Escribe desde stream a memoria global
        out[i] = outStream.read();
    }
}

// Wrapper para determinar los 3 procesos concurrentes en hardware - Dataflow Canonical Form
static void execution_wrapper(RawDataT *in, RawDataT *out, uint64_t size,
                              const uint8_t segment_map_l[MAP_SIZE],
                              const RawStorageT x0_l[MAX_SEG],
                              const RawStorageT m_l[MAX_SEG],
                              const RawStorageT b_l[MAX_SEG],
                              RawStorageT X_MIN, RawStorageT X_MAX,
                              int mode) {
    #pragma HLS DATAFLOW
    StreamT inStream("in_stream");
    StreamT outStream("out_stream");
    #pragma HLS STREAM variable=inStream depth=128
    #pragma HLS STREAM variable=outStream depth=128

    load_input(in, inStream, size);
    compute_engine(inStream, outStream, size, segment_map_l, x0_l, m_l, b_l, X_MIN, X_MAX, mode);
    store_result(out, outStream, size);
}

extern "C" {
void pwl_no_uniforme(RawDataT *in, RawDataT *out, uint64_t size,
                     const uint8_t segment_map[MAP_SIZE],
                     const RawStorageT x0_in[MAX_SEG],
                     const RawStorageT m_in[MAX_SEG],
                     const RawStorageT b_in[MAX_SEG],
                     int n_segments,
                     RawStorageT X_MIN, RawStorageT X_MAX,
                     bool reload, int mode) {

    #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0 depth=1024
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=1024
    #pragma HLS INTERFACE s_axilite port=segment_map bundle=control
    #pragma HLS INTERFACE s_axilite port=x0_in       bundle=control
    #pragma HLS INTERFACE s_axilite port=m_in        bundle=control
    #pragma HLS INTERFACE s_axilite port=b_in        bundle=control
    #pragma HLS INTERFACE s_axilite port=n_segments  bundle=control
    #pragma HLS INTERFACE s_axilite port=X_MIN       bundle=control
    #pragma HLS INTERFACE s_axilite port=X_MAX       bundle=control
    #pragma HLS INTERFACE s_axilite port=size        bundle=control
    #pragma HLS INTERFACE s_axilite port=return      bundle=control
    #pragma HLS INTERFACE s_axilite port=in          bundle=control
    #pragma HLS INTERFACE s_axilite port=out         bundle=control
    #pragma HLS INTERFACE s_axilite port=reload      bundle=control
    #pragma HLS INTERFACE s_axilite port=mode        bundle=control

    static uint8_t segment_map_local[MAP_SIZE];
    static RawStorageT x0_local[MAX_SEG];
    static RawStorageT m_local[MAX_SEG];
    static RawStorageT b_local[MAX_SEG];

    /*
    #pragma HLS ARRAY_PARTITION variable=segment_map_local complete
    #pragma HLS ARRAY_PARTITION variable=x0_local complete
    #pragma HLS ARRAY_PARTITION variable=m_local complete
    #pragma HLS ARRAY_PARTITION variable=b_local complete
    */

    
    #pragma HLS ARRAY_PARTITION variable=segment_map_local cyclic factor=kPackets
    #pragma HLS BIND_STORAGE variable=segment_map_local type=ram_2p impl=lutram

    #pragma HLS ARRAY_PARTITION variable=x0_local cyclic factor=kPackets
    #pragma HLS ARRAY_PARTITION variable=m_local  cyclic factor=kPackets
    #pragma HLS ARRAY_PARTITION variable=b_local  cyclic factor=kPackets
    #pragma HLS BIND_STORAGE variable=x0_local type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=m_local  type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=b_local  type=ram_2p impl=lutram
    

    if (reload) {
        init_map: for (int i = 0; i < MAP_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            segment_map_local[i] = segment_map[i];
        }
        init_coefs: for (int i = 0; i < MAX_SEG; i++) {
            #pragma HLS PIPELINE II=1
            x0_local[i] = x0_in[i];
            m_local[i] = m_in[i];
            b_local[i] = b_in[i];
        }
    } else {
        execution_wrapper(in, out, size, segment_map_local, x0_local, m_local, b_local, X_MIN, X_MAX, mode);
    }
}
} // extern "C"