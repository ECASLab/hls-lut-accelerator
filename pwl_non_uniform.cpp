// Author: Patrick Hugo Nepveu Nelson <patrick.nepveu.ecaslab@gmail.com>
// Year: 2026 
// ECASLab

#include "pwl_non_uniform.h"

// Reduction tree for summation
static AccT sum_reduction_tree(AccT elements[kPackets]) { // Accepts vector of size kPackets where kPackets = kBusWidth / kPaddedWidth;
    #pragma HLS INLINE
    AccT tree[kPackets];
    // Each element becomes an independent register to read all values in parallel
    #pragma HLS ARRAY_PARTITION variable=tree complete

    // Copy the elements array into tree within a single clock cycle to prepare the reduction tree
    init: for(int i=0; i<kPackets; i++) {
        #pragma HLS UNROLL
        tree[i] = elements[i];
    }

    // Binary reduction tree implementation - log2(kPackets) iterations
    levels: for (int curr_size = kPackets / 2; curr_size > 0; curr_size /= 2) {
        #pragma HLS UNROLL
        reduce: for (int i = 0; i < curr_size; i++) {
            #pragma HLS UNROLL
            tree[i] = tree[i*2] + tree[i*2+1];
        }
    }
    return tree[0];
}

// Reduction tree for maximum value calculation
static DataT max_reduction_tree(DataT elements[kPackets]) { // Accepts vector of size kPackets where kPackets = kBusWidth / kPaddedWidth;
    #pragma HLS INLINE
    DataT tree[kPackets];
    // Each element becomes an independent register to read all values in parallel
    #pragma HLS ARRAY_PARTITION variable=tree complete

    // Copy the elements array into tree within a single clock cycle to prepare the reduction tree  
    init: for(int i=0; i<kPackets; i++) {
        #pragma HLS UNROLL
        tree[i] = elements[i];
    }

    levels: for (int curr_size = kPackets / 2; curr_size > 0; curr_size /= 2) {
        #pragma HLS UNROLL
        reduce: for (int i = 0; i < curr_size; i++) {
            #pragma HLS UNROLL
            // Ternary operator to find the maximum value between two contiguous tree elements and store it in the current position
            tree[i] = (tree[i*2] > tree[i*2+1]) ? tree[i*2] : tree[i*2+1];
        }
    }
    return tree[0];
}

// Mathematical Engine
static AccT compute_pwl(DataT x,
                         const uint8_t segment_map[MAP_SIZE],
                         const RawStorageT x0_l[MAX_SEG],
                         const RawStorageT m_l[MAX_SEG],
                         const RawStorageT b_l[MAX_SEG],
                         RawStorageT X_MIN_val, RawStorageT X_MAX_val) {
    #pragma HLS INLINE

    // Convert raw bit-patterns to DataT type
    DataT x_min; x_min.V = X_MIN_val;
    DataT x_max; x_max.V = X_MAX_val;
    
    // Clamping: Restrict x to the allowed boundaries [x_min, x_max]
    DataT x_clamped = (x > x_max) ? x_max : (x < x_min) ? x_min : x;

    // Calculate offset relative to the minimum limit
    DataT offset = x_clamped - x_min;

    // Index Lookup (Bit Slicing)
    constexpr int shift_val = (KDATAWIDTH_FIXED - KFXPDATAINT) - 6;
    // Extract relevant bits to generate the map index
    uint16_t map_idx = (uint16_t)(offset.range(KDATAWIDTH_FIXED-1, (shift_val < 0 ? 0 : shift_val)));
    if (map_idx > (MAP_SIZE - 1)) map_idx = MAP_SIZE - 1;
    // Retrieve the corresponding segment ID
    uint8_t seg_id = segment_map[map_idx];

    // PWL Engine Core Calculation: y = m*(x-x0) + b
    DataT x0_v, m_v, b_v;
    // Convert coefficients from raw formats back to DataT
    x0_v.V = x0_l[seg_id];
    m_v.V  = m_l[seg_id];
    b_v.V  = b_l[seg_id];

    DataT diff = x_clamped - x0_v;
    AccT m_acc = (AccT)m_v;
    AccT b_acc = (AccT)b_v;
    AccT res = m_acc * (AccT)diff + b_acc;
    // Force DSP usage and internal pipeline behavior if needed
    // #pragma HLS BIND_OP variable=prod op=mul impl=dsp latency=2 
    
    // Final segment output execution context
    return res;
}

static void load_input(RawDataT *in, StreamT &inStream, uint64_t size) {
    #pragma HLS INLINE off
    // Number of 512-bit wide packet transfers
    const uint64_t n_packets = size / kPackets;
    for (uint64_t i = 0; i < n_packets; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS loop_tripcount max=kTotalMaxSize
        // Read from Global Memory and push into local hardware stream
        inStream << in[i];
    }
}

static void compute_engine(StreamT &inStream, StreamT &outStream, uint64_t size,
                           const uint8_t segment_map[kPackets][MAP_SIZE],
                           const RawStorageT x0_l[MAX_SEG],
                           const RawStorageT m_l[MAX_SEG],
                           const RawStorageT b_l[MAX_SEG],
                           RawStorageT X_MIN, RawStorageT X_MAX,
                           int mode) {
    #pragma HLS INLINE off
    const uint64_t n_packets = size / kPackets;
    // 512-bit wide local block buffers to store full packet transactions in 1 clock cycle
    // Two independent buffers are used to eliminate RAW (Read-After-Write) hazard dependencies
    static RawDataT internal_buffer_in[2048];
    static RawDataT internal_buffer_out[2048]; 
    #pragma HLS BIND_STORAGE variable=internal_buffer_in type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=internal_buffer_out type=ram_2p impl=uram
    
    // Pass 1
    DataT global_max;
    if (mode == 1) { // Softmax Mode
        global_max.V = (RawStorageT)(1U << (KDATAWIDTH_FIXED - 1)); // Minimum possible value fallback
        // Pass 1: Global Maximum Extraction
        p1_max: for (uint64_t p = 0; p < n_packets; p++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS loop_tripcount max=kTotalMaxSize
            RawDataT raw = inStream.read();
            internal_buffer_in[p] = raw; // Stash in buffer for Pass 2 (softmax context)
            DataT lanes[kPackets];
            #pragma HLS ARRAY_PARTITION variable=lanes complete
            for(int i=0; i<kPackets; i++) {
                #pragma HLS UNROLL
                lanes[i] = GET_NUMBER<DataT>(raw.range((i+1)*kPaddedWidth-1, i*kPaddedWidth));
            }
            DataT p_max = max_reduction_tree(lanes);
            if (p_max > global_max) global_max = p_max;
        }
    } else {
        // Pointwise Bypass: If mode == 0, max acts as 0, resolving expressions like (x - 0) = x
        global_max = 0; 
        p1_bypass: for (uint64_t p = 0; p < n_packets; p++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS loop_tripcount max=kTotalMaxSize
            internal_buffer_in[p] = inStream.read(); // Stash in buffer for Pass 2 (pointwise context)
        }
    }

    // Pass 2: Unified compute_pwl execution wrapper reusing pipelines for Softmax and Pointwise blocks
    AccT global_sum = 0; // Guard against intermediate overflow using AccT
    p2_unified: for (uint64_t p = 0; p < n_packets; p++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS loop_tripcount max=kTotalMaxSize
        RawDataT raw = internal_buffer_in[p];
        RawDataT raw_res;
        AccT lanes_res[kPackets];
        #pragma HLS ARRAY_PARTITION variable=lanes_res complete

        for(int i=0; i<kPackets; i++) {
            #pragma HLS UNROLL
            DataT v = GET_NUMBER<DataT>(raw.range((i+1)*kPaddedWidth-1, i*kPaddedWidth));
            // All kPackets compute_pwl arithmetic lines handle Softmax AND Pointwise actions - functional hardware reuse
            AccT v_res = compute_pwl(v - global_max, segment_map[i], x0_l, m_l, b_l, X_MIN, X_MAX);
            lanes_res[i] = v_res;
            raw_res.range((i+1)*kPaddedWidth-1, i*kPaddedWidth) = GET_RAW((DataT)v_res);
        }
        
        if (mode == 0) {
            // Pointwise Bypass routing: bypass local buffers and dump directly into output streams
            outStream << raw_res; 
        } else {
            // Softmax tracking routing: retain current results inside Pass 3 arrays and execute summation reduction
            internal_buffer_out[p] = raw_res;
            global_sum += sum_reduction_tree(lanes_res);
        }
    }

    // Pass 3: Normalization Phase - Isolated block explicitly allocated for Softmax conditions (mode == 1)
    if (mode == 1) {
        AccT inv_sum = (AccT)1.0 / global_sum;
        p3_norm: for (uint64_t p = 0; p < n_packets; p++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS loop_tripcount max=kTotalMaxSize
            RawDataT raw = internal_buffer_out[p];
            RawDataT raw_out;
            for(int i=0; i<kPackets; i++) {
                #pragma HLS UNROLL
                // Extract previously calculated exponential value
                DataT v_exp;
                v_exp.V = raw.range((i+1)*kPaddedWidth-1, i*kPaddedWidth);
                AccT v_norm_wide = (AccT)v_exp * inv_sum;
                #pragma HLS BIND_OP variable=v_norm_wide op=mul impl=dsp latency=3
                // Mitigation of truncation errors via Convergent Rounding schemes
                // const AccT rounding_bit = (AccT)1 << (12 - 1);
                DataT v_norm = (DataT)v_norm_wide;
                // v_norm = (DataT)(v_norm_wide + rounding_bit);
                raw_out.range((i+1)*kPaddedWidth-1, i*kPaddedWidth) = v_norm.range();
            }
            outStream << raw_out;
        }
    }
}

// Execution Wrapper scheduling 3 concurrent dataflow procedures - Canonical Dataflow Form
static void execution_wrapper(RawDataT *in, RawDataT *out, uint64_t size,
                              const uint8_t segment_map_l[kPackets][MAP_SIZE],
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
void pwl_non_uniform(RawDataT *in, RawDataT *out, uint64_t size,
                     const uint8_t segment_map[MAP_SIZE],
                     const RawStorageT x0_in[MAX_SEG],
                     const RawStorageT m_in[MAX_SEG],
                     const RawStorageT b_in[MAX_SEG],
                     int n_segments,
                     RawStorageT X_MIN, RawStorageT X_MAX,
                     bool reload, int mode) {

    #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0 depth=kTotalMaxSize
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=kTotalMaxSize
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

    // Memory Banking with Redundancy Arrays
    static uint8_t segment_map_local[kPackets][MAP_SIZE];
    static RawStorageT x0_local[MAX_SEG];
    static RawStorageT m_local[MAX_SEG];
    static RawStorageT b_local[MAX_SEG];

    // Memory Banking Array Partitioning Strategy Design
    #pragma HLS ARRAY_PARTITION variable=segment_map_local complete dim=1
    #pragma HLS ARRAY_PARTITION variable=segment_map_local cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=x0_local complete
    #pragma HLS ARRAY_PARTITION variable=m_local complete
    #pragma HLS ARRAY_PARTITION variable=b_local complete

    if (reload) {
        init_map: for (int i = 0; i < MAP_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            uint8_t val = segment_map[i];
            for(int j = 0; j < kPackets; j++){
                #pragma HLS UNROLL
                // Synchronous broadcast: update all lanes concurrently with value 'val'
                segment_map_local[j][i] = val;
            }
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