// Author: Patrick Hugo Nepveu Nelson <patrick.nepveu.ecaslab@gmail.com>
// Year: 2026 
// ECASLab


#ifndef __PWL_NON_UNIFORM__
#define __PWL_NON_UNIFORM__

#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <type_traits>

// Data width - total number of bits and integer bits (including sign)
#ifndef KDATAWIDTH_FIXED
#define KDATAWIDTH_FIXED 32
#endif
static constexpr int kDataWidth = KDATAWIDTH_FIXED;
#ifndef KFXPDATAINT
#define KFXPDATAINT 8
#endif 

// Array / Vector dimensions
#ifndef KCOLS
#define KCOLS 16
#endif
#ifndef KROWS
#define KROWS 64
#endif
static constexpr int kCols = KCOLS;
static constexpr int kRows = KROWS;

// Maximum number of segments and uniform mapping table size
#ifndef MAX_SEG
#define MAX_SEG 128
#endif
#ifndef MAP_SIZE
#define MAP_SIZE 512
#endif

// Raw bit-pattern storage type assignment
#if(KDATAWIDTH_FIXED <= 16)
    typedef uint16_t RawStorageT;
#elif (KDATAWIDTH_FIXED <= 32)
    typedef uint32_t RawStorageT;
#else
    typedef uint64_t RawStorageT;
#endif

typedef ap_fixed<KDATAWIDTH_FIXED + 16, KFXPDATAINT + 12> AccT;

// Padding to guarantee aligned sizes
static constexpr int kPaddedWidth = (KDATAWIDTH_FIXED <= 8)  ? 8 :
                                    (KDATAWIDTH_FIXED <= 16) ? 16 :
                                    (KDATAWIDTH_FIXED <= 32) ? 32 : 64;

// Bus Width and AXI parameters
#ifndef KBUSWIDTH
#define KBUSWIDTH 512 // 512-bit wide AXI4
#endif
static constexpr int kBusWidth = KBUSWIDTH;

// Numeric type used for computation
using DataT = ap_fixed<KDATAWIDTH_FIXED, KFXPDATAINT>;

// Number of packed data values contained within a single 512-bit word
static constexpr int kPackets = kBusWidth / kPaddedWidth;

// Total number of 512-bit words required to process the complete dataset
static constexpr uint64_t kTotalMaxSize = kCols * kRows / kPackets;

// 512-bit wide container utilized for I/O operations
using RawDataT = ap_uint<kBusWidth>;

// Type alias for hls::stream
using StreamT = hls::stream<RawDataT>;

/**
 * @brief Converts raw bits from the AXI bus into the target DataT (ap_fixed) data type.
 * @param raw Bit segment slice from the bus.
 */
template <typename T>
inline T GET_NUMBER(const ap_uint<kDataWidth>& raw) {
    #pragma HLS INLINE
    T result;
    result.range() = raw;
    return result;
}


/**
 * @brief Packs a DataT fixed-point value back into raw bits for the AXI bus.
 * @param val Fixed-point value to capture.
 */
template <typename T>
inline ap_uint<kDataWidth> GET_RAW(const T& val) {
    #pragma HLS INLINE
    return val.range(kDataWidth - 1, 0);
}


extern "C" {

void pwl_non_uniform(
    RawDataT *in, 
    RawDataT *out, 
    uint64_t size,
    const uint8_t segment_map[MAP_SIZE], 
    const RawStorageT x0_in[MAX_SEG],    
    const RawStorageT m_in[MAX_SEG],     
    const RawStorageT b_in[MAX_SEG],     
    int n_segments,
    RawStorageT X_MIN_val, 
    RawStorageT X_MAX_val,
    bool reload,
    int mode
);
}

#endif // __PWL_NON_UNIFORM__