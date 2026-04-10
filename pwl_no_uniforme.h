/*
Autor: Patrick Hugo Nepveu Nelson <patrick.cr1405@gmail.com>
Año: 2026 ECASLab
*/

#ifndef __PWL_NO_UNIFORME__
#define __PWL_NO_UNIFORME__

#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <type_traits>

// Data width - número total de bits y bits de la parte entera (incluyendo signo)
#ifndef KDATAWIDTH_FIXED
#define KDATAWIDTH_FIXED 12
#endif
static constexpr int kDataWidth = KDATAWIDTH_FIXED;
#ifndef KFXPDATAINT
#define KFXPDATAINT 6
#endif 

// Tamaño del vector
#ifndef KCOLS
#define KCOLS 16
#endif
#ifndef KROWS
#define KROWS 16
#endif
static constexpr int kCols = KCOLS;
static constexpr int kRows = KROWS;

// Número máximo de segmentos y tamaño del mapa uniforme
#ifndef MAX_SEG
#define MAX_SEG 128
#endif
#ifndef MAP_SIZE
#define MAP_SIZE 512
#endif

// Almacenimiento del raw bit-pattern
#if(KDATAWIDTH_FIXED <= 16)
    typedef uint16_t RawStorageT;
#elif (KDATAWIDTH_FIXED <= 32)
    typedef uint32_t RawStorageT;
#else
    typedef uint64_t RawStorageT;
#endif

// Padding para asegurar tamaños alineados
static constexpr int kPaddedWidth = (KDATAWIDTH_FIXED <= 8)  ? 8 :
                                    (KDATAWIDTH_FIXED <= 16) ? 16 :
                                    (KDATAWIDTH_FIXED <= 32) ? 32 : 64;

// Bus Width y parámetros AXI
#ifndef KBUSWIDTH
#define KBUSWIDTH 512 // 512-bit wide AXI4
#endif
static constexpr int kBusWidth = KBUSWIDTH;

// Tipo numérico para computación
using DataT = ap_fixed<KDATAWIDTH_FIXED, KFXPDATAINT>;

// Cantidad de números que contiene una palabra de 512 bits 
static constexpr int kPackets = kBusWidth / kPaddedWidth;

// Cantidad de palabras de 512 bits para procesar el dataset completo
static constexpr uint64_t kTotalMaxSize = kCols * kRows / kPackets;

// Contenedor de 512 bits utilizado para I/O
using RawDataT = ap_uint<kBusWidth>;

// Type alias para hls::stream
using StreamT = hls::stream<RawDataT>;

/**
 * @brief Convierte bits crudos del bus AXI al tipo de dato DataT (ap_fixed).
 * @param raw Segmento de bits del bus.
 */
template <typename T>
inline T GET_NUMBER(const ap_uint<kDataWidth>& raw) {
    #pragma HLS INLINE
    T result;
    result.range() = raw;
    return result;
}


/**
 * @brief Empaqueta el valor DataT de vuelta a bits crudos para el bus AXI.
 * @param val Valor en punto fijo.
 */
inline ap_uint<kDataWidth> GET_RAW(const DataT& val) {
    #pragma HLS INLINE
    return val.range(kDataWidth - 1, 0);
}


extern "C" {

void pwl_no_uniforme(
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

#endif // __PWL_NO_UNIFORME_H__