//
//  shader.metal
//  MetalBuffer
//
//  Created by Shogo Nobuhara on 2021/02/16.
//

#include <metal_stdlib>
#include "ShaderTypes.h"

kernel void
reverseValues(constant int32_t *src [[buffer(kReverseIndexSrc)]],
              device int32_t *dst [[buffer(kReverseIndexDst)]],
              constant int32_t &count [[buffer(kReverseIndexCount)]],
              uint position [[thread_position_in_grid]])
{
    // 範囲外チェック
    if (position < (uint)count)
    {
        dst[position] = src[count - position -1];
    }
}


