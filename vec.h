#ifndef VEC_H
#define VEC_H

#include <cstdint>

#if defined(__SSE2__)

#include <emmintrin.h>
typedef __m128i reg128;

#elif defined(__ARM_NEON__)

#include <arm_neon.h>
typedef int32x4_t reg128;

#endif // __ARM_NEON__

template <class type>
class vec {
public:
	vec(reg128 v) : reg(v) {}
	static vec<type> load(const void *addr);
	type operator[](int idx);
	vec<type> operator+(const vec<type> &r);
	vec<type> operator-(const vec<type> &r);
private:
	reg128 reg;
};

typedef vec<int8_t>   char16;
typedef vec<uint8_t>  uchar16;
typedef vec<int16_t>  short8;
typedef vec<uint16_t> ushort8;
typedef vec<int32_t>  int4;
typedef vec<uint32_t> uint4;
typedef vec<int64_t>  long4;
typedef vec<uint64_t> ulong4;
typedef vec<float>    float4;

#if defined(__SSE2__)

template <>
inline vec<int32_t> vec<int32_t>::load(const void *addr) {
	return int4(_mm_load_si128(static_cast<const __m128i *>(addr)));
}

template <>
inline int32_t vec<int32_t>::operator[](int idx) {
	int32_t val;
	val = _mm_extract_epi16(this->reg, idx << 1);
	val += _mm_extract_epi16(this->reg, (idx << 1) + 1) << 16;
	return val;
}

template <>
inline vec<int32_t> vec<int32_t>::operator+(const vec<int32_t> &r) {
	return vec<int32_t>(_mm_add_epi32(this->reg, r.reg));
}
template <>
inline vec<int32_t> vec<int32_t>::operator-(const vec<int32_t> &r) {
	return vec<int32_t>(_mm_sub_epi32(this->reg, r.reg));
}

#elif defined(__ARM_NEON__)

template <>
inline vec<int32_t> vec<int32_t>::load(const void *addr) {
	return vec<int32_t>(vld1q_u32(static_cast<const uint32_t *>(addr)));
}

template <>
inline int32_t vec<int32_t>::operator[](int idx) {
	int32_t val;
	switch (idx) {
		case 0: val = vgetq_lane_s32(this->reg, 0); break;
		case 1: val = vgetq_lane_s32(this->reg, 1); break;
		case 2: val = vgetq_lane_s32(this->reg, 2); break;
		case 3: val = vgetq_lane_s32(this->reg, 3); break;
	}
	return val;
}

template <>
inline vec<int32_t> vec<int32_t>::operator-(const vec<int32_t> &r) {
	return vec<int32_t>(vaddq_s32(this->reg, r.reg));
}
template <>
inline vec<int32_t> vec<int32_t>::operator-(const vec<int32_t> &r) {
	return vec<int32_t>(vsubq_s32(this->reg, r.reg));
}

#endif // __ARM_NEON__

#endif // VEC_H
