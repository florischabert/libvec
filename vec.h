#ifndef VEC_H
#define VEC_H

#include <cstdint>

#if defined(__SSE2__)

#include <emmintrin.h>
typedef __m128i vec128;

#elif defined(__ARM_NEON__)

#include <arm_neon.h>
typedef int32x4_t vec128;

#endif // __ARM_NEON__

class vec {
public:
	vec(vec128 v) : vec_(v) {}
protected:
	vec128 vec_;
};

struct char16  : public vec {};
struct uchar16 : public vec {};
struct short8  : public vec {};
struct ushort8 : public vec {};
struct int4    : public vec {
	int4(vec128 v) : vec(v) {}
	inline static int4 load(const void *addr);
	inline int32_t operator[](int idx);
	inline friend int4 operator+(int4 l, const int4 &r);
	inline friend int4 operator-(int4 l, const int4 &r);
};
struct uint4   : public vec {};
struct long2   : public vec {};
struct ulong2  : public vec {};
struct float4  : public vec {};

#if defined(__SSE2__)

inline int4 int4::load(const void *addr) {
	return int4(_mm_load_si128(static_cast<const __m128i *>(addr)));
}

inline int32_t int4::operator[](int idx) {
	int32_t val;
	val = _mm_extract_epi16(this->vec_, idx << 1);
	val += _mm_extract_epi16(this->vec_, (idx << 1) + 1) << 16;
	return val;
}

inline int4 operator+(int4 l, const int4 &r) {
	return int4(_mm_add_epi32(l.vec_, r.vec_));
}
inline int4 operator-(int4 l, const int4 &r) {
	return int4(_mm_sub_epi32(l.vec_, r.vec_));
}

#elif defined(__ARM_NEON__)

inline int4 int4::load(const void *addr) {
	return int4(vld1q_u32(static_cast<const uint32_t *>(addr)));
}

inline int32_t int4::operator[](int idx) {
	int32_t val;
	switch (idx) {
		case 0: val = vgetq_lane_s32(this->vec_, 0); break;
		case 1: val = vgetq_lane_s32(this->vec_, 1); break;
		case 2: val = vgetq_lane_s32(this->vec_, 2); break;
		case 3: val = vgetq_lane_s32(this->vec_, 3); break;
	}
	return val;
}

inline int4 operator+(int4 l, const int4 &r) {
	return int4(vaddq_s32(l.vec_, r.vec_));
}
inline int4 operator-(int4 l, const int4 &r) {
	return int4(vsubq_s32(l.vec_, r.vec_));
}

#endif // __ARM_NEON__

#endif // VEC_H
