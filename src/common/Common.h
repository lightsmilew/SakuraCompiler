// ---------------------------------------------------------------------------
// Common.h: shared plumbing used by every compiler stage.
//
//  * TypeCat + ConstVal  - the front-end view of scalar types and of
//    compile-time constants (int kept in signed-32-bit range, float as f32).
//  * f2bits/bits2f/f2i_trunc - IEEE-754 helpers mirroring the RISC-V runtime
//    (fcvt.w.s truncation semantics, NaN/overflow -> INT32_MIN).
//  * CompileError        - exception carrying a source line for diagnostics.
//  * Format              - tiny ostringstream-based string builder.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <iostream>

namespace sakura {

enum class TypeCat : uint8_t { VOID, INT, FLOAT };

inline const char *typeCatName(TypeCat t) {
  switch (t) {
  case TypeCat::VOID: return "void";
  case TypeCat::INT: return "int";
  case TypeCat::FLOAT: return "float";
  }
  return "?";
}

// A compile-time scalar constant.
struct ConstVal {
  bool valid = false;
  TypeCat type = TypeCat::VOID;
  int64_t ival = 0;   // INT value (always kept in signed 32-bit range)
  float   fval = 0.0f; // FLOAT value

  static ConstVal intC(int64_t v) {
    ConstVal c;
    c.valid = true;
    c.type = TypeCat::INT;
    c.ival = (int32_t)(v & 0xFFFFFFFFLL);  // wrap to 32-bit
    return c;
  }
  static ConstVal floatC(float v) {
    ConstVal c;
    c.valid = true;
    c.type = TypeCat::FLOAT;
    c.fval = v;
    return c;
  }
  uint32_t fbits() const {
    uint32_t u = 0;
    std::memcpy(&u, &fval, sizeof(u));
    return u;
  }
};

// Float bit helpers (IEEE single precision).
inline uint32_t f2bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
inline float bits2f(uint32_t u) {
  float f = 0;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// Force a float operation result back to single precision (in case the host
// promotes to double/extended).
inline float f32(float v) { return (float)v; }

// Saturating/truncating float -> int32 conversion, mirroring RISC-V fcvt.w.s
// with rtz rounding: truncation toward zero; out-of-range/NaN => INT32_MIN.
inline int32_t f2i_trunc(float f) {
  if (f != f) return INT32_MIN;             // NaN
  if (f >= 2147483648.0f) return INT32_MIN; // overflow positive
  if (f <= -2147483649.0f) return INT32_MIN;
  // truncation toward zero
  int32_t r = (int32_t)f;
  return r;
}

// Nearest-even float -> int (used by printf-like paths only if needed).
struct CompileError : public std::runtime_error {
  int line;
  explicit CompileError(const std::string &msg, int ln = -1)
      : std::runtime_error(msg), line(ln) {}
};

struct Format {
  template <typename... Args>
  static std::string str(Args &&...args) {
    std::ostringstream os;
    (os << ... << std::forward<Args>(args));
    return os.str();
  }
};

} // namespace sakura
