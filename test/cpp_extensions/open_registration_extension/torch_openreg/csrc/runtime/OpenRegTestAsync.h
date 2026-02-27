#pragma once

#include <c10/core/Stream.h>

#include <include/Macros.h>

#include <cstdint>

namespace c10::openreg {

// Test-only helpers for deterministically enqueuing "in-flight" work on an
// OpenReg stream. These APIs are intentionally minimal and are only used by
// OpenReg's Python test suite.
OPENREG_EXPORT int64_t testCreateBlockingGate();
OPENREG_EXPORT void testReleaseBlockingGate(int64_t gate_id);
OPENREG_EXPORT void testEnqueueWaitForGate(const c10::Stream& stream, int64_t gate_id);

} // namespace c10::openreg
