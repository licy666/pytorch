#pragma once

#include <include/Macros.h>

#include <cstdint>

namespace c10::openreg {

// Test-only hooks for validating stream-recording plumbing.
//
// These are intentionally minimal: they do not implement stream-aware lifetime
// semantics (that's handled by the allocator itself), they only help tests
// verify that the guard hook reaches the allocator.
OPENREG_EXPORT uint64_t testGetRecordStreamCallCount();
OPENREG_EXPORT void testResetRecordStreamCallCount();

// Internal helper invoked by the allocator's recordStream implementation.
void testBumpRecordStreamCallCount();

} // namespace c10::openreg
