#include "OpenRegTestAllocator.h"

#include <atomic>

namespace c10::openreg {
namespace {

std::atomic<uint64_t> g_record_stream_call_count{0};

} // namespace

uint64_t testGetRecordStreamCallCount() {
  return g_record_stream_call_count.load(std::memory_order_relaxed);
}

void testResetRecordStreamCallCount() {
  g_record_stream_call_count.store(0, std::memory_order_relaxed);
}

void testBumpRecordStreamCallCount() {
  g_record_stream_call_count.fetch_add(1, std::memory_order_relaxed);
}

} // namespace c10::openreg
