#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/util/flat_hash_map.h>

#include <include/openreg.h>

#include <map>
#include <memory>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace c10::openreg {

class DeviceMemoryAllocator {
 public:
  explicit DeviceMemoryAllocator(c10::DeviceIndex device_index);

  DeviceMemoryAllocator(const DeviceMemoryAllocator&) = delete;
  DeviceMemoryAllocator& operator=(const DeviceMemoryAllocator&) = delete;

  void* malloc(size_t nbytes);

  void free(void* ptr);

  void recordStream(void* ptr, c10::Stream stream);

  // Releases all cached blocks back to OpenReg and returns the pointers that
  // were removed from the cache (successfully freed or already absent).
  std::vector<void*> emptyCache();

  c10::CachingDeviceAllocator::DeviceStats getStats();

  void resetAccumulatedStats();

  void resetPeakStats();

 private:
  enum class BlockState : uint8_t { Allocated, Cached, Deferred };

  struct BlockInfo {
    BlockInfo(size_t size_bytes, c10::Stream alloc_stream)
        : size_bytes(size_bytes),
          alloc_stream(std::move(alloc_stream)),
          state(BlockState::Allocated) {}

    size_t size_bytes{0};
    c10::Stream alloc_stream;
    std::unordered_set<c10::Stream> stream_uses;
    std::vector<orEvent_t> events;
    BlockState state{BlockState::Allocated};
  };

  c10::DeviceIndex device_index_;

  c10::CachingDeviceAllocator::DeviceStats stats_;

  std::unordered_map<void*, BlockInfo> blocks_;

  // Stream-scoped cache: best-fit by block size, partitioned by the stream the
  // block was last allocated on. This avoids unsafe cross-stream reuse without
  // an explicit dependency (recordStream/events).
  using StreamKey = c10::Stream;
  std::unordered_map<StreamKey, std::multimap<size_t, void*>> cached_blocks_by_stream_;

  std::unordered_set<void*> deferred_pointers_;

  std::recursive_mutex mutex_;

  void processDeferredBlocksNoLock();
  void moveBlockToCacheNoLock(void* ptr, BlockInfo& block);
};


class OpenRegDeviceAllocator final : public c10::DeviceAllocator {
 public:
  OpenRegDeviceAllocator();

  at::DataPtr allocate(size_t nbytes) override;
  at::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const final;

  bool initialized() override;
  void emptyCache(MempoolId_t mempool_id = {0, 0}) override;
  void recordStream(const DataPtr& ptr, c10::Stream stream) override;
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;

  void freeMemory(void* ptr);

 private:
  // Per-device allocators
  std::vector<std::unique_ptr<DeviceMemoryAllocator>> device_allocators_;

  // Global mapping from pointer to device index
  std::recursive_mutex mutex_;
  ska::flat_hash_map<void*, c10::DeviceIndex> allocated_blocks_;
};

} // namespace c10::openreg
