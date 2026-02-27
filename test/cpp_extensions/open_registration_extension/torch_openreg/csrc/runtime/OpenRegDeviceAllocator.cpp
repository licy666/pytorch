#include "OpenRegDeviceAllocator.h"
#include "OpenRegFunctions.h"
#include "OpenRegStream.h"
#include "OpenRegTestAllocator.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/irange.h>
#include <c10/util/safe_numerics.h>

#include <unistd.h>

using namespace c10::CachingAllocator;

namespace c10::openreg {

constexpr size_t kAggregate = static_cast<size_t>(StatType::AGGREGATE);

namespace {

constexpr size_t kDefaultPageSizeBytes = 4096;

size_t getPageSizeBytes() {
  static const long page_size_long = sysconf(_SC_PAGESIZE);
  static const size_t page_size_bytes =
      page_size_long > 0 ? static_cast<size_t>(page_size_long)
                         : kDefaultPageSizeBytes;
  return page_size_bytes;
}

size_t roundUpToPageSize(size_t nbytes) {
  const size_t page_size = getPageSizeBytes();
  // Overflow-safe version of: ((nbytes + page_size - 1) / page_size) * page_size
  // Fail closed: if rounding overflows, refuse the allocation.
  TORCH_INTERNAL_ASSERT(page_size != 0);
  size_t rounded_up_dividend = 0;
  TORCH_CHECK(
      !c10::add_overflows(nbytes, page_size - 1, &rounded_up_dividend),
      "OpenReg allocator size overflow while aligning ",
      nbytes,
      " bytes to page size ",
      page_size);
  const size_t num_pages = rounded_up_dividend / page_size;
  size_t aligned_nbytes = 0;
  TORCH_CHECK(
      !c10::mul_overflows(num_pages, page_size, &aligned_nbytes),
      "OpenReg allocator size overflow while aligning ",
      nbytes,
      " bytes to page size ",
      page_size);
  return aligned_nbytes;
}

} // namespace


DeviceMemoryAllocator::DeviceMemoryAllocator(c10::DeviceIndex device_index)
    : device_index_(device_index) {}

void* DeviceMemoryAllocator::malloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  const auto alloc_stream = getCurrentOpenRegStream(device_index_).unwrap();
  const StreamKey stream_key = alloc_stream;

  // OpenReg aligns device allocations to page size internally.
  const size_t aligned_nbytes = roundUpToPageSize(nbytes);

  auto tryReuseFromCacheNoLock = [&](StreamKey key) -> void* {
    auto cache_it = cached_blocks_by_stream_.find(key);
    if (cache_it == cached_blocks_by_stream_.end()) {
      return nullptr;
    }

    auto& cache = cache_it->second;
    auto cached_it = cache.lower_bound(aligned_nbytes);
    if (cached_it == cache.end()) {
      return nullptr;
    }

    const size_t cached_nbytes = cached_it->first;
    void* data = cached_it->second;
    cache.erase(cached_it);
    if (cache.empty()) {
      cached_blocks_by_stream_.erase(cache_it);
    }

    auto it = blocks_.find(data);
    TORCH_INTERNAL_ASSERT(
        it != blocks_.end(),
        "OpenReg allocator missing metadata for cached pointer ",
        data);
    auto& block = it->second;
    block.size_bytes = cached_nbytes;
    block.alloc_stream = alloc_stream;
    block.stream_uses.clear();
    block.events.clear();
    block.state = BlockState::Allocated;

    stats_.allocated_bytes[kAggregate].increase(cached_nbytes);
    return data;
  };

  // Fast path: reuse from the current stream's cache.
  if (void* data = tryReuseFromCacheNoLock(stream_key)) {
    return data;
  }

  // Avoid scanning deferred blocks on every allocation. Only process deferred
  // blocks when we miss the cache (and only if there is work to do), then
  // retry the current stream's cache.
  if (!deferred_pointers_.empty()) {
    processDeferredBlocksNoLock();
    if (void* data = tryReuseFromCacheNoLock(stream_key)) {
      return data;
    }
  }

  void* data = nullptr;
  auto ret = orMalloc(&data, aligned_nbytes);

  TORCH_CHECK(
      ret == orSuccess && data != nullptr,
      "Failed to allocate ",
      aligned_nbytes,
      " bytes on openreg device ",
      device_index_,
      ". ",
      "Allocated: ",
      stats_.allocated_bytes[kAggregate].current,
      " bytes, ",
      "Reserved: ",
      stats_.reserved_bytes[kAggregate].current,
      " bytes");

  blocks_.emplace(data, BlockInfo(aligned_nbytes, alloc_stream));

  // Update statistics
  stats_.allocated_bytes[kAggregate].increase(aligned_nbytes);
  stats_.reserved_bytes[kAggregate].increase(aligned_nbytes);
  stats_.num_device_alloc++;

  return data;
}

void DeviceMemoryAllocator::free(void* ptr) {
  if (!ptr) {
    return;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = blocks_.find(ptr);
  if (it != blocks_.end()) {
    auto& block = it->second;
    if (block.state == BlockState::Cached) {
      TORCH_WARN(
          "Attempted to free an OpenReg memory pointer ",
          ptr,
          " on device ",
          device_index_,
          " that is already cached. This likely indicates a double-free.");
      return;
    }
    if (block.state == BlockState::Deferred) {
      TORCH_WARN(
          "Attempted to free an OpenReg memory pointer ",
          ptr,
          " on device ",
          device_index_,
          " that is pending deferred reuse. This likely indicates a double-free.");
      return;
    }

    if (block.stream_uses.empty()) {
      stats_.allocated_bytes[kAggregate].decrease(block.size_bytes);
      moveBlockToCacheNoLock(ptr, block);
      return;
    }

    // Defer reuse until all recorded streams complete. We record one event per
    // stream-use and move the block back to the cache once all events are ready.
    c10::DeviceGuard device_guard{
        c10::Device(c10::DeviceType::PrivateUse1, device_index_)};

    std::vector<orEvent_t> created_events;
    created_events.reserve(block.stream_uses.size());
    auto cleanup = c10::make_scope_exit([&]() noexcept {
      // Best-effort cleanup on exceptional paths. Avoid throwing from a
      // destructor-like scope guard.
      for (auto ev : created_events) {
        if (ev != nullptr) {
          (void)orEventDestroy(ev);
        }
      }
    });

    for (const auto& use_stream : block.stream_uses) {
      TORCH_INTERNAL_ASSERT(
          use_stream.device_type() == c10::DeviceType::PrivateUse1,
          "OpenReg allocator recorded a non-PrivateUse1 stream unexpectedly");
      TORCH_CHECK(
          use_stream.device_index() == device_index_,
          "recordStream() recorded a stream on device ",
          use_stream.device_index(),
          " for an allocation on device ",
          device_index_,
          ".");

      orEvent_t ev = nullptr;
      OPENREG_CHECK(orEventCreate(&ev));
      OPENREG_CHECK(orEventRecord(ev, OpenRegStream(use_stream)));
      created_events.push_back(ev);
    }

    // Transfer ownership to the block now that all events are recorded.
    block.events = std::move(created_events);
    cleanup.release();

    stats_.allocated_bytes[kAggregate].decrease(block.size_bytes);
    block.stream_uses.clear();
    block.state = BlockState::Deferred;
    deferred_pointers_.insert(ptr);
    return;
  }

  // Best-effort orFree for untracked pointers; don't update stats.
  const auto ret = orFree(ptr);
  if (ret == orSuccess) {
    TORCH_WARN(
        "Successfully freed OpenReg memory pointer ",
        ptr,
        " on device ",
        device_index_,
        " that was not tracked by the allocator. "
        "Statistics may be inaccurate.");
  } else {
    TORCH_WARN(
        "orFree failed for untracked pointer ",
        ptr,
        " on device ",
        device_index_,
        ". Return code: ",
        ret,
        ". This likely indicates a double-free or invalid pointer.");
  }
}

void DeviceMemoryAllocator::recordStream(void* ptr, c10::Stream stream) {
  if (!ptr) {
    return;
  }
  if (stream.device_type() != c10::DeviceType::PrivateUse1) {
    // Best-effort: ignore streams that aren't owned by the OpenReg backend.
    return;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = blocks_.find(ptr);
  if (it == blocks_.end()) {
    // Best-effort: ignore pointers not owned by this allocator.
    return;
  }

  auto& block = it->second;
  if (block.state != BlockState::Allocated) {
    // Only track usage for active allocations. This mirrors how other caching
    // allocators treat late/invalid recordStream calls.
    return;
  }

  TORCH_CHECK(
      stream.device_index() == device_index_,
      "recordStream was called with a stream on device ",
      stream.device_index(),
      " for an allocation on device ",
      device_index_,
      ".");

  if (stream == block.alloc_stream) {
    return;
  }

  block.stream_uses.insert(stream);
}

std::vector<void*> DeviceMemoryAllocator::emptyCache() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  processDeferredBlocksNoLock();

  std::vector<void*> removed;
  for (auto cache_it = cached_blocks_by_stream_.begin();
       cache_it != cached_blocks_by_stream_.end();) {
    auto& cache = cache_it->second;
    for (auto it = cache.begin(); it != cache.end();) {
      const size_t nbytes = it->first;
      void* ptr = it->second;
      const auto ret = orFree(ptr);

      if (ret == orSuccess || ret == orErrorUnknown) {
        stats_.reserved_bytes[kAggregate].decrease(nbytes);
        if (ret == orSuccess) {
          stats_.num_device_free++;
        }

        blocks_.erase(ptr);
        removed.push_back(ptr);
        it = cache.erase(it);
      } else {
        TORCH_WARN(
            "orFree failed while emptying OpenReg cache for pointer ",
            ptr,
            " on device ",
            device_index_,
            ". Return code: ",
            ret);
        ++it;
      }
    }

    if (cache.empty()) {
      cache_it = cached_blocks_by_stream_.erase(cache_it);
    } else {
      ++cache_it;
    }
  }
  return removed;
}

c10::CachingDeviceAllocator::DeviceStats DeviceMemoryAllocator::getStats() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return stats_;
}

void DeviceMemoryAllocator::resetAccumulatedStats() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Reset accumulated statistics for all StatTypes
  for (const auto stat_type :
       c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
    stats_.allocated_bytes[stat_type].reset_accumulated();
    stats_.reserved_bytes[stat_type].reset_accumulated();
    stats_.active_bytes[stat_type].reset_accumulated();
    stats_.inactive_split_bytes[stat_type].reset_accumulated();
    stats_.requested_bytes[stat_type].reset_accumulated();
  }

  stats_.num_alloc_retries = 0;
  stats_.num_ooms = 0;
  stats_.num_sync_all_streams = 0;
  stats_.num_device_alloc = 0;
  stats_.num_device_free = 0;
}

void DeviceMemoryAllocator::resetPeakStats() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Reset peak statistics for all StatTypes
  for (const auto stat_type :
       c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
    stats_.allocated_bytes[stat_type].reset_peak();
    stats_.reserved_bytes[stat_type].reset_peak();
    stats_.active_bytes[stat_type].reset_peak();
    stats_.inactive_split_bytes[stat_type].reset_peak();
    stats_.requested_bytes[stat_type].reset_peak();
  }

  stats_.oversize_allocations.reset_peak();
  stats_.oversize_segments.reset_peak();
}

void DeviceMemoryAllocator::processDeferredBlocksNoLock() {
  c10::DeviceGuard device_guard{
      c10::Device(c10::DeviceType::PrivateUse1, device_index_)};

  for (auto it = deferred_pointers_.begin(); it != deferred_pointers_.end();) {
    void* ptr = *it;
    auto block_it = blocks_.find(ptr);
    TORCH_INTERNAL_ASSERT(
        block_it != blocks_.end(),
        "OpenReg allocator missing metadata for deferred pointer ",
        ptr);
    auto& block = block_it->second;
    TORCH_INTERNAL_ASSERT(
        block.state == BlockState::Deferred,
        "Deferred pointer has unexpected state");

    bool all_ready = true;
    for (orEvent_t ev : block.events) {
      const auto err = orEventQuery(ev);
      if (err == orSuccess) {
        continue;
      }
      if (err == orErrorNotReady) {
        all_ready = false;
        break;
      }
      TORCH_CHECK(
          false,
          "orEventQuery failed for OpenReg deferred-free event with error code ",
          err);
    }

    if (!all_ready) {
      ++it;
      continue;
    }

    for (orEvent_t ev : block.events) {
      OPENREG_CHECK(orEventDestroy(ev));
    }
    block.events.clear();

    moveBlockToCacheNoLock(ptr, block);
    it = deferred_pointers_.erase(it);
  }
}

void DeviceMemoryAllocator::moveBlockToCacheNoLock(void* ptr, BlockInfo& block) {
  TORCH_INTERNAL_ASSERT(
      block.state == BlockState::Allocated || block.state == BlockState::Deferred);
  block.state = BlockState::Cached;
  cached_blocks_by_stream_[block.alloc_stream].emplace(block.size_bytes, ptr);
}

namespace {

OpenRegDeviceAllocator g_allocator;

void deleteOpenRegMemory(void* ptr) {
  g_allocator.freeMemory(ptr);
}

}

OpenRegDeviceAllocator::OpenRegDeviceAllocator() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto device_count = c10::openreg::device_count();
  device_allocators_.resize(device_count);
  for (const auto i : c10::irange(device_count)) {
    device_allocators_[i] = std::make_unique<DeviceMemoryAllocator>(i);
  }
}


at::DataPtr OpenRegDeviceAllocator::allocate(size_t nbytes) {
  int current_device_index = -1;
  auto ret = orGetDevice(&current_device_index);
  TORCH_CHECK(ret == orSuccess, "Failed to get current OpenReg device");

  auto curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, current_device_index);

  void* data = nullptr;
  if (nbytes > 0) {
    // Allocate memory via device-specific allocator
    data = device_allocators_[current_device_index]->malloc(nbytes);

    // Track which device owns this pointer
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    allocated_blocks_[data] = current_device_index;
  }

  return {data, data, &deleteOpenRegMemory, curr_device};
}

at::DeleterFnPtr OpenRegDeviceAllocator::raw_deleter() const {
  return &deleteOpenRegMemory;
}

void OpenRegDeviceAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  auto ret = orMemcpy(dest, src, count, orMemcpyDeviceToDevice);
  TORCH_CHECK(
      ret == orSuccess, "Failed to copy ", count, " bytes on openreg device");
}

bool OpenRegDeviceAllocator::initialized() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return !device_allocators_.empty();
}

void OpenRegDeviceAllocator::freeMemory(void* ptr) {
  if (!ptr) {
    return;
  }

  // Try to find which device owns this pointer
  c10::DeviceIndex device_index = -1;
  bool found_in_map = false;

  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = allocated_blocks_.find(ptr);
    if (it != allocated_blocks_.end()) {
      device_index = it->second;
      found_in_map = true;
    }
  }

  if (found_in_map) {
    // Pointer was tracked - free via device-specific allocator with stats
    device_allocators_[device_index]->free(ptr);
  } else {
    // Pointer not tracked - might be already freed by storage or other path
    // Try to free it directly via orFree without updating statistics
    auto ret = orFree(ptr);

    // Only warn if orFree actually failed (not just "not found")
    if (ret != orSuccess && ret != orErrorUnknown) {
      TORCH_WARN(
          "orFree failed for untracked OpenReg memory pointer ",
          ptr,
          ". Error code: ", ret);
    }
  }
}

c10::CachingDeviceAllocator::DeviceStats OpenRegDeviceAllocator::
    getDeviceStats(c10::DeviceIndex device) {
  return device_allocators_[device]->getStats();
}

void OpenRegDeviceAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  device_allocators_[device]->resetAccumulatedStats();
}

void OpenRegDeviceAllocator::resetPeakStats(c10::DeviceIndex device) {
  device_allocators_[device]->resetPeakStats();
}

void OpenRegDeviceAllocator::emptyCache(MempoolId_t mempool_id) {
  (void)mempool_id;
  for (auto& allocator : device_allocators_) {
    auto removed = allocator->emptyCache();
    if (removed.empty()) {
      continue;
    }

    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (void* ptr : removed) {
      allocated_blocks_.erase(ptr);
    }
  }
}

void OpenRegDeviceAllocator::recordStream(
    const DataPtr& ptr,
    c10::Stream stream) {
  if (!ptr.get()) {
    return;
  }
  if (stream.device_type() != c10::DeviceType::PrivateUse1) {
    // Best-effort: ignore streams that aren't owned by the OpenReg backend.
    return;
  }
  if (ptr.get_deleter() != &deleteOpenRegMemory) {
    // Ignore pointers not owned by the OpenReg allocator.
    return;
  }

  c10::DeviceIndex device_index = -1;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = allocated_blocks_.find(ptr.get());
    if (it == allocated_blocks_.end()) {
      // Best-effort: ignore unknown pointers.
      return;
    }
    device_index = it->second;
  }

  // Test-only plumbing counter: this is used by tests to ensure the PrivateUse1
  // guard hook forwards recordDataPtrOnStream() to the allocator's recordStream().
  testBumpRecordStreamCallCount();

  device_allocators_[device_index]->recordStream(ptr.get(), stream);
}
// ============ Global Registration ============

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

} // namespace c10::openreg
