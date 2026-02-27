#include "OpenRegTestAsync.h"

#include "OpenRegStream.h"

#include <include/openreg.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace c10::openreg {
namespace {

struct BlockingGate {
  std::mutex mutex;
  std::condition_variable cv;
  bool released{false};
};

std::atomic<int64_t> g_next_gate_id{1};
std::mutex g_gates_mutex;
std::unordered_map<int64_t, std::shared_ptr<BlockingGate>> g_gates;

std::shared_ptr<BlockingGate> getGateOrThrow(int64_t gate_id) {
  std::lock_guard<std::mutex> lock(g_gates_mutex);
  auto it = g_gates.find(gate_id);
  TORCH_CHECK(it != g_gates.end(), "Unknown OpenReg test gate id: ", gate_id);
  return it->second;
}

} // namespace

int64_t testCreateBlockingGate() {
  auto gate = std::make_shared<BlockingGate>();
  const int64_t gate_id = g_next_gate_id.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(g_gates_mutex);
    g_gates.emplace(gate_id, std::move(gate));
  }
  return gate_id;
}

void testReleaseBlockingGate(int64_t gate_id) {
  std::shared_ptr<BlockingGate> gate;
  {
    std::lock_guard<std::mutex> lock(g_gates_mutex);
    auto it = g_gates.find(gate_id);
    if (it == g_gates.end()) {
      return;
    }
    gate = std::move(it->second);
    g_gates.erase(it);
  }

  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    gate->released = true;
  }
  gate->cv.notify_all();
}

void testEnqueueWaitForGate(const c10::Stream& stream, int64_t gate_id) {
  TORCH_CHECK(
      stream.device_type() == c10::DeviceType::PrivateUse1,
      "Expected an OpenReg (PrivateUse1) stream, but got ",
      stream);

  auto gate = getGateOrThrow(gate_id);

  // Enqueue a task that blocks until the gate is released. This is used by
  // tests to deterministically create in-flight work on a stream without
  // relying on wall-clock sleeps.
  OpenRegStream or_stream{stream};
  const auto status =
      ::openreg::addTaskToStream(or_stream, [gate = std::move(gate)]() {
        std::unique_lock<std::mutex> lock(gate->mutex);
        gate->cv.wait(lock, [&] { return gate->released; });
      });
  TORCH_CHECK(status == ::orSuccess, "Failed to enqueue OpenReg stream task");
}

} // namespace c10::openreg
