#include <ATen/Context.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/util/env.h>
#include <torch/csrc/Stream.h>

#include <runtime/OpenRegFunctions.h>
#include <runtime/OpenRegTestAllocator.h>
#include <runtime/OpenRegTestAsync.h>

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(torch::utils::is_device_in_bad_fork(at::kPrivateUse1));
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR
static PyObject* _getDefaultGenerator(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_default_generator expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  return THPGenerator_initDefaultGenerator(
      at::globalContext().defaultGenerator(
          c10::Device(c10::DeviceType::PrivateUse1, idx)));

  END_HANDLE_TH_ERRORS
}
// LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR

// LITERALINCLUDE START: MODULE SET DEVICE HELPER

PyObject* _setDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackDeviceIndex(arg);
  torch::utils::device_lazy_init(at::kPrivateUse1);
  c10::openreg::set_device(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE END: MODULE SET DEVICE HELPER

PyObject* _exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = c10::openreg::ExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDevice(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto device = static_cast<int32_t>(c10::openreg::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDeviceCount(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  return THPUtils_packUInt64(c10::openreg::device_count());
  END_HANDLE_TH_ERRORS
}

namespace {

c10::Stream unpackStreamFromPyObject(PyObject* stream_obj) {
  TORCH_CHECK_TYPE(
      THPStream_Check(stream_obj),
      "Expected a torch.Stream object, but got ",
      Py_TYPE(stream_obj)->tp_name);
  auto* thp_stream = reinterpret_cast<THPStream*>(stream_obj);
  return c10::Stream::unpack3(
      thp_stream->stream_id,
      static_cast<c10::DeviceIndex>(thp_stream->device_index),
      static_cast<c10::DeviceType>(thp_stream->device_type));
}

} // namespace

static PyObject* _test_create_blocking_gate(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  return THPUtils_packInt64(c10::openreg::testCreateBlockingGate());
  END_HANDLE_TH_ERRORS
}

static PyObject* _test_release_blocking_gate(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS

  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_test_release_blocking_gate expects an int, but got ",
      THPUtils_typename(arg));

  const int64_t gate_id = THPUtils_unpackLong(arg);
  c10::openreg::testReleaseBlockingGate(gate_id);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _test_enqueue_wait_for_gate(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS

  PyObject* stream_obj = nullptr;
  PyObject* gate_id_obj = nullptr;
  TORCH_CHECK(
      PyArg_ParseTuple(args, "OO", &stream_obj, &gate_id_obj),
      "Expected (stream, gate_id) arguments");

  torch::utils::device_lazy_init(at::kPrivateUse1);

  TORCH_CHECK(
      THPUtils_checkLong(gate_id_obj),
      "_test_enqueue_wait_for_gate expects an int gate id, but got ",
      THPUtils_typename(gate_id_obj));

  const auto stream = unpackStreamFromPyObject(stream_obj);
  TORCH_CHECK(
      stream.device_type() == c10::DeviceType::PrivateUse1,
      "Expected an OpenReg (PrivateUse1) stream, but got ",
      stream);

  const int64_t gate_id = THPUtils_unpackLong(gate_id_obj);
  c10::openreg::testEnqueueWaitForGate(stream, gate_id);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _test_get_record_stream_calls(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(c10::openreg::testGetRecordStreamCallCount());
  END_HANDLE_TH_ERRORS
}

static PyObject* _test_reset_record_stream_calls(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::openreg::testResetRecordStreamCallCount();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _test_record_data_ptr_on_stream(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS

  PyObject* tensor_obj = nullptr;
  PyObject* stream_obj = nullptr;
  TORCH_CHECK(
      PyArg_ParseTuple(args, "OO", &tensor_obj, &stream_obj),
      "Expected (tensor, stream) arguments");

  torch::utils::device_lazy_init(at::kPrivateUse1);

  TORCH_CHECK_TYPE(
      THPVariable_Check(tensor_obj),
      "Expected a Tensor, but got ",
      Py_TYPE(tensor_obj)->tp_name);

  const auto tensor = THPVariable_Unpack(tensor_obj);
  TORCH_CHECK(
      tensor.device().type() == c10::DeviceType::PrivateUse1,
      "Expected an OpenReg (PrivateUse1) tensor, but got ",
      tensor.device());

  const auto stream = unpackStreamFromPyObject(stream_obj);
  TORCH_CHECK(
      stream.device_type() == c10::DeviceType::PrivateUse1,
      "Expected an OpenReg (PrivateUse1) stream, but got ",
      stream);

  const auto guard = c10::impl::VirtualGuardImpl(tensor.device().type());
  guard.recordDataPtrOnStream(tensor.storage().data_ptr(), stream);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE START: OPENREG MODULE METHODS
static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_isInBadFork", _isInBadFork, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {"_get_device", _getDevice, METH_NOARGS, nullptr},
    {"_set_device", _setDevice, METH_O, nullptr},
    {"_exchangeDevice", _exchangeDevice, METH_O, nullptr},
    {"_get_device_count", _getDeviceCount, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};
// LITERALINCLUDE END: OPENREG MODULE METHODS

static PyMethodDef test_methods[] = {
    {"_test_create_blocking_gate", _test_create_blocking_gate, METH_NOARGS, nullptr},
    {"_test_release_blocking_gate", _test_release_blocking_gate, METH_O, nullptr},
    {"_test_enqueue_wait_for_gate", _test_enqueue_wait_for_gate, METH_VARARGS, nullptr},
    {"_test_get_record_stream_calls", _test_get_record_stream_calls, METH_NOARGS, nullptr},
    {"_test_reset_record_stream_calls", _test_reset_record_stream_calls, METH_NOARGS, nullptr},
    {"_test_record_data_ptr_on_stream", _test_record_data_ptr_on_stream, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static bool testApisEnabled() {
  auto flag = c10::utils::check_env("TORCH_OPENREG_ENABLE_TEST_APIS");
  return flag.has_value() && *flag;
}
/*
 * When ASAN is enabled, PyTorch modifies the dlopen flag during import,
 * causing all global and weak symbols in _C.so and its dependent libraries
 * to be exposed to the global symbol scope, which in turn causes
 * subsequent symbols with the same name in other libraries to be intercepted.
 * Therefore, it cannot be named initModule here, otherwise initModule
 * in torch/csrc/Module.cpp will be called, resulting in failure.
 */
extern "C" OPENREG_EXPORT PyObject* initOpenRegModule(void) {
  static struct PyModuleDef openreg_C_module = {
      PyModuleDef_HEAD_INIT, "torch_openreg._C", nullptr, -1, methods};
  PyObject* mod = PyModule_Create(&openreg_C_module);

  if (testApisEnabled()) {
    TORCH_CHECK(
        PyModule_AddFunctions(mod, test_methods) == 0,
        "Failed to register torch_openreg._C test-only APIs");
  }

  return mod;
}
