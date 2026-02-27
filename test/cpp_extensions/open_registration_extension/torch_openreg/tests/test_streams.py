# Owner(s): ["module: PrivateUse1"]

import os

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


# Enable deterministic test-only helpers exposed by torch_openreg._C.
os.environ.setdefault("TORCH_OPENREG_ENABLE_TEST_APIS", "1")


class TestStream(TestCase):
    @skipIfTorchDynamo()
    def test_stream_create(self):
        """Test stream creation with different methods"""
        stream = torch.Stream(device="openreg")
        self.assertEqual(stream.device_index, torch.openreg.current_device())
        stream = torch.Stream(device="openreg:1")
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream = torch.Stream(1)
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream1 = torch.Stream(
            stream_id=stream.stream_id,
            device_type=stream.device_type,
            device_index=stream.device_index,
        )
        self.assertEqual(stream, stream1)

    @skipIfTorchDynamo()
    def test_stream_context(self):
        """Test stream context manager"""
        with torch.Stream(device="openreg:1") as stream:
            self.assertEqual(torch.accelerator.current_stream(), stream)

    def test_stream_context_exception_restore(self):
        prev = torch.accelerator.current_stream()
        inner_stream = torch.Stream(device="openreg:1")
        try:
            with inner_stream:
                # inside the context we should be on the inner stream
                self.assertEqual(torch.accelerator.current_stream(), inner_stream)
                raise RuntimeError("forced")
        except RuntimeError:
            pass
        # After the exception, the current stream should be restored.
        self.assertEqual(torch.accelerator.current_stream(), prev)

    @skipIfTorchDynamo()
    def test_stream_switch(self):
        """Test switching between streams"""
        stream1 = torch.Stream(device="openreg:0")
        torch.accelerator.set_stream(stream1)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)

        stream2 = torch.Stream(device="openreg:1")
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)
        torch.accelerator.set_stream(stream2)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream2)

    @skipIfTorchDynamo()
    def test_stream_synchronize(self):
        """Test stream synchronization"""
        stream = torch.Stream(device="openreg:1")
        self.assertEqual(True, stream.query())

        event = torch.Event()
        event.record(stream)
        stream.synchronize()
        self.assertEqual(True, stream.query())

    @skipIfTorchDynamo()
    def test_stream_repr(self):
        """Test stream string representation"""
        stream = torch.Stream(device="openreg:1")
        self.assertTrue(
            "torch.Stream device_type=openreg, device_index=1" in repr(stream)
        )

    @skipIfTorchDynamo()
    def test_stream_wait_stream(self):
        """Test stream waiting on another stream"""
        stream_1 = torch.Stream(device="openreg:0")
        stream_2 = torch.Stream(device="openreg:1")
        stream_2.wait_stream(stream_1)

    @skipIfTorchDynamo()
    def test_stream_wait_event(self):
        """Test stream waiting on event"""
        s1 = torch.Stream(device="openreg")
        s2 = torch.Stream(device="openreg")
        e = s1.record_event()
        s2.wait_event(e)

    @skipIfTorchDynamo()
    def test_stream_equality(self):
        """Test stream equality comparison"""
        stream1 = torch.Stream(device="openreg:0")
        stream2 = torch.Stream(device="openreg:0")

        # Different streams should not be equal
        self.assertNotEqual(stream1, stream2)

        # Same stream should be equal to itself
        self.assertEqual(stream1, stream1)

        # Stream created with same parameters should be equal
        stream3 = torch.Stream(
            stream_id=stream1.stream_id,
            device_type=stream1.device_type,
            device_index=stream1.device_index,
        )
        self.assertEqual(stream1, stream3)

    @skipIfTorchDynamo()
    def test_stream_multiple_devices(self):
        """Test streams on multiple devices"""
        stream0 = torch.Stream(device="openreg:0")
        stream1 = torch.Stream(device="openreg:1")

        self.assertEqual(stream0.device_index, 0)
        self.assertEqual(stream1.device_index, 1)

        # Set current stream for each device
        torch.accelerator.set_device_index(0)
        torch.accelerator.set_stream(stream0)
        self.assertEqual(torch.accelerator.current_stream(), stream0)

        torch.accelerator.set_device_index(1)
        torch.accelerator.set_stream(stream1)
        self.assertEqual(torch.accelerator.current_stream(), stream1)

    @skipIfTorchDynamo()
    def test_stream_context_nested(self):
        """Test nested stream contexts"""
        stream1 = torch.Stream(device="openreg:0")
        stream2 = torch.Stream(device="openreg:0")

        with stream1:
            self.assertEqual(torch.accelerator.current_stream(), stream1)
            with stream2:
                self.assertEqual(torch.accelerator.current_stream(), stream2)
            # Should restore to stream1
            self.assertEqual(torch.accelerator.current_stream(), stream1)

    @skipIfTorchDynamo()
    def test_stream_record_event(self):
        """Test recording events on streams"""
        stream = torch.Stream(device="openreg")
        event = stream.record_event()

        self.assertIsNotNone(event)
        self.assertEqual(event.device.type, "openreg")
        stream.synchronize()
        self.assertTrue(event.query())


class TestOpenRegAsyncHelper(TestCase):
    @skipIfTorchDynamo()
    def test_event_not_ready_until_stream_sync(self):
        # This test is intentionally deterministic: it does not rely on
        # wall-clock sleeps. Instead, we enqueue a blocking task onto the stream
        # and confirm that an event recorded *after* that task is not ready
        # until the stream is explicitly synchronized.
        import torch_openreg._C  # type: ignore[import-not-found]

        torch.openreg.init()

        stream = torch.Stream(device="openreg:0")
        gate = torch_openreg._C._test_create_blocking_gate()
        released = False
        try:
            torch_openreg._C._test_enqueue_wait_for_gate(stream, gate)

            event = torch.Event(device="openreg:0")
            event.record(stream)
            self.assertFalse(event.query())

            torch_openreg._C._test_release_blocking_gate(gate)
            released = True
            stream.synchronize()
            self.assertTrue(event.query())
        finally:
            # Best-effort cleanup to avoid leaving a blocked stream worker
            # thread if the test fails mid-way.
            if not released:
                torch_openreg._C._test_release_blocking_gate(gate)


class TestOpenRegGuardRecordDataPtrOnStream(TestCase):
    @skipIfTorchDynamo()
    def test_record_data_ptr_on_stream_forwards_to_allocator(self):
        # This is a smoke test for the PrivateUse1 guard hook
        # DeviceGuardImplInterface::recordDataPtrOnStream.
        #
        # CUDA/XPU override this hook to forward to their caching allocators.
        # OpenReg should do the same so stream recording is exercised by core
        # call sites.
        import torch_openreg._C  # type: ignore[import-not-found]

        torch.openreg.init()

        torch_openreg._C._test_reset_record_stream_calls()
        x = torch.ones(8, 8, device="openreg:0")
        s = torch.Stream(device="openreg:0")

        torch_openreg._C._test_record_data_ptr_on_stream(x, s)
        self.assertGreater(torch_openreg._C._test_get_record_stream_calls(), 0)


if __name__ == "__main__":
    run_tests()
