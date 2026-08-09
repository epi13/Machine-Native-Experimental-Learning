import unittest

from mnel.torch_runtime import offload_evidence


class Parameter:
    def __init__(self, device: str, size: int = 4, element_size: int = 4) -> None:
        self.device = device
        self._size = size
        self._element_size = element_size

    def numel(self) -> int:
        return self._size

    def element_size(self) -> int:
        return self._element_size


class Module:
    def __init__(self, hook: bool) -> None:
        if hook:
            self._hf_hook = object()


class Model:
    def __init__(self, *, hook: bool, devices: tuple[str, ...]) -> None:
        self._modules = (Module(hook),)
        self._parameters = tuple(Parameter(device) for device in devices)

    def modules(self):
        return iter(self._modules)

    def parameters(self):
        return iter(self._parameters)


class TorchAdapterEvidenceTests(unittest.TestCase):
    def test_requested_offload_without_residency_evidence_is_not_verified(self) -> None:
        evidence = offload_evidence(Model(hook=True, devices=("cuda:0",)), inference_completed=True)
        self.assertFalse(evidence["sequential_offload_verified"])

    def test_completed_hooked_cpu_backed_run_can_be_verified(self) -> None:
        evidence = offload_evidence(Model(hook=True, devices=("meta",)), inference_completed=True)
        self.assertTrue(evidence["sequential_offload_verified"])
        self.assertEqual(evidence["persistent_cuda_parameter_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
