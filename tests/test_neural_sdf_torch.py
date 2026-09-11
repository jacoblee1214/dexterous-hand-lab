import pytest

torch = pytest.importorskip("torch")
NeuralSDFModel = pytest.importorskip("neural_sdf.model").NeuralSDFModel
device_module = pytest.importorskip("neural_sdf.device")


def _input(batch=2, contacts=8):
    generator = torch.Generator().manual_seed(91)
    valid = torch.tensor([[True, True, True, False, False, False, False, False]]).expand(batch, -1).clone()
    return {
        "rgb_feature_map": torch.randn(batch, 16, 28, 28, generator=generator),
        "visibility_masks": torch.zeros(batch, 4, 32, 32),
        "intrinsic": torch.tensor([[[30., 0., 15.], [0., 30., 15.], [0., 0., 1.]]]).expand(batch, -1, -1).clone(),
        "camera_from_object": torch.eye(4).expand(batch, -1, -1).clone(),
        "coverage": torch.full((batch, 1), .7),
        "tactile": {
            "position": torch.randn(batch, contacts, 3, generator=generator) * .02,
            "normal": torch.nn.functional.normalize(torch.randn(batch, contacts, 3, generator=generator), dim=-1),
            "scalar_n": torch.rand(batch, contacts, 1, generator=generator),
            "indentation_m": torch.rand(batch, contacts, 1, generator=generator) * .002,
            "sigma_m": torch.full((batch, contacts, 1), .004),
            "patch": torch.randn(batch, contacts, 25, 3, generator=generator) * .02,
            "valid": valid,
            "sensor_index": torch.arange(contacts).expand(batch, -1).clone(),
            "finger_index": (torch.arange(contacts) % 4).expand(batch, -1).clone(),
        },
    }


def test_tactile_encoder_is_permutation_invariant():
    torch.manual_seed(4)
    model = NeuralSDFModel("learned_rgb_tactile_reliability").eval()
    value = _input()
    query = torch.randn(2, 13, 3) * .02
    expected = model(query, value)
    order = torch.tensor([5, 1, 7, 0, 4, 2, 6, 3])
    permuted = {key: item for key, item in value.items() if key != "tactile"}
    permuted["tactile"] = {key: item[:, order] for key, item in value["tactile"].items()}
    assert torch.allclose(expected, model(query, permuted), atol=1e-6, rtol=1e-5)


def test_zero_tactile_is_finite_and_query_has_gradient():
    model = NeuralSDFModel("learned_rgb_tactile_reliability").eval()
    value = _input(batch=1)
    value["tactile"]["valid"][:] = False
    value["coverage"][:] = 0
    query = torch.randn(1, 9, 3).mul(.02).requires_grad_(True)
    prediction = model(query, value)
    prediction.sum().backward()
    assert torch.isfinite(prediction).all()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_rgb_only_has_no_tactile_coverage_side_channel():
    torch.manual_seed(12)
    model = NeuralSDFModel("learned_rgb_only").eval()
    value = _input(batch=1)
    query = torch.randn(1, 11, 3) * .02
    expected = model(query, value)
    value["coverage"][:] = 0
    value["tactile"]["valid"][:] = False
    value["tactile"]["position"] += 5
    assert torch.equal(expected, model(query, value))


def test_model_interface_contains_no_ground_truth_or_shape_identity():
    model = NeuralSDFModel("learned_rgb_only")
    forbidden = ("ground_truth", "oracle", "family", "shape_parameter", "object_id", "radius")
    assert not any(token in name.lower() for name in model.state_dict() for token in forbidden)


def test_auto_device_prefers_cuda_when_visible_and_cpu_remains_explicit():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device_module.resolve_device("auto").type == expected
    assert device_module.resolve_device("cpu").type == "cpu"
