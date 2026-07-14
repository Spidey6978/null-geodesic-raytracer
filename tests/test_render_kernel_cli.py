from types import SimpleNamespace

from scripts.render_kernel import resolve_render_args


def test_mode_preset_does_not_override_explicit_cli_values():
    args = SimpleNamespace(
        mode="preview",
        dt=0.33,
        max_steps=1234,
        width=777,
        height=888,
    )

    resolved = resolve_render_args(args)

    assert resolved.dt == 0.33
    assert resolved.max_steps == 1234
    assert resolved.width == 777
    assert resolved.height == 888


def test_mode_preset_applies_when_cli_values_are_missing():
    args = SimpleNamespace(mode="quality", dt=None, max_steps=None, width=None, height=None)

    resolved = resolve_render_args(args)

    assert resolved.dt == 0.1
    assert resolved.max_steps == 5000
    assert resolved.width == 960
    assert resolved.height == 540
