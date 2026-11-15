import importlib


def test_gui_module_importable():
    mod = importlib.import_module("bse_renderer.gui")
    assert hasattr(mod, "App")

