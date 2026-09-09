def test_package_imports():
    import pa_copilot

    assert isinstance(pa_copilot.__version__, str)
    assert pa_copilot.__version__
