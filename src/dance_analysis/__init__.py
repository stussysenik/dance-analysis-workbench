"""Interactive dance analysis workbench."""

__all__ = ["build_app", "launch"]


def build_app():
    from dance_analysis.app import build_app as _build_app

    return _build_app()


def launch():
    from dance_analysis.app import launch as _launch

    return _launch()
