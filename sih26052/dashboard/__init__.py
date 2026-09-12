from .bridge import DashboardBridge

__all__ = [
    "DashboardBridge",
    "broadcast",
    "create_app",
]


def __getattr__(name: str):
    if name in ("broadcast", "create_app"):
        from .server import broadcast, create_app
        return {"broadcast": broadcast, "create_app": create_app}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

