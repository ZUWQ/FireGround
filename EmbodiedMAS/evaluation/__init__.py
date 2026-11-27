"""Episode evidence and offline scoring."""

def __getattr__(name):
    if name in {"EpisodeMonitor", "record_event"}:
        from . import monitor
        return getattr(monitor, name)
    raise AttributeError(name)
