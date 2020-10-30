from . import training, plotting


def deserialize(data):
    if hasattr(training, data["name"]):
        return getattr(training, data["name"])(
            *data.get("args", []), **data.get("kwargs", {})
        )

    if hasattr(plotting, data["name"]):
        return getattr(plotting, data["name"])(
            *data.get("args", []), **data.get("kwargs", {})
        )

    raise ValueError("Callback '{}' not found".format(data["name"]))
