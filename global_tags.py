
class GlobalTags:
    LEGACY_LAYER_BASE = False

    def legacy_layer_base(value=None):
        if value is not None:
            GlobalTags.LEGACY_LAYER_BASE = value
        return GlobalTags.LEGACY_LAYER_BASE
    