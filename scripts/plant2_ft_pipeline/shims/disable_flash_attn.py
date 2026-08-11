"""Disable broken flash_attn before transformers loads Bert/modeling utils."""
def apply():
    import transformers.utils.import_utils as iu
    import transformers.utils as u
    def _no():
        return False
    iu.is_flash_attn_2_available = _no
    u.is_flash_attn_2_available = _no
    # Also clear any cached attribute checks if present
    for fn_name in ("is_flash_attn_greater_or_equal", "is_flash_attn_greater_or_equal_2_10"):
        if hasattr(iu, fn_name):
            setattr(iu, fn_name, lambda *a, **k: False)
            setattr(u, fn_name, lambda *a, **k: False)
