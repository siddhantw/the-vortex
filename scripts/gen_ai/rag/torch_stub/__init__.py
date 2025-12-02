# PyTorch stub for Streamlit compatibility
class _path(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getattr__(self, name):
        return self
