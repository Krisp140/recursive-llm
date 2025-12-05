
class FinalAnswer(Exception):
    """Exception raised when FINAL() is called in REPL."""
    def __init__(self, value):
        self.value = value

