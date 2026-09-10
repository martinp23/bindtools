class EquilibriumError(Exception):
    def __init__(self, message, val, params):
        super().__init__(message)
        self.val = val
        self.params = params

