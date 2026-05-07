class SemanticEngineError(Exception):
    pass


class DataArtifactMissingError(SemanticEngineError):
    pass


class RowAlignmentError(SemanticEngineError):
    pass


class ViewShapeMismatchError(SemanticEngineError):
    pass


class InvalidValueError(SemanticEngineError):
    pass
