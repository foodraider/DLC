class Batch:
    """
    This class provides the most basic data structure for training:

    """

    def __init__(self):
        self.inputMsgIDs = []
        self.inputMsgLength = []
        self.outputResponseIDs = []
        self.outputResponseLength = []
