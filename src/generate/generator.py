from model.network import *

TERMINATION = 1
ADDITION = 2
FAIL = 3

class MoleculeBuilder():

    def __init__(self, config):
        self.model: BlockConnectionPredictor
        self.library: 