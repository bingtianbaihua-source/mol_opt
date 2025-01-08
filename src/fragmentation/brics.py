from .fragmentation import Unit, Connection

class BRICS_Unit(Unit):
    def to_fragment(self, 
                    connection: Connection):
        assert connection in self.connections
        atomMap = {}
        submol = self.graph.get_submol([self], \
                                       atomMap=atomMap)
        if self == connection.units[0]:
            atom_index = atomMap[]