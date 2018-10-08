from .activation import fn

class Layer:
    def __init__(self, n_nodes, **kwargs):
        self.n_nodes = n_nodes
        try :
            # Some Layer has no activation function like Input Layer
            self.activation = kwargs['activation']
        except :
            pass
