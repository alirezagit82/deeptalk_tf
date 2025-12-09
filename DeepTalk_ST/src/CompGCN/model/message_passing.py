import inspect
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

# Assuming the scatter_ function from above is defined in the same file or imported

class MessagePassing(Layer):
    r"""Base class for creating message passing layers of Keras.

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    """

    def __init__(self, aggr='add', **kwargs):
        super(MessagePassing, self).__init__(**kwargs)

        # Use inspect to get the arguments of the 'message' and 'update' methods.
        # This allows for flexible, user-defined functions.
        # Note: inspect.getfullargspec is the modern equivalent of getargspec
        try:
            # Python 3
            self.message_args = inspect.getfullargspec(self.message)[0][1:]
            self.update_args = inspect.getfullargspec(self.update)[0][2:]
        except AttributeError:
            # Python 2
            self.message_args = inspect.getargspec(self.message)[0][1:]
            self.update_args = inspect.getargspec(self.update)[0][2:]

    def call(self, edge_index, aggr='add', **kwargs):
        r"""The initial call to start propagating messages.
        This is the TensorFlow equivalent of the PyTorch `forward` or `propagate` method.

        Args:
            edge_index (Tensor): The edge indices.
            aggr (string): The aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
                :obj:`"max"`).
            **kwargs: All additional data which is needed to construct messages
                and to update node embeddings.
        """
        assert aggr in ['add', 'mean', 'max']

        # Determine the number of nodes in the graph.
        # We inspect the kwargs to find a node feature tensor (e.g., 'x') to get the size.
        size = None
        for arg in self.message_args:
            if arg[-2:] in ['_i', '_j']:
                tensor = kwargs[arg[:-2]]
                size = tf.shape(tensor)[0]
                break
        
        # Collect arguments for the `message` function
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                # Gather features for the source nodes of each edge
                message_args.append(tf.gather(kwargs[arg[:-2]], edge_index[0]))
            elif arg[-2:] == '_j':
                # Gather features for the target nodes of each edge
                message_args.append(tf.gather(kwargs[arg[:-2]], edge_index[1]))
            else:
                message_args.append(kwargs[arg])

        # Collect arguments for the `update` function
        update_args = [kwargs[arg] for arg in self.update_args]

        # 1. Construct messages for each edge
        out = self.message(*message_args)
        
        # 2. Aggregate messages for each target node
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        
        # 3. Update node embeddings
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`call`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""
        return x_j

    def update(self, aggr_out):
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`call`."""
        return aggr_out
