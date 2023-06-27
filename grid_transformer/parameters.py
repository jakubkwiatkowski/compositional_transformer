"""
The  module is a set of classes that define the parameters for Transformer models and GridTransformer models. The `TransformerParameters` class is a dataclass that holds the parameters for Transformer models, such as the number of layers, hidden state dimension, and dropout rate. The `GridTransformerParameters` class is a subclass of `TransformerParameters` that holds additional parameters specific to GridTransformer models, such as the extractor type, mapping function, and masking strategy.
You can then pass the instance to a function or class that uses these parameters to define the architecture of the model.

To use these classes, you can create an instance of the class with the desired values for the parameters. For example:
```python
from dataclasses import asdict
params = TransformerParameters(no=6, size=256, dropout=0.2)
model = transformer(asdict(params))
```

For GridTransformerParameters:
```python
params = GridTransformerParameters(extractor='ef',last='start',last_index=8,col=3,row=3,return_attention_scores=True)
model = grid_trans(asdict(params))
```

"""
from dataclasses import dataclass
from typing import Tuple, Union

from grid_transformer.constants import LEARNABLE, BATCH, LAST


@dataclass
class TransformerParameters:
    """
    Class that holds parameters for Transformer models.
    :param pos_emd: Type of position embedding. Can be 'sum' or 'cat'.
    :param no: Number of layers in the Transformer encoder.
    :param size: Dimension of the hidden states and the self-attention layers.
    :param ff_mul: Multiplier for the dimension of the feed-forward layers.
    :param ff_size: Dimension of the feed-forward layers. If None, it is set to size * ff_mul.
    :param dropout: Dropout rate for the self-attention and feed-forward layers.
    :param num_heads: Number of heads in the multi-head self-attention layers.
    :param ff_act: Activation function for the feed-forward layers. Can be 'relu' or 'gelu'.
    :param extractor: Type of extractor to use before the transformer layers. Can be 'cnn' or 'rnn'.
    :param extractor_pooling: Type of pooling to use after the extractor. Can be 'first' or 'flat2d'.
    :param pooling: Type of pooling to use after the transformer layers. Can be 'first' or 'mean'.
    :param out_layers: Tuple containing the number of neurons for the output layers.
    :param show_shape: Whether to print shape of the intermediate tensors.
    """
    pos_emd: str = "sum"
    no: int = 4
    size: int = 128
    ff_mul: int = 4
    ff_size: int = None
    dropout: float = 0.1
    num_heads: int = 8
    ff_act: str = "gelu"
    extractor: str = None
    extractor_pooling: str = "flat2d"
    pooling: str = "first"
    out_layers: Tuple = (1000, 1000)
    out_pre: Union[None, str, Tuple[str, ...]] = None
    out_post: Union[None, str, Tuple[str, ...]] = None
    show_shape: bool = True


@dataclass
class GridTransformerParameters(TransformerParameters):
    """
    Class that holds parameters for grid_transformer models.
    :param extractor: Type of extractor to use for the image sequence. Can be 'ef' for EfficientNet or 'res' for ResNet.
    :param extractor_input: The shape of the extracted features from the extractor.
    :param mask_value: How to handle the last element of the sequence, can be 'start' to make it the first element or 'end' to make it the last element
    :param map_fn: The function used to map the image sequence, can be 'batch' or 'map'
    :param last_index: The index of the last element of the sequence
    :param channel: The channel used to map the image sequence, can be 'channel' or 'spatial'
    :param batch_no: Number of batches to use for the map function
    :param col: Number of columns in the image sequence
    :param row: Number of rows in the image sequence
    :param return_extractor_input: Whether to return the input to the extractor
    :param return_attention_scores: Whether to return the attention scores
    :param mask: The masking strategy to use, can be 'last' or 'all'
    :param pre: Preprocessing function to use on the image sequence, can be 'normalize' or 'standardize'
    :param noise: Noise function to use on the image sequence, can be 'gaussian' or 'poisson'
    :param augmentation: Augmentation function to use on the image sequence, can be 'random_rotate' or 'random_flip'
    :param print_: Whether to print information about the model
    """
    extractor: int = "ef"
    extractor_input: int = 84
    mask_value: str = LEARNABLE
    map_fn: str = BATCH
    last_index: int = None
    channel: str = "channel"
    batch_no: int = None
    col: int = 3
    row: int = 3
    return_extractor_input: bool = False
    return_attention_scores: bool = False
    mask: str = LAST
    pre: str = None
    noise: str = None
    augmentation: Union[str, None] = None
    print_: bool = False


@dataclass
class MidTransformerParameters(TransformerParameters):
    """
    Class that holds parameters for Mid-level Transformer models.
    """
    size: int = 256
    num_heads: int = 8
    no: int = 4


@dataclass
class SmallTransformerParameters(TransformerParameters):
    """
    Class that holds parameters for Small-level Transformer models.
    """
    size: int = 64
    num_heads: int = 4
    no: int = 4


@dataclass
class BigTransformerParameters(TransformerParameters):
    """
    Class that holds parameters for Big-level Transformer models.
    """
    size: int = 768
    num_heads: int = 12
    no: int = 12


@dataclass
class LargeTransformerParameters(TransformerParameters):
    """
    Class that holds parameters for Large-level Transformer models.
    """
    size: int = 1024
    num_heads: int = 16
    no: int = 24
