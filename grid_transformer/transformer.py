"""
This module contains the `transformer` function for creating a transformer model with customizable position embedding, pooling, and output layers.
Functions:

- transformer: Returns a transformer model constructor with the provided options.
"""
from typing import Union, Callable, Tuple

from tensorflow.keras.layers import LayerNormalization

from core_tools.core import is_model, log_shape, Predictor
from grid_transformer.transformer_block import TransformerBlock
from grid_transformer.position_embedding import CatPositionEmbedding, SumPositionEmbedding
from grid_transformer.utils import MultiPooling


def Transformer(
        pos_emd="cat",
        size=256,
        pooling="first",
        no=8,
        out_layers=(1000, 1000),
        output_size=10,
        last_norm=True,
        out_pre: Union[None, str, Tuple[str, ...]] = None,
        out_post: Union[None, str, Tuple[str, ...]] = None,
        num_heads=8,
        ff_mul=4,
        ff_size=None,
        dropout=0.1,
        ff_act="gelu",
        show_shape: Union[bool, Callable] = False,
        save_shape=True,
        block_class=TransformerBlock,
        return_attention_scores=False,
):
    if pos_emd == "sum":
        pos_emd = SumPositionEmbedding()
    elif callable(pos_emd):
        pos_emd = pos_emd
    else:
        pos_emd = CatPositionEmbedding()

    # todo mid This should be fix in future.
    # As pooling is just one layers model can't do more then one thing, like detokenization and then "last".
    # The pooling could be list that each element is passed by MultiPooling and then wrapped in Sequential or SeqModel.
    # todo mid Add pooling params like max_output in detokenization.
    pool = MultiPooling(pooling, show_shape=show_shape)
    predictor = out_layers if is_model(out_layers) else Predictor(
        model=out_layers,
        output_size=output_size,
        pre=out_pre,
        post=out_post,
        activation="gelu"
    )

    @log_shape(show_shape, "Transformer")
    def apply(x):
        nonlocal show_shape
        nonlocal save_shape
        nonlocal size

        x = pos_emd(x)

        scores = []
        for _ in range(no):
            trans = block_class(
                num_heads=num_heads,
                # size=size,
                ff_mul=ff_mul,
                ff_size=ff_size,
                dropout=dropout,
                ff_act=ff_act,
                return_attention_scores=return_attention_scores
            )
            if return_attention_scores:
                x, score = trans(x)
                scores.append(score)
            else:
                x = trans(x)

        if last_norm:
            x = LayerNormalization(epsilon=1e-6)(x)

        # if show_shape:
        #     print(f"Transformer block shape: {x.shape}")

        x = pool(x)

        if predictor:
            x = predictor(x)

        if return_attention_scores:
            return x, scores
        return x

    return apply
