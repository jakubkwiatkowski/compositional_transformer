"""
This module contains functions and classes for creating grid transformer with augmentation and preprocessing.

Functions:

- `augmented_transformer`:
  This function creates grid transformer with augmentation and preprocessing. It takes several parameters such as last_index, pre, noise, mask, augmentation, root, transpose, print_, output_names, return_attention_scores and return_extractor_input and returns a constructor for grid transformer with augmentation.
"""

from functools import partial
from typing import Union, List, Callable, Tuple

from core_tools.core import TARGET, OUTPUT, INDEX, MASK, ATTENTION, IMAGE, LABELS, PREDICT, INPUT, INPUTS, \
    InferencePass, lw, log_shape, il

from grid_transformer import aug
from grid_transformer.image_transformer import ImageTransformer
from grid_transformer.mask import random_mask, constant_mask, no_mask
from grid_transformer.preprocessing import get_images, get_images_with_index, random_last, get_images_no_answer, \
    repeat_last

RANDOM = "random"

LAST = "last"

RANDOM_LAST = "random_last"

NO_ANSWER = "no_answer"

IMAGES = "images"


def GridTransformer(
        last_index: int = 8,
        pre: str = 'images',
        noise: Union[Tuple[str, float], float, None] = None,
        mask: Union[str, int, Callable, None] = LAST,
        augmentation: Union[str, List[str], None] = None,
        root: int = 2,
        transpose: bool = True,
        print_: bool = False,
        output_names: Union[str, List[str], None] = None,
        return_attention_scores: bool = False,
        return_extractor_input: bool = False,
        show_shape: Union[bool, Callable] = True,
        **kwargs) -> Callable:
    """
    This function return constructor for grid transformer.
    The Grid Transformer is a wrapper model for ImageTransformers that operates on grid-like inputs.
    This model extends the ImageTransformer by adding a pre-processing step, masking creation and an augmentation step for grid-like inputs,
    inputs that have multiple outputs.

     that operates on grid-like inputs.


    Args:
        last_index (int): The last index of the dataset. Default is 8.
        pre (str): The type of pre-processing to apply to the images. Can be 'images', 'index', 'no_answer', 'last' or 'random_last'. Default is 'images'.
        noise (Union[Tuple[str, float], float, None]): The type of noise to apply to the images. Can be 'batch' or 'image'. Default is None.
        mask (Union[str, int, Callable, None]): The type of mask to apply to the images. Can be 'random', 'last', an integer, or a callable. Default is None.
        augmentation (Union[str, List[str], None]): The type of augmentation to apply to the images. Can be a string or a list of strings. Default is None.
        root (int): The root value to apply to the augmentation. Default is 2.
        transpose (bool): Whether to transpose the images before applying the augmentation. Default is True.
        print_ (bool): Whether to print the progress of the augmentation. Default is False.
        output_names (Union[str, List[str], None]): The names of the output. Default is None.
        return_attention_scores (bool): Whether to return attention scores or not. Default is False.
        return_extractor_input (bool): Whether to return the input to the extractor or not. Default is False.
        kwargs: Additional keyword arguments to pass to the grid_trans function.

    Returns:
        A constructor for grid transformer with augmentation.
    """
    if pre == IMAGES:
        pre = partial(get_images, last_index=last_index)
    elif pre == INDEX:
        pre = partial(get_images_with_index, index=0, last_index=last_index)
    elif pre == NO_ANSWER:
        pre = partial(get_images_no_answer, last_index=last_index)
    elif pre == LAST:
        pre = partial(repeat_last, last_index=last_index)
    elif pre == RANDOM_LAST:
        pre = partial(random_last, last_index=last_index)
    elif il(pre):
        pre = SubClassing(pre)

    if il(noise):
        noise, noise_prob = noise[0], noise[1]
    elif isinstance(noise, float):
        noise, noise_prob = "batch", noise
    else:
        noise_prob = 1.0

    if noise == "batch":
        noise = InferencePass(aug.BatchNoise(last_index=last_index, prob=noise_prob), print_=print_)
        # noise = InferencePass(partial(aug.batch_noise, last_index=last_index, prob=noise_prob))
    elif noise:
        noise = InferencePass(aug.Noise(last_index=last_index, prob=noise_prob), print_=print_)
        # noise = InferencePass(partial(aug.Noise, last_index=last_index, prob=noise_prob))

    if mask:
        if mask == RANDOM:
            mask = partial(random_mask, last_index=last_index + 1)
        elif mask == LAST:
            mask = partial(constant_mask, value=last_index)
        elif mask == "no":
            mask = partial(no_mask, last_index=last_index + 1)
        elif mask == INPUT:
            mask = None
        elif isinstance(mask, int):
            mask = partial(constant_mask, value=mask)

    if augmentation:
        # noinspection PyTypeChecker
        augmentation = InferencePass(aug.rand(augmentation, root=root, transpose=transpose), print_=print_)

    output_names = get_transformer_output_names(output_names, return_attention_scores, return_extractor_input)

    model = ImageTransformer(
        return_attention_scores=return_attention_scores,
        return_extractor_input=return_extractor_input,
        show_shape=show_shape,
        **kwargs
    )

    # def _input_shape(*args, **kwargs):
    #     if hasattr(args[0][INPUTS], "numpy"):
    #         show_raven(args[0])

    # @log_shape(show_shape, "GridTransformer", in_fn=_input_shape)
    @log_shape(show_shape, "GridTransformer")
    def apply(x):
        inputs = x[INPUTS]
        target = x[TARGET] if TARGET in x else None
        labels = target  # later will be use for metrics calculation
        if pre:
            index = x[INDEX] if INDEX in x else None
            inputs = pre((inputs, index))
            if target is not None:
                target = pre((target, index))
        if noise:
            inputs = noise(inputs)

        if mask:
            mask_ = mask(inputs)
        else:
            mask_ = x[MASK]

        if augmentation:
            inputs, target, mask_ = augmentation(inputs, target, mask_)

        output = lw(model(inputs, mask_))
        return {
            **x,
            INPUTS: inputs,
            LABELS: labels,
            TARGET: target,
            MASK: mask_,
            **{name: output[i] for i, name in enumerate(output_names)}
        }

    return apply


def get_transformer_output_names(
        output_names: Union[str, List[str], None],
        return_attention_scores: bool,
        return_extractor_input: bool
) -> List[str]:
    """
    This function returns the output names of the model.

    Args:
        output_names (Union[str, List[str], None]): The names of the output. Default is None.
        return_attention_scores (bool): Whether to return attention scores or not.
        return_extractor_input (bool): Whether to return the input to the extractor or not.

    Returns:
        A list of names of the outputs.
    """
    if not output_names:
        output_names = [OUTPUT]
        if return_attention_scores:
            output_names += [ATTENTION]
        if return_extractor_input:
            output_names += [IMAGE]
    return output_names


def get_model_output_names(
        predict_names: Union[str, List[str], None],
        return_attention_scores: bool,
        return_extractor_input: bool,
        additional_names: Union[str, List[str], Tuple[str, ...], None] = None

) -> List[str]:
    """
    This function returns the output names of the model.

    Args:
        predict_names (Union[str, List[str], None]): The names of the output. Default is None.
        return_attention_scores (bool): Whether to return attention scores or not.
        return_extractor_input (bool): Whether to return the input to the extractor or not.
        return_metrics (bool): Whether to return metrics or not

    Returns:
        A list of names of the outputs.
    """
    if not predict_names:
        predict_names = [PREDICT, "predict_mask"]
        if return_attention_scores:
            predict_names += [ATTENTION]
        if return_extractor_input:
            predict_names += [IMAGE]
        if additional_names:
            predict_names += additional_names
    return predict_names
