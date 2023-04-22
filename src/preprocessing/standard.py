__all__ = [get_string_lookup_layer, get_integer_lookup_layer, get_normalization_layer]

from tensorflow.keras.layers import IntegerLookup     # Used in Feature processing
from tensorflow.keras.layers import Normalization     # Used in Feature processing
from tensorflow.keras.layers import StringLookup      # Used in Feature processing

def get_string_lookup_layer(name, dataset, max_tokens=None):
    """Generates a StringLookup layer for a feature.

    Args:
        name (str): Name of the feature.
        dataset (tf.data.Dataset): Dataset that the feature belongs to.
        max_tokens (int): Maximum number of unique values to encode. If None, all unique values will be encoded.

    Returns:
        A StringLookup layer.
    """
    lookup = StringLookup(max_tokens=max_tokens)
    feature_ds = dataset.map(lambda x, y: x[name])
    lookup.adapt(feature_ds)

    return lookup



def get_integer_lookup_layer(name, dataset, max_tokens=None):
    """Generates an IntegerLookup layer for a feature.

    Args:
        name (str): Name of the feature.
        dataset (tf.data.Dataset): Dataset that the feature belongs to.
        max_tokens (int): Maximum number of unique values to encode. If None, all unique values will be encoded.

    Returns:
        An IntegerLookup layer.
    """
    lookup = IntegerLookup(max_tokens=max_tokens)
    feature_ds = dataset.map(lambda x, y: x[name])
    lookup.adapt(feature_ds)

    return lookup




def get_normalization_layer(name, dataset):
    """Generates a Normalization layer for a feature.

    Args:
        name (str): Name of the feature.
        dataset (tf.data.Dataset): Dataset that the feature belongs to.

    Returns:
        A Normalization layer.
    """
    normalizer = Normalization()
    feature_ds = dataset.map(lambda x, y: x[name])
    normalizer.adapt(feature_ds)

    return normalizer