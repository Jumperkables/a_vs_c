__author__ = "Jumperkables"
# Standard
import os, sys

# Deep learning
import torch
from torch import nn
import torch.nn.functional as F
import hopfield_layers.modules as hpf
from transformers import BertModel, BertForMultipleChoice, BertForQuestionAnswering, LxmertModel, LxmertForQuestionAnswering, LxmertConfig

#class Hopfield(Module):
#    """
#    Module with underlying Hopfield association.
#    """
#
#    def __init__(self,
#                 input_size: Optional[int] = None,
#                 hidden_size: Optional[int] = None,
#                 output_size: Optional[int] = None,
#                 pattern_size: Optional[int] = None,
#                 num_heads: int = 1,
#                 scaling: Optional[Union[float, Tensor]] = None,
#                 update_steps_max: Optional[Union[int, Tensor]] = 0,
#                 update_steps_eps: Union[float, Tensor] = 1e-4,
#
#                 normalize_stored_pattern: bool = True,
#                 normalize_stored_pattern_affine: bool = True,
#                 normalize_state_pattern: bool = True,
#                 normalize_state_pattern_affine: bool = True,
#                 normalize_pattern_projection: bool = True,
#                 normalize_pattern_projection_affine: bool = True,
#                 normalize_hopfield_space: bool = False,
#                 normalize_hopfield_space_affine: bool = False,
#                 stored_pattern_as_static: bool = False,
#                 state_pattern_as_static: bool = False,
#                 pattern_projection_as_static: bool = False,
#                 pattern_projection_as_connected: bool = False,
#                 stored_pattern_size: Optional[int] = None,
#                 pattern_projection_size: Optional[int] = None,
#
#                 batch_first: bool = True,
#                 association_activation: Optional[str] = None,
#                 dropout: float = 0.0,
#                 input_bias: bool = True,
#                 concat_bias_pattern: bool = False,
#                 add_zero_association: bool = False,
#                 disable_out_projection: bool = False
#                 ):
#        """
#        Initialise new instance of a Hopfield module.
#        :param input_size: depth of the input (state pattern)
#        :param hidden_size: depth of the association space
#        :param output_size: depth of the output projection
#        :param pattern_size: depth of patterns to be selected
#        :param num_heads: amount of parallel association heads
#        :param scaling: scaling of association heads, often represented as beta (one entry per head)
#        :param update_steps_max: maximum count of association update steps (None equals to infinity)
#        :param update_steps_eps: minimum difference threshold between two consecutive association update steps
#        :param normalize_stored_pattern: apply normalization on stored patterns
#        :param normalize_stored_pattern_affine: additionally enable affine normalization of stored patterns
#        :param normalize_state_pattern: apply normalization on state patterns
#        :param normalize_state_pattern_affine: additionally enable affine normalization of state patterns
#        :param normalize_pattern_projection: apply normalization on the pattern projection
#        :param normalize_pattern_projection_affine: additionally enable affine normalization of pattern projection
#        :param normalize_hopfield_space: enable normalization of patterns in the Hopfield space
#        :param normalize_hopfield_space_affine: additionally enable affine normalization of patterns in Hopfield space
#        :param stored_pattern_as_static: interpret specified stored patterns as being static
#        :param state_pattern_as_static: interpret specified state patterns as being static
#        :param pattern_projection_as_static: interpret specified pattern projections as being static
#        :param pattern_projection_as_connected: connect pattern projection with stored pattern
#        :param stored_pattern_size: depth of input (stored pattern)
#        :param pattern_projection_size: depth of input (pattern projection)
#        :param batch_first: flag for specifying if the first dimension of data fed to "forward" reflects the batch size
#        :param association_activation: additional activation to be applied on the result of the Hopfield association
#        :param dropout: dropout probability applied on the association matrix
#        :param input_bias: bias to be added to input (state and stored pattern as well as pattern projection)
#        :param concat_bias_pattern: bias to be concatenated to stored pattern as well as pattern projection
#        :param add_zero_association: add a new batch of zeros to stored pattern as well as pattern projection
#        :param disable_out_projection: disable output projection
#        """

# To be sorted
class Assoc_vs_Ctgrcl(nn.Module):
    def __init__(self):
        # Associative Stream
        pass
        # Categorical Stream

    def forward(self, x):
        # Associative Stream

        # Categorical Stream

        return x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--purpose", type=str, default="test_hopfield", choices=["test_hopfield"])
    args = parser.parse_args()
    if args.purpose == "test_hopfield":
        hopfield = Hopfield(
            scaling=1.0,
        
            # do not project layer input
            state_pattern_as_static=True,
            stored_pattern_as_static=True,
            pattern_projection_as_static=True,
        
            # do not pre-process layer input
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
        
            # do not post-process layer output
            disable_out_projection=True
        )


