import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import torch
import time

import datasets
import numpy as np
import copy
import collator
import utils

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback 
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from modeling_llama_seq2seq import LlamaCrossAttentionEncDec

check_min_version("4.38.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


    ## new add
    model_method: str = field(
        default="default", 
        metadata={
            "help": "The default refers to the general seq2seq model, such as t5,bart"
        },
    )
    decoder_layer_num: int = field(default=8)
    run_mode: str = field(default="resume")
    do_sample: bool = field(default=False)
    patience: int = field(default=3)
    encoder_method: str = field(default="causal")

    decoder_param_method: str = field(default="freeze")
    decoder_hidden_size: int = field(default=1024)
    decoder_intermediate_size: int = field(default=2752)
    decoder_num_attention_heads: int = field(default=16)
    decoder_num_key_value_heads: int = field(default=16)
    decoder_model_name_or_path: str = field(default=None)
    encoder_layer_num: int = field(default=8)

def load_model():
    parser = HfArgumentParser((ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code
    )

    if model_args.model_method == "default":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    
    elif model_args.model_method == "lamate":
        # stage 1
        if model_args.run_mode == "init":
            # seting decoder config
            decoder_config = copy.deepcopy(config.to_dict())
            decoder_config["num_hidden_layers"] = model_args.decoder_layer_num
            decoder_config["num_encoder_layers"] = config.num_hidden_layers
            decoder_config["decoder_param_method"] = model_args.decoder_param_method
            decoder_config["model_method"] = model_args.model_method
            decoder_config["hidden_size"] = model_args.decoder_hidden_size
            decoder_config["intermediate_size"] = model_args.decoder_intermediate_size
            decoder_config["num_attention_heads"] = model_args.decoder_num_attention_heads
            decoder_config["num_key_value_heads"] = model_args.decoder_num_key_value_heads
            config.decoder =  decoder_config
            # set encoder config
            config.use_cache = False
            config.is_encoder_decoder = True
            config.decoder_start_token_id = config.bos_token_id
            config.encoder_method = model_args.encoder_method
            config.encoder_layer_num = model_args.encoder_layer_num
            # make param dict
            state_dict = utils.make_model_state_dict(model_path=model_args.model_name_or_path)
            model = LlamaCrossAttentionEncDec.from_pretrained(None, config=config, state_dict=state_dict, ignore_mismatched_sizes=True)
            model.freeze_llm() # frozen LLM
        # stage 2
        else:
            model = LlamaCrossAttentionEncDec.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        print("Not implement this model yet!")
        exit()

    return model