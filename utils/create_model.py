import transformers
import torch

from transformers import AutoTokenizer
from typing import Dict
from torch.nn.parallel import DistributedDataParallel as DDP


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg

        # FIXME: make sure this is correct if statement
        if model.get_output_embeddings() is not None:
            output_embeddings = model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg



def create_model(model_name, rank):
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left", truncation_side="left")
    # quick fix for the tokenizer.model_max_length issue
    if tokenizer.model_max_length > 1000000000000000019:
        tokenizer.model_max_length = 1024

    # if "llama2" in model_name:
    #     tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_name:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])

    # FIXME: keep it here until we have moved it out everywhere
    # model.eval()

    return model, tokenizer


def create_tokenizer(model_name_or_path, trust_remote_code, model):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        trust_remote_code=trust_remote_code,
        # truncation_side="left"
    )
    # if tokenizer.model_max_length > 1000000000000000019:
    #     tokenizer.model_max_length = 1024

    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    
    if "llama" in model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    
    return tokenizer
