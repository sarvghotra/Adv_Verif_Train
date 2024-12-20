from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification

original_model = AutoModelForCausalLM.from_pretrained("/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B")
tokenizer = AutoTokenizer.from_pretrained("/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B")

from transformers import AutoConfig
loaded_config = AutoConfig.from_pretrained("/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B_seq_clf/config.json")

num_labels = 1  # Adjust this based on your classification task
new_model_f_config = AutoModelForSequenceClassification.from_config(
    loaded_config
)


# "/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/ckpts/itr0_baseline_onlypos_mistral_1e-5/checkpoint-210",

new_model_f_config.base_model.load_state_dict(original_model.base_model.state_dict())
new_model_f_config.score.weight.data.normal_(mean=0.0, std=new_model_f_config.config.initializer_range)

new_model_f_config.save_pretrained("/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B_seq_clf")
tokenizer.save_pretrained("/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B_seq_clf")
