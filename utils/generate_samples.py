import torch
from transformers import GenerationConfig
from torch import autocast


class SampleLLMPolicy:
    def __init__(self,
                 model,
                 tokenizer,
                 do_sample,
                 temperature,
                 max_new_tokens,
                 return_seq_num,
                 gen_config=None):
        
        self.model = model
        self.tokenizer = tokenizer
        # stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", \
        #                 "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
        self.stop_tokens = ["Problem:"]
        def_gen_config = {
            "do_sample": True,
            "temperature": 1.0,
            "max_new_tokens": 512,
            "num_return_sequences": return_seq_num,
        }
        def_gen_config.update(gen_config)
        self.generation_config = GenerationConfig(**def_gen_config)
        # self.generation_config = GenerationConfig(
        #     do_sample=True,
        #     temperature=1.0,
        #     # top_p=0.95,
        #     max_new_tokens=512,
        #     num_return_sequences=return_seq_num,
            # temperature=0.8 if args.diverse_beam > 1 else 1.0,
            # num_beam_groups=args.diverse_beam,
        #     # diversity_penalty=1.0,
        #     # num_beams=return_seq_num,
        # )

    def sample(self, batch):
        with autocast("cuda", dtype=torch.bfloat16):
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            with torch.no_grad():
                generation_output = self.model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    stop_strings=self.stop_tokens,
                    tokenizer=self.tokenizer,
                    # synced_gpus=True,
                )
                        
            s = generation_output.sequences
            
            outputs_string = self.tokenizer.batch_decode(s, skip_special_tokens=True)
            inputs_string = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        return inputs_string, outputs_string
