import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


class LLM4Rec(nn.Module):
    def __init__(self,
                 base_model,
                 input_dim,
                 output_dim,
                 lora_r,
                 lora_alpha,
                 lora_dropout,
                 lora_target_modules,
                 input_embeds
                 ):
        super(LLM4Rec, self).__init__()
        self.input_dim, self.output_dim = input_dim, output_dim

        print(f'Initializing language decoder ...')

        # add the lora module
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias='none',
        )

        self.model = AutoModel.from_pretrained(base_model, trust_remote_code=True)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        self.model.config.use_cache = False

        self.input_embeds = nn.Embedding.from_pretrained(input_embeds, freeze=True)
        self.input_proj = nn.Linear(self.input_dim, self.model.config.hidden_size)
        self.output_proj = nn.Linear(self.model.config.hidden_size, 64)
        self.score = nn.Linear(64, self.output_dim, bias=False)
        print('Language decoder initialized.')

    def forward(self, inputs, inputs_mask, instruct_ids, instruct_mask, response_ids, response_mask):
        bs = inputs.shape[0]

        instruct_embeds = self.model.embedding(instruct_ids).expand(-1, bs, -1)
        response_embeds = self.model.embedding(response_ids).expand(-1, bs, -1)
        instruct_mask = instruct_mask.expand(bs, -1)
        response_mask = response_mask.expand(bs, -1)
        _, bs, dim = instruct_embeds.shape

        inputs = self.input_proj(self.input_embeds(inputs))
        inputs = inputs.transpose(0, 1)
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=0)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)

        outputs = self.model(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state.transpose(0, 1).contiguous()
        pooled_output = last_hidden_state[:, -1]
        hidden_logits = self.output_proj(pooled_output)
        pooled_logits = self.score(hidden_logits).view(-1, self.output_dim)

        return hidden_logits, pooled_logits
