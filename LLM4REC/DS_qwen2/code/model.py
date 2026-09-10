import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


class SimCIE(nn.Module):
    def __init__(self,
                 base_model,
                 input_dim,
                 output_dim,
                 item_embeds,
                 lora_config,
                 embedding_mode
                 ):
        super(SimCIE, self).__init__()
        self.input_dim, self.output_dim = input_dim, output_dim

        print(f'Initializing language decoder ...')

        # add the lora module
        lora_r, lora_alpha, lora_dropout, lora_target_modules = lora_config
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias='none',
        )

        self.embedding_mode = embedding_mode

        self.model = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        self.model.config.use_cache = False

        self.item_embeddings = nn.Embedding.from_pretrained(item_embeds).to(torch.bfloat16)
        self.input_proj = nn.Linear(self.input_dim, self.model.config.hidden_size).to(torch.bfloat16)
        self.output_proj = nn.Linear(self.model.config.hidden_size, self.output_dim).to(torch.bfloat16)
        self.loss_fn = nn.CrossEntropyLoss()
        print('Language decoder initialized.')

    def forward(self, input_ids, input_mask, item_ids):
        input_embeds = self.model.embed_tokens(input_ids)

        item_embed = self.input_proj(self.item_embeddings(item_ids))
        item_embed = item_embed.unsqueeze(0).transpose(0, 1).contiguous()

        inputs = torch.cat([input_embeds, item_embed], dim=1).to(input_embeds.dtype)

        outputs = self.model(inputs_embeds=inputs, attention_mask=input_mask, return_dict=True)

        last_hidden_state = outputs.last_hidden_state

        last_token_embedding = last_hidden_state[:, -1]

        input_mask_expanded = input_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(torch.bfloat16)
        if self.embedding_mode == 'mean_pooling':
            sum_embeddings = torch.sum((last_hidden_state * input_mask_expanded)[:, :-1], 1)  # 求和
            sum_mask = input_mask_expanded[:, :-1].sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_pooling_embeddings = sum_embeddings / sum_mask  # 求平均
            output_embeddings = last_token_embedding + mean_pooling_embeddings
        elif self.embedding_mode == 'max_pooling':
            last_hidden_state[input_mask_expanded == 0] = -1e4  # Set padding tokens to large negative value
            max_pooling_embeddings = torch.max(last_hidden_state[:, :-1], 1)[0]
            output_embeddings = last_token_embedding + max_pooling_embeddings
        else:
            output_embeddings = last_token_embedding

        pooled_output = self.output_proj(output_embeddings)

        return pooled_output

    def loss(self, batch_emb, temp=0.05):
        batch_size = batch_emb.size(0)
        device = batch_emb.device

        y_true = torch.cat([torch.arange(1, batch_size, step=2, dtype=torch.long).unsqueeze(1),
                            torch.arange(0, batch_size, step=2, dtype=torch.long).unsqueeze(1)],
                           dim=1).reshape([batch_size, ])
        y_true = y_true.to(device)

        # 计算score和loss
        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        sim_score = F.cosine_similarity(norm_emb.unsqueeze(0), norm_emb.unsqueeze(1), dim=2)

        sim_score = sim_score - torch.eye(batch_size, device=device) * 1e12
        sim_score = sim_score / temp

        loss = self.loss_fn(sim_score, y_true)
        return torch.mean(loss)
