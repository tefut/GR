import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModel


class SimCIE(nn.Module):
    def __init__(self,
                 base_model,
                 input_dim,
                 output_dim,
                 item_embeds,
                 lora_config,
                 embedding_mode,
                 save_path=None
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

        if save_path is not None:
            _model = AutoModel.from_pretrained(base_model, trust_remote_code=True)
            self.model = PeftModel.from_pretrained(_model, save_path)
        else:
            self.model = AutoModel.from_pretrained(base_model, trust_remote_code=True)
            self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        self.model.config.use_cache = False

        self.item_embeddings = nn.Embedding.from_pretrained(item_embeds)
        self.input_proj = nn.Linear(self.input_dim, self.model.config.hidden_size)
        self.output_proj = nn.Linear(self.model.config.hidden_size, self.output_dim)
        print('Language decoder initialized.')

    def forward(self, input_ids, input_mask, item_ids):
        input_embeds = self.model.embedding(input_ids)

        item_embed = self.input_proj(self.item_embeddings(item_ids))
        item_embed = item_embed.unsqueeze(0)

        inputs = torch.cat([input_embeds, item_embed], dim=0).to(input_embeds.dtype)

        outputs = self.model(inputs_embeds=inputs, attention_mask=input_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state.transpose(0, 1).contiguous()

        last_token_embedding = last_hidden_state[:, -1]

        input_mask_expanded = input_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
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

    def unsup_loss(self, batch_emb, temp=0.05):
        #
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

        loss = F.cross_entropy(sim_score, y_true)
        return torch.mean(loss)

    def sup_loss(self, batch_emb, lamda=0.05):
        batch_size = batch_emb.size(0)
        device = batch_emb.device

        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        sim_score = F.cosine_similarity(norm_emb.unsqueeze(0), norm_emb.unsqueeze(1), dim=2)
        row = torch.arange(0, batch_size, 3)
        col = torch.arange(0, batch_size)
        col = col[col % 3 != 0]

        sim_score = sim_score[row, :]
        sim_score = sim_score[:, col]
        sim_score = sim_score / lamda

        y_true = torch.arange(0, len(col), 2, device=device)
        loss = F.cross_entropy(sim_score, y_true)
        return loss


class GameLLM(SimCIE):
    def __init__(self,
                 base_model,
                 input_dim,
                 output_dim,
                 item_embeds,
                 lora_config,
                 save_path=None
                 ):
        super(GameLLM, self).__init__(base_model,
                                      input_dim,
                                      output_dim,
                                      item_embeds,
                                      lora_config,
                                      save_path)

    def forward(self, input_ids, input_mask, item_ids):
        input_embeds = self.model.embedding(input_ids)

        item_embed = self.input_proj(self.item_embeddings(item_ids))
        item_embed = item_embed.unsqueeze(0)

        inputs = torch.cat([input_embeds, item_embed], dim=0).to(input_embeds.dtype)

        outputs = self.model(inputs_embeds=inputs, attention_mask=input_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state.transpose(0, 1).contiguous()

        pooled_output = last_hidden_state[:, -1]
        pooled_output = self.output_proj(pooled_output)

        return pooled_output
