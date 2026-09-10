from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import GenerationConfig, AutoTokenizer, AutoModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 20005] = 5e4
        return scores


class LLMKG():
    def __init__(self,
                 model_path: Union[str, Path],
                 instruction_formatter: str = None,
                 generation_config: GenerationConfig = None,
                 max_length=128):
        self.max_length = max_length
        self.name = 'llm'
        self.model_path = Path(model_path)
        self.generation_config = generation_config
        self.instruction_formatter = instruction_formatter
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device('cpu')
        self.prepare_model()

    def prepare_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()

    def generate_knowledge(self, instructions: List[str], batch_size: int = 1):
        raise NotImplementedError

    def encode_knowledge(self, knowledge, aggre_type: str = 'avg'):
        x = self.tokenizer(knowledge, padding=True, truncation=True, return_tensors="pt",
                           return_attention_mask=True, max_length=self.max_length).to(self.device)
        mask = x['attention_mask']
        outputs = self.model(**x, output_hidden_states=True, return_dict=True)
        pred = self.get_paragraph_representation(outputs, mask, aggre_type).tolist()
        return pred

    def get_paragraph_representation(self, outputs, mask, pooler='cls', dim=1):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        # Apply different poolers

        if pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return outputs.pooler_output.cpu()  # chatglm不能用，用于bert
        elif pooler == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()
        elif pooler == "avg":
            return ((last_hidden * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)).cpu()
        elif pooler == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * mask.unsqueeze(-1)).sum(dim) / mask.sum(
                dim).unsqueeze(-1)
            return pooled_result.cpu()
        elif pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * mask.unsqueeze(-1)).sum(dim) / mask.sum(
                dim).unsqueeze(-1)
            return pooled_result.cpu()
        elif pooler == 'len_last':  # 根据padding方式last方式也不一样
            lens = mask.unsqueeze(-1).sum(dim)
            pooled_result = [last_hidden[i, lens[i] - 1, :] for i in range(last_hidden.shape[0])]
            pooled_result = torch.concat(pooled_result, dim=0)
            return pooled_result.cpu()
        elif pooler == 'last':
            if dim == 0:
                return last_hidden[-1, :, :]
            else:
                return last_hidden[:, -1, :]
        elif pooler == 'wavg':
            # Get weights of shape [bs, seq_len, hid_dim]
            weights = (
                torch.arange(start=1, end=last_hidden.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(last_hidden.size())
                    .float().to(last_hidden.device)
            )

            # Get attn mask of shape [bs, seq_len, hid_dim]
            input_mask_expanded = (
                mask
                    .unsqueeze(-1)
                    .expand(last_hidden.size())
                    .float()
            )

            # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded * weights, dim=dim)
            sum_mask = torch.sum(input_mask_expanded * weights, dim=dim)

            pooled_result = sum_embeddings / sum_mask
            return pooled_result.cpu()
        else:
            raise NotImplementedError


class ChatGLMKG(LLMKG):
    def prepare_model(self):
        self.name = 'chatglm'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()
        self.model.eval()

        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": 1, "do_sample": False, "top_p": 0.7,
                           "temperature": 0.95, "logits_processor": logits_processor}

    def generate_knowledge(self, instructions: List[str], batch_size: int = 1):
        if batch_size > 1:
            responses = self.parallel_generate(instructions, batch_size)
        else:
            responses = []
            total_step = 100
            instruct_per_step = len(instructions) // batch_size // total_step
            print(f'total steps: {total_step}, instructions per step: {instruct_per_step}')
            for i, instruction in enumerate(instructions):
                response, history = self.model.chat(self.tokenizer, instruction, max_length=self.max_length)
                responses.append(response)
                if i % instruct_per_step == 0:
                    print(f'step {i}, {i // instruct_per_step}/{total_step}')
        return responses

    def batch_generate(self, data: List[str]):
        pred_list = []
        torch.cuda.empty_cache()
        x = self.tokenizer(data, padding=True, return_tensors="pt", truncation=True).to(self.device)
        preds = self.model.generate(**x, **self.gen_kwargs)
        for pred in preds.tolist():
            decode = self.tokenizer.decode(pred)
            pred_list.append(decode)
        return pred_list

    def parallel_generate(self, instructions: List[str], batch_size: int = 1):
        pred_list = []
        self.model.eval()
        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": self.max_length, "num_beams": 1, "do_sample": False, "top_p": 0.7,
                      "temperature": 0.95, "logits_processor": logits_processor}
        dataloader = DataLoader(instructions, batch_size, shuffle=False)
        total_step = 100
        instruct_per_step = len(instructions) // batch_size // total_step
        print(f'total steps: {total_step}, instructions per step: {instruct_per_step}')
        with torch.no_grad():
            for i, x in enumerate(dataloader):
                torch.cuda.empty_cache()
                x = self.tokenizer(x, padding=True, return_tensors="pt", truncation=True).to(self.device)
                preds = self.model.generate(**x, **gen_kwargs)
                for pred in preds.tolist():
                    decode = self.tokenizer.decode(pred)
                    pred_list.append(decode)
                if i % instruct_per_step == 0:
                    print(f'step {i}, {i // instruct_per_step}/{total_step}')
        return pred_list

    def encode_knowledge(self, knowledge, aggre_type: str = 'avg'):
        x = self.tokenizer(knowledge, padding=True, truncation=True, return_tensors="pt",
                           return_attention_mask=True, max_length=self.max_length).to(self.device)
        mask = x['attention_mask']
        outputs = self.model.transformer(**x, output_hidden_states=True, return_dict=True)
        outputs.last_hidden_state = outputs.last_hidden_state.transpose(1, 0)
        pred = self.get_paragraph_representation(outputs, mask, aggre_type).tolist()
        return pred


class Mistral(LLMKG):
    def prepare_model(self):
        self.name = 'mistral'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()
        self.model.eval()

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode_knowledge(self, knowledge, aggre_type: str = 'avg'):
        knowledge = [v.replace("[Instruct]", "Instruct:").replace("[Query]", "\nQuery:") for v in knowledge]
        x = self.tokenizer(knowledge, max_length=self.max_length, padding=True, return_attention_mask=True,
                           return_tensors='pt').to(self.device)

        outputs = self.model(**x)
        embeddings = self.last_token_pool(outputs.last_hidden_state, x['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()
