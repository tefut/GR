import time
from pathlib import Path
from typing import Union, List

import torch
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
    def __init__(self, model_path: Union[str, Path],
                 instruction_formatter: str = None,
                 generation_config: GenerationConfig = None):
        self.name = 'llm'
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.generation_config = generation_config
        self.instruction_formatter = instruction_formatter
        self.prepare_model()

    def prepare_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()

    def encode_knowledge(self, knowledge: List[str], batch_size: int = 1, aggre_type: str = 'avg'):
        encodings = []
        self.model.eval()
        dataloader = DataLoader(knowledge, batch_size, shuffle=False)
        with torch.no_grad():
            total_step = 10
            instruct_per_step = len(knowledge) // batch_size // total_step + 1
            for i, x in enumerate(dataloader):
                x = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt",
                                   return_attention_mask=True, max_length=512).to(self.device)
                mask = x['attention_mask']
                outputs = self.model(**x, output_hidden_states=True, return_dict=True)
                pred = self.get_paragraph_representation(outputs, mask, aggre_type)
                encodings.extend(pred.tolist())
                if i % instruct_per_step == 0:
                    print(f'step {i}, {i // instruct_per_step}/{total_step}')
        return encodings

    def generate_knowledge(self, instructions: List[str], batch_size: int = 1):
        raise NotImplementedError

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


# @LLMKG.register("chatglm")
class chatGLMKG(LLMKG):
    def prepare_model(self):
        self.name = 'chatglm'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True,
                                                       encode_special_tokens=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).quantize(8).cuda()

    def parallel_generate(self, instructions: List[str], batch_size: int = 1):
        # chatglm-v3
        pred_list = []
        self.model.eval()
        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.get_command("<|user|>"),
            self.tokenizer.get_command("<|assistant|>"),
        ]
        gen_kwargs = {"max_new_tokens": 512, "num_beams": 1, "do_sample": True, "top_p": 0.9,
                      "temperature": 0.1, "logits_processor": logits_processor, "eos_token_id": eos_token_id}
        dataloader = DataLoader(instructions, batch_size, shuffle=False)
        total_step = 100
        instruct_per_step = len(instructions) // batch_size // total_step + 1
        print(f'total steps: {total_step}, instructions per step: {instruct_per_step}')
        parallel_generate_begin_time = time.time()
        with torch.no_grad():
            for i, prompts in enumerate(dataloader):
                torch.cuda.empty_cache()
                batched_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", truncation=True,
                                                max_length=1024).to(self.device)
                batched_outputs = self.model.generate(**batched_inputs, **gen_kwargs)
                for _, output_ids in zip(batched_inputs.input_ids, batched_outputs):
                    decoded_text = self.tokenizer.decode(output_ids)
                    pred_list.append(decoded_text.strip())
                if i % instruct_per_step == 0:
                    print("--------------------------------------------------------------------------")
                    print(f'step {i}, {i // instruct_per_step}/{total_step}, '
                          f'batched_inputs:{len(batched_inputs.input_ids[0])}, '
                          f'batched_outputs:{len(batched_outputs[0])},'
                          f'cost time:{time.time() - parallel_generate_begin_time}')
                if i % (instruct_per_step * 5) == 0:
                    print(f"example:{pred_list[-1]}")
        return pred_list

    def generate_knowledge(self, instructions: List[str], batch_size: int = 1):
        if batch_size > 1:
            responses = self.parallel_generate(instructions, batch_size)
        else:
            responses = []
            total_step = 100
            instruct_per_step = len(instructions) // batch_size // total_step + 1
            print(f'total steps: {total_step}, instructions per step: {instruct_per_step}')
            for i, instruction in enumerate(instructions):
                response, history = self.model.chat(self.tokenizer, instruction, max_length=1024)
                responses.append(response)
                if i % instruct_per_step == 0:
                    print(f'step {i}, {i // instruct_per_step}/{total_step}')
        return responses

    def encode_knowledge(self, knowledge: List[str], batch_size: int = 1, aggre_type: str = 'avg'):
        encodings = []
        self.model.eval()
        dataloader = DataLoader(knowledge, batch_size, shuffle=False)
        with torch.no_grad():
            total_step = 100
            instruct_per_step = len(knowledge) // batch_size // total_step + 1
            for i, x in enumerate(dataloader):
                x = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt",
                                   return_attention_mask=True, max_length=1024).to(self.device)
                mask = x['attention_mask']
                outputs = self.model.transformer(**x, output_hidden_states=True, return_dict=True)

                outputs.last_hidden_state = outputs.last_hidden_state.transpose(1, 0)
                pred = self.get_paragraph_representation(outputs, mask, aggre_type)
                encodings.extend(pred.tolist())
                if i % instruct_per_step == 0:
                    print(f'step {i}, {i // instruct_per_step}/{total_step}')
        return encodings
