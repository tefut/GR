import json
import pickle

from transformers import (
    Trainer,
    TrainingArguments,
    AdamW,
)

from dataset import CLDataset, CLCollator
from model import SimCIE


def load_dataset(dataset_args):
    '''加载训练数据集和验证数据集'''
    max_length = dataset_args.get("max_length", 128)
    base_model = dataset_args.get("pretrained_model_path")
    cache_dir = dataset_args.get("cache_dir")
    train_file = dataset_args.get("train_data_path", "")
    eval_file = dataset_args.get("val_data_path", "")

    train_dataset = CLDataset(train_file)
    eval_dataset = CLDataset(eval_file)

    data_collator = CLCollator(base_model, cache_dir, max_length)
    return train_dataset, eval_dataset, data_collator


def load_model(model_args):
    '''
    模型加载
    :param model_args:
    :return: torch.nn.module
    '''
    input_dim = model_args.get("input_dim", 64)
    output_dim = model_args.get("output_dim", 32)
    embedding_mode = model_args.get("embedding_mode", "last_token_embedding")  # pooled_output,max_pooling,mean_pooling
    base_model = model_args.get("pretrained_model_path")
    item_embedding_file = model_args.get("item_embedding_file", "")
    item_embeds = pickle.load(open(item_embedding_file, 'rb'))

    lora_r = model_args.get("lora_r", 16)
    lora_alpha = model_args.get("lora_alpha", 16)
    lora_dropout = model_args.get("lora_dropout", 0.05)
    lora_target_modules = model_args.get("lora_target_modules", ['query_key_value', 'dense_h_to_4h', 'dense_4h_to_h'])
    lora_config = [lora_r, lora_alpha, lora_dropout, lora_target_modules]
    model = SimCIE(
        base_model=base_model,
        input_dim=input_dim,
        output_dim=output_dim,
        item_embeds=item_embeds,
        lora_config=lora_config,
        embedding_mode=embedding_mode
    )
    return model


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = model.module.loss(outputs)

        return (loss, outputs) if return_outputs else loss


def main():
    train_config_file = "/opt/huawei/schedule-train/algorithm/train.config"
    with open(train_config_file, 'r', encoding='utf-8') as fin:
        train_config = json.load(fin)
    print(train_config)

    dataset_configs = train_config.get("dataset_configs")
    model_configs = train_config.get("model_configs")
    training_configs = train_config.get("training_configs")

    train_dataset, eval_dataset, data_collator = load_dataset(dataset_configs)
    model = load_model(model_configs)
    training_args = TrainingArguments(**training_configs)

    for name, param in model.named_parameters():
        if "lora" in name or "input_proj" in name or "output_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(parameters,
                      lr=0.00003,
                      betas=(0.8, 0.999), weight_decay=3e-7)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )
    trainer.train()


if __name__ == '__main__':
    main()
