import torch
import json
import stat
import os


def text_to_token_ids(text, tokenizer):
    text_encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(text_encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    token_ids_flat = token_ids.squeeze(0)
    return tokenizer.decode(token_ids_flat.tolist())


def load_file(data_path):
    propmt_list = []
    with open(data_path, encoding='utf-8') as rf:
        respose_json = json.load(rf)
        for item_info in respose_json["游戏信息"]:
            propmt_list.append(item_info)
    return propmt_list


def write_to_file(save_file, mode='w'):
    flags = os.O_WRONLY | os.O_CREAT
    stats = stat.S_IWUSR | stat.S_IRUSR
    file_hander = os.fdopen(os.open(save_file, flags, stats), mode)
    return file_hander


def build_prompt(text):
    instruction = "请根据以下对游戏的描述，包括游戏名称，游戏开发者，游戏介绍及游戏评价等信息，生成该游戏的标签，标签个数不超过10个."
    item_name = text['游戏名称'].strip()
    item_dev = text['游戏开发者'].strip()
    item_info = text['游戏介绍'].strip()
    item_comm = text['游戏评价'].strip()
    item_id = text['游戏ID'].strip()
    input_text = f"游戏名称:{item_name},游戏开发者:{item_dev},游戏介绍:{item_info},游戏评价:{item_comm}"
    prompt = f"{instruction}内容信息:{input_text},该游戏的向量表征为:'[EMB]',该游戏的标签为{text['游戏标签'].strip()}"
    generate_input = f"{instruction}内容信息:{input_text},该游戏的向量表征为:'[EMB]',该游戏的标签为"
    emb_start_index = prompt.find('[EMB]')
    emb_end_index = emb_start_index + len('[EMB]')
    prompt_head = prompt[:emb_start_index]
    prompt_emb = "[EMB]"
    prompt_tail = prompt[emb_end_index:]

    return prompt, generate_input, item_id, [prompt_head, prompt_emb, prompt_tail, emb_start_index]


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    output = model(input_batch)
    logits = output.logits.flatten(0, 1)
    loss = torch.nn.functional.cross_entropy(logits, target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
