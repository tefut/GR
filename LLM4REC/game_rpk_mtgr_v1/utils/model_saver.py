import os
import logging
import torch


class ModelSaver:
    def __init__(self, save_dir: str, export_save_dir_name: str):
        self.best_auc = 0
        self.save_dir = save_dir
        self.export_save_dir_name = export_save_dir_name
        self.save_path = os.path.join(save_dir, export_save_dir_name)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            logging.info("Create save folder: %s", self.save_path)

    def save_model(self, model, model_file_name: str):
        """
        保存模型
        """

        model_path = os.path.join(self.save_path, model_file_name)
        model.eval()
        logging.info("Saving model to %s", model_path)

        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save(state_dict, model_path)

    def save_metric(self):
        try:
            write_result = os.path.join(self.save_path, 'result.txt')
            with open(write_result, 'w') as result_file:
                result_file.write(f'best_auc:{self.best_auc}')

            logging.info("Save metric to %s", write_result)
        except Exception as e:
            logging.error("Save metric error: %s", e)
