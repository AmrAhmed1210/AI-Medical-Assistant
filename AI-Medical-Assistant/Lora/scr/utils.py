import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ExcelDataset(Dataset):
    def init(self, excel_path, tokenizer_name="Qwen/Qwen-VL-3B", text_column="text"):
        df = pd.read_excel(excel_path)
        self.texts = df[text_column].astype(str).tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encodings = self.tokenizer(self.texts, padding=True, truncation=True, return_tensors="pt")

    def len(self):
        return len(self.texts)

    def getitem(self, idx):
        return self.encodings["input_ids"][idx], self.encodings["attention_mask"][idx]