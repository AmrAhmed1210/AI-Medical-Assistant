# ru.py
import pandas as pd
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
import arabic_reshaper
from bidi.algorithm import get_display

def fix_arabic(text_list):
    return [get_display(arabic_reshaper.reshape(label)) for label in text_list]

model_path = "../models"
test_path = "../data/Balanced/Balanced/Test.xlsx"

df_test = pd.read_excel(test_path).sample(n=200, random_state=42)
df_test = df_test.rename(columns={'q_body': 'text', 'category': 'specialty'})

specialties = sorted(df_test['specialty'].unique().tolist())
label2id = {label: i for i, label in enumerate(specialties)}
df_test['label'] = df_test['specialty'].map(label2id)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def tokenize_fn(x):
    return tokenizer(x["text"], padding="max_length", truncation=True)

tokenized_test = Dataset.from_pandas(df_test).map(tokenize_fn, batched=True)

trainer = Trainer(model=model, data_collator=DataCollatorWithPadding(tokenizer))
print("Predicting results...")
output = trainer.predict(tokenized_test)
y_preds = np.argmax(output.predictions, axis=-1)
y_true = output.label_ids

cm = confusion_matrix(y_true, y_preds)

plt.figure(figsize=(20, 15))

fixed_labels = fix_arabic(specialties)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=fixed_labels,
            yticklabels=fixed_labels,
            annot_kws={"size": 12})

plt.title(get_display(arabic_reshaper.reshape("نتائج تصنيف التخصصات الطبية")), fontsize=22)
plt.xlabel(get_display(arabic_reshaper.reshape("التخصص المتوقع")), fontsize=16)
plt.ylabel(get_display(arabic_reshaper.reshape("التخصص الحقيقي")), fontsize=16)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.savefig("confusion_matrix_fixed.png", bbox_inches='tight', dpi=300)
print("Done! The المحدد matrix is saved as 'confusion_matrix_fixed.png'.")