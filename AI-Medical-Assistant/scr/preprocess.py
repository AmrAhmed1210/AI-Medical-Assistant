# preprocess.py
import pandas as pd
from datasets import Dataset

def load_data(train_path, test_path):
    # Load Excel files
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    # Rename columns
    train_df = train_df.rename(columns={'q_body': 'question', 'category': 'specialty'})
    test_df = test_df.rename(columns={'q_body': 'question', 'category': 'specialty'})

    # Numeric labels
    train_df['label'] = train_df['specialty'].astype('category').cat.codes
    test_df['label'] = test_df['specialty'].astype('category').cat.codes

    # Label mappings
    specialties = train_df['specialty'].astype('category').cat.categories.tolist()
    id2label = {i: label for i, label in enumerate(specialties)}
    label2id = {label: i for i, label in enumerate(specialties)}

    # Convert to HuggingFace datasets
    train_ds = Dataset.from_pandas(train_df[['question', 'label']])
    test_ds = Dataset.from_pandas(test_df[['question', 'label']])

    print(f"Loaded {len(specialties)} specialties: {specialties}")
    return train_ds, test_ds, id2label, label2id, specialties
