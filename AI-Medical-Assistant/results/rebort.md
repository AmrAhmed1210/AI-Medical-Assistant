## Training & Dataset Details

### 1. Dataset Source
I utilized the **MAQA Dataset** available on Kaggle, which consists of high-quality Arabic medical questions and answers.
* **Kaggle Dataset:** [MAQA Dataset on Kaggle](https://www.kaggle.com/datasets/manarm7md/maqa-dataset)

### 2. Model Architecture
The project is built using **AraBERT** (specifically `aubmindlab/bert-base-arabertv2`). This model was chosen for its superior ability to understand the complex nuances of the Arabic language in a medical context.

### 3. Notebook & Implementation
You can view the full training process, data cleaning steps, and model evaluation directly in the Colab notebook below:
* **Colab Notebook:** [Medical Classification Notebook](https://colab.research.google.com/drive/1KPnjNq-je86PhHBYNal-jvAJVMtTx9Oy)

---

##  Performance Analysis

After training for **3 Epochs**, the model achieved the following metrics across 20 medical specialties:

* **Final Accuracy:** 73.1%
* **Weighted F1-Score:** 0.729
* **Performance Insight:** Achieving ~73% accuracy on a 20-class classification task is a strong result, especially considering the semantic overlap between various medical fields.

