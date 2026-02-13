# AI-Project: Smart Medical Diagnostic Assistant 

##  Overview
**AI-Project** is a hybrid smart medical assistant designed to help users get a preliminary diagnosis based on text symptoms and medical images. The system integrates the power of Natural Language Processing (NLP) and Computer Vision to provide a comprehensive diagnostic experience.

##  Key Features
* **Multi-Modal Analysis:** Simultaneously analyzes user text (complaints) and medical images (X-rays or skin conditions).
* **Arabic NLP Support:** Powered by **AraBERT** to understand Arabic dialects and the Egyptian colloquial language with high accuracy.
* **Visual Diagnosis:** A specialized **CNN** model to analyze patterns in medical images and identify disease types.
* **Smart Explanations:** Integration of **Gemini API** to transform dry technical results into human-friendly, easy-to-understand medical responses.
* **Specialized Domain:** The system is hard-coded to focus only on the medical field and will politely decline any non-medical queries.

##  Technical Architecture
The project relies on a robust pipeline connecting several AI models:
1.  **NLU Layer (AraBERT):** Extracts "Intent" and determines the appropriate medical specialty.
2.  **Vision Layer (Custom CNN):** Processes images to extract vital disease indicators.
3.  **Orchestration Layer (FastAPI):** The "Maestro" that manages data flow between models and the frontend.
4.  **Generative Layer (Google Gemini):** Formulates the final response based on "verified doctor responses" from our database.

##  Tech Stack
* **Language:** Python 3.9+
* **Framework:** FastAPI
* **AI Models:** Transformers (Hugging Face), PyTorch/TensorFlow
* **LLM:** Google Gemini Pro API
* **Data Handling:** Pandas & NumPy

