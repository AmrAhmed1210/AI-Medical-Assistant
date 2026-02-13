# inference.py
from transformers import pipeline

# Model path relative to 'scr' folder
model_path = "../models"

pipe = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    top_k=3
)

texts = [
    "حاسس بزغللة في عيني وصداع نصفي بقاله يومين",
    "عندي كحة وضيق تنفس من امبارح",
    "ألم شديد في اسفل الظهر مع تنميل في الرجل",
    "طفلي عنده حرارة عالية واسهال"
]

print("Starting Medical Classification...")

for text in texts:
    prediction_output = pipe(text)

    print("=" * 50)
    print(f"Sentence: {text}")

    # Check if output is double-nested [[{...}]] and flatten it if so
    results = prediction_output[0] if isinstance(prediction_output[0], list) else prediction_output

    for res in results:
        print(f"- Specialty: {res['label']} ({res['score']:.2f})")