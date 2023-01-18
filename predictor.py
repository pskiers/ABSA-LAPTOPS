from transformers import BertTokenizer
import torch
from models import ALMBert


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    model = torch.load("./trained/ALMBert_sentiment_classification.pth").to(torch.device("cpu"))

    try:
        while True:
            review = input("Enter review: ")
            aspect = input("Enter aspect: ")

            review_encoding = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=75,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            )
            aspect_encoding = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=4,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            )

            labels = {
                0: "positive",
                1: "negative",
                2: "neutral",
            }
            with torch.no_grad():
                preds = model(
                    review_encoding['input_ids'],
                    review_encoding['attention_mask'],
                    aspect_encoding['input_ids'],
                    aspect_encoding['attention_mask'],
                )
            print(preds)
            print(labels[int(torch.argmax(preds))])
    except KeyboardInterrupt:
        print("ending ...")
