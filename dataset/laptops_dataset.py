import pandas as pd
import torch
import torch.utils.data as data
from transformers import BertTokenizer


class LaptopReviewDataset(data.Dataset):
  def __init__(self, reviews, aspects, sentiments, tokenizer, max_len_review, max_len_aspect):
    self.reviews = reviews
    self.aspects = aspects
    self.sentiments = sentiments
    self.tokenizer = tokenizer
    self.max_len_review = max_len_review
    self.max_len_aspect = max_len_aspect

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    aspect = str(self.aspects[item])
    sentiment = self.sentiments[item]
    review_encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len_review,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    aspect_encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len_aspect,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return (
      review,
      review_encoding['input_ids'].flatten(),
      review_encoding['attention_mask'].flatten(),
      aspect_encoding['input_ids'].flatten(),
      aspect_encoding['attention_mask'].flatten(),
      torch.tensor(sentiment, dtype=torch.long)
    )


def get_sentiment_classification_dataset (no_labels_number: int):
    dataset = pd.read_csv("./data/dataset.csv")

    just_labels = dataset.loc[dataset.sentiment != "no"]
    no_labels = dataset.loc[dataset.sentiment == "no"]

    new_dataset = pd.concat([no_labels.sample(no_labels_number), just_labels])
    new_dataset["sentiment"].replace(["positive", "negative", "neutral", "no"], [0, 1, 2, 3], inplace=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    full_ds =  LaptopReviewDataset(
        reviews=new_dataset.text.values,
        aspects=new_dataset.aspect.values,
        sentiments=new_dataset.sentiment.values,
        tokenizer=tokenizer,
        max_len_review=75,
        max_len_aspect=4
    )
    train_len, valid_len, test_len = len(full_ds) - 2 * int(len(full_ds) * 0.15), int(len(full_ds) * 0.15), int(len(full_ds) * 0.15)
    return data.random_split(dataset=full_ds, lengths=[train_len, valid_len, test_len], generator=torch.Generator().manual_seed(42))


def get_sentiment_detection_dataset(no_labels_number: int):
    dataset = pd.read_csv("./data/dataset.csv")

    just_labels = dataset.loc[dataset.sentiment != "no"]
    no_labels = dataset.loc[dataset.sentiment == "no"]

    new_dataset = pd.concat([no_labels.sample(no_labels_number), just_labels])
    new_dataset["sentiment"].replace(["positive", "negative", "neutral", "no"], [1, 1, 1, 0], inplace=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    full_ds = LaptopReviewDataset(
        reviews=new_dataset.text.values,
        aspects=new_dataset.aspect.values,
        sentiments=new_dataset.sentiment.values,
        tokenizer=tokenizer,
        max_len_review=75,
        max_len_aspect=4
    )
    train_len, valid_len, test_len = len(full_ds) - 2 * int(len(full_ds) * 0.15), int(len(full_ds) * 0.15), int(len(full_ds) * 0.15)
    return data.random_split(dataset=full_ds, lengths=[train_len, valid_len, test_len], generator=torch.Generator().manual_seed(42))