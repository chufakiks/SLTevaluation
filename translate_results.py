#!/usr/bin/env python3
"""Translate German translations in results.csv to English using Helsinki-NLP."""

import argparse
import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

def translate_batch(texts, model, tokenizer, device="cpu"):
    """Translate a batch of texts."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Translate results to English")
    parser.add_argument("--input", type=str, default="spamores/results.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="spamores/results_en.csv", help="Output CSV file")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for translation")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    print("Loading Helsinki-NLP German→English model...")
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model = model.to(args.device)
    model.eval()

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input)

    translations = df["translation"].fillna("").tolist()
    english_translations = []

    print(f"Translating {len(translations)} entries...")
    for i in tqdm(range(0, len(translations), args.batch_size)):
        batch = translations[i:i + args.batch_size]
        translated = translate_batch(batch, model, tokenizer, args.device)
        english_translations.extend(translated)

    df["translation_en"] = english_translations
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
