# Transformers Under the Hood

A set of small, hands-on exercises exploring what Hugging Face's `pipeline()` does internally — built while learning the `transformers` library as part of a AI engineering roadmap.

## What's inside

* **Temperature & sampling** — using the same prompt with different temperatures to observe how temperature affects output variety
* **Sentiment analysis: pipeline vs. manual** — breaking down `pipeline("sentiment-analysis")` into tokenizer → model → logits → softmax → prediction
* **Zero-shot classification** — classifying text using candidate labels provided at inference time
* **Model heads** — comparing `AutoModel` (contextual representations) with `AutoModelForSequenceClassification` (class logits)
* **Manual next-token prediction** — looking inside text generation: logits → softmax → top-5 candidate next tokens
* **Padding, attention masks, and decoding** — batching sequences of different lengths and converting token IDs back into text

## Stack

`transformers`, `torch`

## Run it

```bash
pip install transformers torch
```

Then open `transformers_under_the_hood.ipynb`.
