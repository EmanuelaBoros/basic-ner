# Named Entity Recognition (NER) with HIPE-2020 (French)

This notebook demonstrates how to train and evaluate a **Named Entity Recognition (NER)** model on the [**HIPE-2020** dataset](https://github.com/hipe-eval/HIPE-2022-data) using a multilingual BERT model adapted for historical texts.

## Overview

The HIPE-2020 dataset contains historical newspaper text annotated with multiple layers of named entities:
- `NE-COARSE-LIT`
- `NE-COARSE-METO`
- `NE-FINE-LIT`
- `NE-FINE-METO`
- `NE-FINE-COMP`
- `NE-NESTED`

For simplicity, this notebook focuses on **`NE-COARSE-LIT`**, which provides coarse-grained entity categories such as `PER`, `LOC`, and `ORG`.

## Main Steps

1. **Load data files**  
   The dataset files (`train`, `dev`, `test`) are loaded from the HIPE-2022 directory structure.

2. **Initialize the tokenizer**  
   Uses the multilingual model [`dbmdz/bert-base-historic-multilingual-cased`](https://huggingface.co/dbmdz/bert-base-historic-multilingual-cased).

3. **Create or load a label map**  
   - When run for the first time, the notebook builds a `label_map.json` file from the training data.  
   - On later runs, the same map is loaded to ensure **label-ID consistency** across datasets and inference.

4. **Inspect data samples**  
   Decodes tokens and labels to verify correct alignment between tokenized text and entity annotations.

5. **Train the model**  
   A token classification model is trained on `NE-COARSE-LIT` using the specified maximum sequence length and optimizer settings.

6. **Save the model and tokenizer**  
   The trained model and tokenizer are stored in the `experiments/` folder for later evaluation or inference.

## Example Output

After preprocessing, an example token-label pair looks like:

```
Token           NE-COARSE-LIT
------------------------------
Barack          B-pers
Obama           I-pers
was             O
born            O
in              O
Hawaï           B-pers
```

## Folder Structure

```
experiments/
 ├── label_map.json
 ├── model/
 │   ├── config.json
 │   ├── pytorch_model.bin
 │   └── tokenizer.json
```

## Notes

- The label map is crucial for ensuring that label IDs remain stable across training, validation, and inference.
- The code can be extended to support all HIPE annotation layers for multi-task learning.

---

**Author:** [Your Name]  
**Project:** Historical NER with HIPE-2020  
**Model:** `dbmdz/bert-base-historic-multilingual-cased`
