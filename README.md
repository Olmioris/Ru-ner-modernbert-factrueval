# Final Analysis and Interpretation of Experiments

## 1. Environment and Data Preparation

- Installed libraries: `datasets`, `evaluate`, `seqeval`, `transformers`, `accelerate`, `natasha`, `razdel`, `gliner`, `seaborn`.
- Fixed random seeds:
  - `SEED = 42`, then `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`  
    (Fixing seeds ensures **reproducibility** of all experiments.)
- Defined device:
  - `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` — for faster training on GPU (T4).

**Data:**

- Loaded the dataset `gusevsk/factrueval2016` via `load_dataset`.
- Converted into a convenient `DatasetDict` with `train / validation / test` splits.
- Built a list of BIO labels from `ner_tags_str`, created `id2label`, `label2id`, and fixed entity types: `("PER", "ORG", "LOC")`.  
  These are required for correct seqeval evaluation and token‑classification models.
- Created a reduced dataset `ds_small`:
  - `train = 7000`, `validation = 2000`, `test = 2000`, with `shuffle(seed=SEED)`.

- Using `ds_small` is a compromise between **sufficient data volume** and **training time**.

---

## 2. Utilities and Metrics

- Implemented functions for:
  - text normalization and reconstruction from tokens with offsets,
  - BIO ↔ spans conversion (token‑level and character‑level),
  - strict `seqeval`‑F1 (`seqeval_strict_micro_f1`),
  - detailed error breakdown: exact, type_confusion, boundary_mismatch, missing, spurious,
  - confusion matrix by entity type,
  - `summarize(model_name, y_true, y_pred)` — unified summary format for the final comparison table.
- Functions `compute_metrics` and `trainer_predictions_to_seqeval` for integration with `Trainer`.

- Strict span‑level evaluation and error breakdown allow us to understand **not only F1**, but also the **nature of errors** (boundaries, types, omissions).
- The unified `summarize` function makes final model comparison transparent and consistent.

---

## 3. External Baselines: Natasha / Slovnet and GLiNER Multi

### 3.1. Natasha / Slovnet

- Used `Segmenter`, `NewsEmbedding`, `NewsNERTagger`.
- For each example:
  - tokens → text + offsets,
  - run Natasha,
  - convert predicted spans into BIO labels.
- Obtained `y_true_nat`, `y_pred_nat`, computed strict F1.

- Natasha/Slovnet is a strong and **traditional baseline** for Russian NER.
- It provides a reference point for evaluating how far ModernBERT and additional techniques can go.

### 3.2. GLiNER Multi

- Loaded model `urchade/gliner_multi-v2.1`.
- Predicted entities with labels `["person", "organization", "location"]`, mapped to `PER/ORG/LOC`.
- Same pipeline: text → spans → BIO → `y_true_gl`, `y_pred_gl`, metrics.

- GLiNER is a modern multilingual NER model and provides a **strong external baseline**.
- It is important to compare ModernBERT not only with Natasha, but also with newer architectures.

---

## 4. Data Preparation for ModernBERT

- Selected `model_id = "deepvk/RuModernBERT-base"`.
- Tokenization with `AutoTokenizer`:
  - `is_split_into_words=True`,
  - `max_length=192`,
  - BIO label alignment for subword tokens:
    - the first token of a word receives the label,
    - subsequent subwords receive `-100` (ignored in loss).
- Created `DataCollatorForTokenClassification` with `padding="longest"`.

- ModernBERT‑base is a specialized Russian‑language model well suited for NER.
- `max_length=192` balances coverage and memory efficiency.
- Ignoring internal subword tokens in the loss is standard and correct for token classification.

---

## 5. Baseline Fine‑Tuning ModernBERT‑base (NER only)

- `TrainingArguments` (`args_base`):
  - `learning_rate=2e-5` — typical LR for BERT‑like models,
  - `per_device_train_batch_size=16`, `per_device_eval_batch_size=32`,
  - `gradient_accumulation_steps=2` — effective batch size without memory overflow,
  - `num_train_epochs=10`,
  - `fp16=True` — GPU acceleration,
  - `eval_strategy="steps"`, `eval_steps=200`, `save_steps=200`,
  - `load_best_model_at_end=True`, `metric_for_best_model="f1"`,
  - `warmup_ratio=0.1`, `weight_decay=0.01`,
  - `EarlyStoppingCallback(early_stopping_patience=3)`.
- Model: `AutoModelForTokenClassification` with `num_labels=len(label_names)`.
- Training on `tokenized["train"]`, validation on `tokenized["validation"]`.
- Predictions on `tokenized["test"]` → `y_true_base`, `y_pred_base`, `res_base`.

- LR `2e-5` and 10 epochs are well‑established settings for BERT‑like NER models.
- Early stopping prevents overfitting and saves time.
- Warmup + weight decay ensure stable transformer training.

---

## 6. MLM Pretraining ModernBERT‑base

- MLM model: `AutoModelForMaskedLM.from_pretrained(model_id)`.
- Converted `tokens` → full text (`factru_to_texts`), train split only.
- Tokenized for MLM with `max_length=192`.
- `DataCollatorForLanguageModeling` with `mlm_probability=0.15`.
- `TrainingArguments` for MLM:
  - `learning_rate=5e-5`,
  - `per_device_train_batch_size=32`,
  - `num_train_epochs=3`,
  - `fp16=True`.
- Trained → saved as `"modernbert_base_mlm_factru_final"`.

Then:

- Loaded `AutoModelForTokenClassification.from_pretrained("modernbert_base_mlm_factru_final", ...)`.
- Fine‑tuned with the same `args_base` and same NER data.
- Predictions → `y_true_base_mlm`, `y_pred_base_mlm`, `res_base_mlm`.

- **Idea:** adapt ModernBERT to the factRuEval domain via MLM to improve language representations before NER.
- 3 epochs and LR `5e-5` provide meaningful adaptation without catastrophic forgetting.
- Reusing `args_base` ensures a fair comparison: only the MLM stage changes.

---

## 7. Concept Masking (Entity‑Focused Masking)

- Standard WWM collators were incompatible with offsets, so a **custom collator** `NERConceptMaskCollator` was implemented:
  - padding inputs and labels,
  - selecting entity tokens (BIO ≠ "O", not `-100`, not special tokens),
  - replacing tokens with `[MASK]` with probability `mask_prob=0.15`.
- Fine‑tuning:
  - model `AutoModelForTokenClassification.from_pretrained(model_id, ...)`,
  - same `args_base`,
  - `train_dataset=tokenized["train"]`, `eval_dataset=tokenized["validation"]`,
  - `data_collator=concept_collator`.
- Predictions → `y_true_base_concept`, `y_pred_base_concept`, `res_base_concept`.

- Concept masking is a soft alternative to WWM: we **mask only entity tokens**, forcing the model to better reconstruct and distinguish them.
- This strengthens entity‑focused learning without complex WWM integration or breaking the offset pipeline.

---

## 8. Synthetic Annotation via GLiNER and Combined Dataset

1. **Text corpus collection:**
   - For all factRuEval splits (train + validation + test), collected a corpus of sentences:
     - `example_to_sentences` → `{"text": text}`.
   - Obtained `factru_texts`.

2. **GLiNER annotation:**
   - Function `gliner_annotate_texts`:
     - `text` → tokens → normalized text + offsets,
     - predicted `gliner_char_spans` with `threshold=0.4`,
     - converted to BIO and then numeric `ner_tags`.
   - Obtained `synthetic` with fields `["tokens", "ner_tags"]`.
   - Filtered empty examples.

3. **Combined dataset:**
   - `synthetic_train = synthetic["train"]`.
   - `combined_train = concatenate_datasets([ds_small["train"], synthetic_train])`.
   - `combined_ds = DatasetDict({"train": combined_train, "validation": ds_small["validation"], "test": ds_small["test"]})`.
   - `combined_tokenized = combined_ds.map(tokenize_and_align_labels, batched=True)`.

4. **Fine‑tuning on combined dataset:**
   - Model `AutoModelForTokenClassification.from_pretrained(model_id, ...)`.
   - Same `args_base`.
   - `train_dataset=combined_tokenized["train"]`, `eval_dataset=combined_tokenized["validation"]`.
   - Collator — standard `data_collator`.
   - Predictions → `y_true_base_synth`, `y_pred_base_synth`, `res_base_synth`.

- GLiNER is used as an **external annotator** to generate synthetic NER examples.
- The idea is to expand the training corpus with additional (noisy but useful) annotations, making the model **more robust and generalizable**.
- Combining original factRuEval with synthetic data increases the diversity of contexts and entity examples.

---

## 9. Final Table and Interpretation

The final table (see the next cell with `results_df`) includes:

- *Natasha / Slovnet* — classical baseline.
- *GLiNER Multi* — modern multilingual NER.
- *ModernBERT‑base (NER only)* — pure fine‑tuning.
- *ModernBERT‑base (MLM)* — effect of domain MLM pretraining.
- *ModernBERT‑base (concept masking)* — effect of entity masking.
- *ModernBERT‑base (NER + synthetic via GLiNER)* — effect of expanded corpus.

Key conclusions:

- **ModernBERT‑base** already outperforms external baselines on strict F1.
- **MLM pretraining** provides additional gains via better domain adaptation.
- **Concept masking** helps the model focus on entities, improving boundary and type accuracy.
- **Synthetic annotation** via GLiNER expands the training corpus and offers another path to improved robustness and generalization.

---

## 10. Reproducibility

- All key seeds are fixed (`SEED = 42` + seeds for `random`, `numpy`, `torch`).
- All steps — from data loading to the final table — are organized as sequential notebook cells.
- The notebook can be run **end‑to‑end**, with GPU and Hugging Face Hub access, without manual intervention.


**See full experiment summary in [GitHub Issue #1](https://github.com/Olmioris/Ru-ner-modernbert-factrueval/issues/1#issue-4840110393)**.
