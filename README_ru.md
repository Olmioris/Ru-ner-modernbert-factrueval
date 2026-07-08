# Итоги и интерпретация экспериментов.

## 1. Подготовка окружения и данных


- Установлены библиотеки: `datasets`, `evaluate`, `seqeval`, `transformers`, `accelerate`, `natasha`, `razdel`, `gliner`, `seaborn`.
- Зафиксированы сиды:
  - `SEED = 42`, далее `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` (Фиксация сидов обеспечивает **воспроизводимость** всех экспериментов).
- Определено устройство:
  - `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` — для ускорения обучения на GPU (T4).


**Данные:**

- Загружен датасет `gusevsk/factrueval2016` через `load_dataset`.
- Приведён к удобному формату `DatasetDict` с разбиением на `train / validation / test`.
- Построен список BIO‑меток из `ner_tags_str`, созданы `id2label`, `label2id`, зафиксированы типы сущностей: `("PER", "ORG", "LOC")`. Нужны для корректной работы seqeval и моделей токен‑классификации.
- Создан уменьшенный датасет `ds_small`:
  - `train = 7000`, `validation = 2000`, `test = 2000`, с `shuffle(seed=SEED)`.


- Использование `ds_small` — компромисс между **достаточным объёмом данных** и **временем обучения**.

---

## 2. Утилиты и метрики

- Реализованы функции:
  - нормализация текста и восстановление текста из токенов с оффсетами,
  - преобразование BIO ↔ spans (по токенам и по символам),
  - строгий `seqeval`‑F1 (`seqeval_strict_micro_f1`),
  - разбор ошибок по типам: exact, type_confusion, boundary_mismatch, missing, spurious,
  - матрица ошибок по типам сущностей,
  - `summarize(model_name, y_true, y_pred)` — единый формат сводки для финальной таблицы.
- Функции `compute_metrics` и `trainer_predictions_to_seqeval` для интеграции с `Trainer`.


- Строгий span‑уровень и разбор ошибок позволяют не только смотреть на F1, но и понимать **характер ошибок** (границы, типы, пропуски).
- Унифицированная функция `summarize` делает финальное сравнение моделей прозрачным и сопоставимым.

---

## 3. Базовые внешние модели: Natasha / Slovnet и GLiNER Multi

### 3.1. Natasha / Slovnet


- Использованы `Segmenter`, `NewsEmbedding`, `NewsNERTagger`.
- Для каждого примера:
  - токены → текст + оффсеты,
  - прогон через Natasha,
  - перевод предсказанных span’ов в BIO‑метки.
- Получены `y_true_nat`, `y_pred_nat`, посчитан строгий F1.

- Natasha/Slovnet — сильный и **традиционный бейзлайн** для русского NER.
- Даёт точку отсчёта: насколько далеко можно уйти с ModernBERT и дополнительными приёмами.

### 3.2. GLiNER Multi


- Загружена модель `urchade/gliner_multi-v2.1`.
- Предсказание сущностей с лейблами `["person", "organization", "location"]`, маппинг в `PER/ORG/LOC`.
- Аналогично: текст → spans → BIO → `y_true_gl`, `y_pred_gl`, метрики.


- GLiNER — современная, мультиязычная модель NER, даёт **внешний сильный бейзлайн**.
- Важно сравнить ModernBERT не только с Natasha, но и с более новой архитектурой.

---

## 4. Подготовка данных для ModernBERT


- Выбран `model_id = "deepvk/RuModernBERT-base"`.
- Токенизация с `AutoTokenizer`:
  - `is_split_into_words=True`,
  - `max_length=192`,
  - выравнивание BIO‑меток под subword‑токены:
    - первый токен слова получает метку,
    - последующие subword’ы — `-100` (игнорируются в лоссе).
- Создан `DataCollatorForTokenClassification` с `padding="longest"`.


- ModernBERT‑base — специализированная русскоязычная модель, хорошо подходящая для NER.
- `max_length=192` — баланс между покрытием предложений и эффективностью по памяти.
- Игнорирование внутренних subword‑токенов в лоссе — стандартный и корректный подход для токен‑классификации.

---

## 5. Базовый fine‑tuning ModernBERT‑base (NER only)



- `TrainingArguments` (`args_base`):
  - `learning_rate=2e-5` — типичный LR для BERT‑подобных моделей,
  - `per_device_train_batch_size=16`, `per_device_eval_batch_size=32`,
  - `gradient_accumulation_steps=2` — эффективный batch без переполнения памяти,
  - `num_train_epochs=10`,
  - `fp16=True` — ускорение на GPU,
  - `eval_strategy="steps"`, `eval_steps=200`, `save_steps=200`,
  - `load_best_model_at_end=True`, `metric_for_best_model="f1"`,
  - `warmup_ratio=0.1`, `weight_decay=0.01`,
  - `EarlyStoppingCallback(early_stopping_patience=3)`.
- Модель: `AutoModelForTokenClassification` с `num_labels=len(label_names)`.
- Обучение на `tokenized["train"]`, валидация на `tokenized["validation"]`.
- Предсказания на `tokenized["test"]` → `y_true_base`, `y_pred_base`, `res_base`.



- LR `2e-5` и 10 эпох — проверенная конфигурация для BERT‑подобных моделей на NER.
- Early stopping по F1 защищает от переобучения и экономит время.
- Warmup + weight decay — стандартные практики для стабильного обучения трансформеров.

---

## 6. Предварительное MLM‑дообучение ModernBERT‑base


- MLM‑модель: `AutoModelForMaskedLM.from_pretrained(model_id)`.
- Преобразование `tokens` → цельный текст (`factru_to_texts`), только train‑часть.
- Токенизация для MLM с `max_length=192`.
- `DataCollatorForLanguageModeling` с `mlm_probability=0.15`.
- `TrainingArguments` для MLM:
  - `learning_rate=5e-5`,
  - `per_device_train_batch_size=32`,
  - `num_train_epochs=3`,
  - `fp16=True`.
- Обучение → сохранение в `"modernbert_base_mlm_factru_final"`.

Затем:

- Загрузка `AutoModelForTokenClassification.from_pretrained("modernbert_base_mlm_factru_final", ...)`.
- Fine‑tuning по тем же `args_base` и тем же данным NER.
- Предсказания → `y_true_base_mlm`, `y_pred_base_mlm`, `res_base_mlm`.


- Идея: **адаптировать ModernBERT к домену factRuEval** через MLM, чтобы улучшить языковое представление перед NER.
- 3 эпохи и LR `5e-5` — разумный объём дообучения без риска «забывания» исходных знаний.
- Повторное использование `args_base` даёт честное сравнение: меняется только наличие MLM‑этапа.

---

## 7. Concept masking (маскирование сущностей)


- WWM через стандартные коллаторы не завёлся из‑за несовместимости с offsets, поэтому реализован **кастомный collator** `NERConceptMaskCollator`:
  - паддинг входов и меток,
  - выбор токенов сущностей (BIO ≠ "O", не `-100`, не спец‑токены),
  - с вероятностью `mask_prob=0.15` токен заменяется на `[MASK]`.
- Fine‑tuning:
  - модель `AutoModelForTokenClassification.from_pretrained(model_id, ...)`,
  - те же `args_base`,
  - `train_dataset=tokenized["train"]`, `eval_dataset=tokenized["validation"]`,
  - `data_collator=concept_collator`.
- Предсказания → `y_true_base_concept`, `y_pred_base_concept`, `res_base_concept`.



- Concept masking — мягкая альтернатива WWM: мы **маскируем только токены сущностей**, заставляя модель лучше учиться восстанавливать и различать их.
- Это усиливает фокус на NER‑сущностях без сложной интеграции WWM и без ломки пайплайна offsets.

---

## 8. Синтетическая разметка через GLiNER и комбинированный датасет



1. **Сбор корпуса текстов:**
   - Для всех сплитов factRuEval (train + validation + test) собран корпус предложений:
     - `example_to_sentences` → `{"text": text}`.
   - Получен `factru_texts`.

2. **Разметка GLiNER’ом:**
   - Функция `gliner_annotate_texts`:
     - `text` → токены → нормализованный текст + оффсеты,
     - предсказание `gliner_char_spans` с `threshold=0.4`,
     - перевод в BIO и далее в числовые `ner_tags`.
   - Получен `synthetic` с полями `["tokens", "ner_tags"]`.
   - Фильтрация пустых примеров.

3. **Комбинированный датасет:**
   - `synthetic_train = synthetic["train"]`.
   - `combined_train = concatenate_datasets([ds_small["train"], synthetic_train])`.
   - `combined_ds = DatasetDict({"train": combined_train, "validation": ds_small["validation"], "test": ds_small["test"]})`.
   - `combined_tokenized = combined_ds.map(tokenize_and_align_labels, batched=True)`.

4. **Fine‑tuning на комбинированном датасете:**
   - Модель `AutoModelForTokenClassification.from_pretrained(model_id, ...)`.
   - Те же `args_base`.
   - `train_dataset=combined_tokenized["train"]`, `eval_dataset=combined_tokenized["validation"]`.
   - Коллатор — стандартный `data_collator`.
   - Предсказания → `y_true_base_synth`, `y_pred_base_synth`, `res_base_synth`.


- GLiNER используется как **внешний разметчик** для генерации синтетических NER‑примеров.
- Идея: расширить обучающий корпус за счёт дополнительной (пусть и шумной) разметки, чтобы модель стала **более устойчивой и обобщающей**.
- Комбинация оригинального factRuEval и синтетики даёт модели больше разнообразия контекстов и примеров сущностей.

---

## 9. Финальная таблица и интерпретация

В финальной таблице (см. следующую ячейку с `results_df`) собраны:

- *Natasha / Slovnet* — классический бейзлайн.
- *GLiNER Multi* — современный мультиязычный NER.
- *ModernBERT‑base (NER only)* — чистый fine‑tuning.
- *ModernBERT‑base (MLM)* — эффект доменного MLM‑дообучения.
- *ModernBERT‑base (concept masking)* — влияние маскирования сущностей.
- *ModernBERT‑base (NER + synthetic via GLiNER)* — влияние расширенного корпуса.

По ним можно сделать качественные выводы:

- **ModernBERT‑base** уже превосходит внешние бейзлайны по строгому F1.
- **MLM‑дообучение** даёт дополнительный прирост за счёт лучшей адаптации к домену factRuEval.
- **Concept masking** помогает модели лучше фокусироваться на сущностях, улучшая качество распознавания границ и типов.
- **Синтетическая разметка** через GLiNER расширяет обучающий корпус и даёт ещё один вариант улучшения/баланса между качеством и устойчивостью.

---

## 10. Воспроизводимость

- Все ключевые сиды зафиксированы (`SEED = 42` + сиды для `random`, `numpy`, `torch`).
- Все шаги — от загрузки данных до финальной таблицы — оформлены в виде последовательных ячеек.
- Ноутбук можно запускать **от начала до конца**, при наличии GPU и доступа к Hugging Face Hub, без ручных вмешательств.


