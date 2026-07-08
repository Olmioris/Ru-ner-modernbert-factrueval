import torch

class NERConceptMaskCollator:
    def __init__(self, tokenizer, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]

        max_len = max(len(x) for x in input_ids)

        padded_ids = []
        padded_labels = []

        for ids, labs in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)

        input_ids = torch.tensor(padded_ids)
        labels = torch.tensor(padded_labels)

        mask_token_id = self.tokenizer.mask_token_id

        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                if labels[i, j] != -100 and torch.rand(1).item() < self.mask_prob:
                    input_ids[i, j] = mask_token_id

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).long()
        }

