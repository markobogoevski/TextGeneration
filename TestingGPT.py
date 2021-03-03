import random

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from transformers import get_linear_schedule_with_warmup, AdamW

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def     choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


class SentenceDataset(Dataset):
    def __init__(self, dataset_path='Data'):
        super().__init__()

        train_path = os.path.join(dataset_path, 'alllines.txt')

        self.sentence_list = []
        self.start_of_text_token = "<BOS>"
        self.end_of_text_token = "<EOS>"

        with open(train_path) as file:
            file_lines = file.readlines()[:15000]
            for line in file_lines:
                line = f"{self.start_of_text_token}{line}{self.end_of_text_token}"
                self.sentence_list.append(line)

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, item):
        return self.sentence_list[item]


def generate_shakespeare():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)

    models_folder = "GPT2_model"

    model_path = os.path.join(models_folder, f"gpt2_medium_shakespeare_best.pt")
    model.load_state_dict(torch.load(model_path))

    shake_output_file_path = "GPT2_generated_shakespeare"
    os.makedirs(shake_output_file_path, exist_ok=True)
    model.eval()
    number_of_plays = 15
    number_of_lines_per_play = 15
    with torch.no_grad():
        for shake_idx in range(number_of_plays):
            print(f"Generating play number : {shake_idx + 1}")
            final_output_path = os.path.join(shake_output_file_path, f"play_{shake_idx + 1}.txt")

            i = 0
            while i < number_of_lines_per_play:
                cur_ids = torch.tensor(tokenizer.encode("<BOS>")).unsqueeze(0).to(device)
                word_count = 0
                while word_count<15:
                    outputs = model(cur_ids, labels=cur_ids)
                    loss, logits = outputs[:2]
                    softmax_logits = torch.softmax(logits[0, -1],
                                                   dim=0)  # Take the first(from only one in this case) batch and the
                    # last predicted embedding
                    if word_count < 3:
                        n = 20  # For randomness
                    else:
                        n = 5 # For context
                    next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(),
                                                    n=n)  # Randomly(from the topN probability distribution) select the
                    # next word
                    if next_token_id in tokenizer.encode('<EOS>'):
                        if word_count < 15:
                            continue
                        else:
                            break
                    else:
                        cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id],
                                            dim=1)  # Add the last word to the running sequence
                        word_count += 1
                i += 1
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                with open(final_output_path, 'a') as f:
                    output_text = output_text.split("<BOS>")[1]
                    f.write(f"{output_text}")  # Write the line


if __name__ == "__main__":
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # model = model.to(device)
    # dataset = SentenceDataset()
    # sentence_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    #
    # BATCH_SIZE = 32
    # EPOCHS = 100
    # LEARNING_RATE = 3e-5
    # WARMUP_STEPS = 5000
    # MAX_SEQ_LEN = 400
    #
    # model.train()
    # optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=WARMUP_STEPS,
    #                                             num_training_steps=5000)
    # proc_seq_count = 0
    # sum_loss = 0.0
    # batch_count = 0
    #
    # tmp_sentence_tens = None
    # models_folder = "GPT2_model"
    # if not os.path.exists(models_folder):
    #     os.mkdir(models_folder)
    # best_sum = 10e6
    # for epoch in range(EPOCHS):
    #
    #     print(f"EPOCH {epoch} started" + '=' * 30)
    #
    #     for idx, sentence in enumerate(sentence_loader):
    #
    #         #################### "Fit as many sentence sequences into MAX_SEQ_LEN sequence as possible" logic start ####
    #         sentence_tens = torch.tensor(tokenizer.encode(sentence[0])).unsqueeze(0).to(device)
    #         # Skip sample from dataset if it is longer than MAX_SEQ_LEN
    #         if sentence_tens.size()[1] > MAX_SEQ_LEN:
    #             continue
    #
    #         # The first sentence sequence in the sequence
    #         if not torch.is_tensor(tmp_sentence_tens):
    #             tmp_sentence_tens = sentence_tens
    #             continue
    #         else:
    #             # The next sentence does not fit in so we process the sequence and leave the last sentence
    #             # as the start for next sequence
    #             if tmp_sentence_tens.size()[1] + sentence_tens.size()[1] > MAX_SEQ_LEN:
    #                 work_sentence_tens = tmp_sentence_tens
    #                 tmp_sentence_tens = sentence_tens
    #             else:
    #                 # Add the sentence to sequence, continue and try to add more
    #                 tmp_sentence_tens = torch.cat([tmp_sentence_tens, sentence_tens[:, 1:]], dim=1)
    #                 continue
    #         ################## Sequence ready, process it trough the model ##################
    #
    #         outputs = model(work_sentence_tens, labels=work_sentence_tens)
    #         loss, logits = outputs[:2]
    #         loss.backward()
    #         sum_loss = sum_loss + loss.detach().data
    #
    #         proc_seq_count = proc_seq_count + 1
    #         if proc_seq_count == BATCH_SIZE:
    #             proc_seq_count = 0
    #             batch_count += 1
    #             optimizer.step()
    #             scheduler.step()
    #             optimizer.zero_grad()
    #             model.zero_grad()
    #
    #     if sum_loss < best_sum:
    #         best_sum = sum_loss
    #         # New best epoch
    #         torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_shakespeare_best.pt"))
    #     sum_loss = 0

    generate_shakespeare()
