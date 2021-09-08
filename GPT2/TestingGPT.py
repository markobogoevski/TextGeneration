import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, set_seed
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import re
from datasets import tqdm
from transformers import get_linear_schedule_with_warmup, AdamW

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def choose_from_top(probs, n=5):
    probs = np.asarray(probs).astype('float64')
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


class SentenceDataset(Dataset):
    def __init__(self, dataset_path='Data/SourceData'):
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


def generate_shakespeare(seed_texts):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)
    models_folder = "GPT2_model"
    model_path = os.path.join(models_folder, f"gpt2_medium_shakespeare_best.pt")
    model.load_state_dict(torch.load(model_path))
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    set_seed(42)
    path_to_save = "../Generations/GPT2_generated_shakespeare"
    os.makedirs(path_to_save, exist_ok=True)
    song_num = 0
    for seed_text in seed_texts:
        song_num += 1
        line = generator(seed_text, max_length=1000, num_return_sequences=1)
        line = line[0]['generated_text']
        lines = line.split(seed_text)[1].split('\n')
        lines = [line.split("BOS>")[1] if "BOS>" in str(line) else "\n" for line in lines]
        final_string = '\n'.join(lines)
        final_text = seed_text + "\n" + final_string
        with open(os.path.join(path_to_save, f'song_number_{song_num}'), 'w') as f:
            f.write(final_text)


def calculate_and_write_perplexity():
    perplexity_path = 'perplexity.txt'
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)
    models_folder = "GPT2_model"
    model_path = os.path.join(models_folder, f"gpt2_medium_shakespeare_best.pt")
    model.load_state_dict(torch.load(model_path))

    dataset_path = "../Data/SourceData"
    dataset_path = os.path.join(dataset_path, 'alllines.txt')
    test_list = []
    start_of_text_token = "<BOS>"
    end_of_text_token = "<EOS>"

    with open(dataset_path) as file:
        file_lines = file.readlines()[15000:18750]  # In this way 20% of the set is used as test, 80% for training
        for line in file_lines:
            line = f"{start_of_text_token} {line} {end_of_text_token}"
            test_list.append(line)

    test_set = '\n'.join(test_list)
    encodings = tokenizer(test_set, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc).cpu().numpy()
    with open(perplexity_path, 'w') as f:
        f.write(str(ppl))


if __name__ == "__main__":
    pass
    # Training the model

    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # model = model.to(device)
    # dataset = SentenceDataset()
    # sentence_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # # This is used to optimize the dataset for gpu memory issues
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
    #
    # Now testing the model for generation
    # seed_text = "<BOS> My royal father, cheer these noble lords <EOS>" \
    #             "<BOS> And hearten those that fight in your defence: <EOS>" \
    #             "<BOS> Unsheathe your sword, good father, cry 'Saint George!' <EOS>" \
    #             "<BOS> March. Enter EDWARD, GEORGE, RICHARD, WARWICK, NORFOLK, MONTAGUE, and Soldiers <EOS>"
    #
    # seed_text1 = "<BOS> ACT I. <EOS>" \
    #              "<BOS> SCENE I. London. The palace. <EOS>" \
    #              "<BOS> Enter KING HENRY, LORD JOHN OF LANCASTER, the EARL of WESTMORELAND, SIR WALTER BLUNT, and others <EOS>" \
    #              "<BOS> So shaken as we are, so wan with care, <EOS>" \
    #              "<BOS> Find we a time for frighted peace to pant, <EOS>"
    #
    # seed_text2 = "<BOS> What, gone, my lord, and bid me not farewell! <EOS>" \
    #              "<BOS> Witness my tears, I cannot stay to speak. <EOS>" \
    #              "<BOS> Exeunt GLOUCESTER and Servingmen <EOS>" \
    #              "<BOS> Art thou gone too? all comfort go with thee! <EOS>"
    #
    # seed_text3 = "<BOS> How proud, how peremptory, and unlike himself? <EOS>" \
    #              "<BOS> We know the time since he was mild and affable, <EOS>" \
    #              "<BOS> And if we did but glance a far-off look, <EOS>" \
    #              "<BOS> Immediately he was upon his knee, <EOS>"
    #
    # seed_text4 = "<BOS> Then, executioner, unsheathe thy sword: <EOS>" \
    #              "<BOS> By him that made us all, I am resolved <EOS>" \
    #              "<BOS> that Clifford's manhood lies upon his tongue. <EOS>" \
    #              "<BOS> Say, Henry, shall I have my right, or no? <EOS>"
    #
    # seed_text5 = "<BOS> My royal father, cheer these noble lords <EOS>" \
    #              "<BOS> And hearten those that fight in your defence: <EOS>" \
    #              "<BOS> Unsheathe your sword, good father, cry 'Saint George!' <EOS>" \
    #              "<BOS> March. Enter EDWARD, GEORGE, RICHARD, WARWICK, NORFOLK, MONTAGUE, and Soldiers <EOS>"
    #
    # seed_texts = [seed_text1, seed_text2, seed_text3, seed_text4, seed_text5]
    # generate_shakespeare(seed_texts)

    # Calculating perplexity on test set
    # calculate_and_write_perplexity()
