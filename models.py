import time
import torch.utils.data
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import random
from score_methods import *
from torch.utils.tensorboard import SummaryWriter

class GRULyrics(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size,word2vec_matrix,experiment_num, num_layers=1,
                 dropout=0.3):
        super(GRULyrics, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word2vec_matrix), freeze=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.writer = SummaryWriter(f"final_runs/experiment_4")

    def forward(self, x, h=None):
        x = self.embedding(x)
        x = x.float()
        if h is None:
            h = torch.zeros((self.num_layers, x.size(0), self.hidden_size))
        out, h = self.gru(x, h)

        out = self.dropout(out[:,-1,:])
        logits = self.linear(out)

        return logits, h

    def fit(self, x_train,y_train, x_val, y_val,epochs, batch_size,learning_rate,device=None):

        self.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        batch_total = 0
        total_train_batches = len(train_loader)
        total_val_batches = len(val_loader)
        start_time = time.time()
        for epoch in range(epochs):
            train_loss_per_epoch = 0
            for batch_idx, batch in enumerate(train_loader):
                x = batch[0].long()  #.to(device)
                y = batch[1].float()    #.to(device)

                preds,h = self.forward(x)
                loss = criterion(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_per_epoch += loss.item()
                if (batch_idx) % 10 == 0:
                    print(
                        f'Training - Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')
                self.writer.add_scalar('Train_Loss/Batch', loss.item(), batch_total)
                batch_total += 1
            # Validation phase
            self.eval()

            with torch.no_grad():
                val_loss_per_epoch = 0
                for batch_idx, batch in enumerate(val_loader):
                    x = batch[0].long()  # .to(device)
                    y = batch[1].float()

                    preds, h = self.forward(x)
                    loss = criterion(preds, y)
                    val_loss_per_epoch += loss.item()
                avg_train_loss = train_loss_per_epoch / total_train_batches
                avg_val_loss = val_loss_per_epoch/ total_val_batches
            self.writer.add_scalars('Loss_Epochs',
                                    {"Train": avg_train_loss, "Validation": avg_val_loss},
                                    epoch)
            self.val_loss = avg_val_loss    # store the avg loss per epoch. finally the avg of last epoch stored
        end_time = time.time()
        training_time = end_time-start_time
        self.writer.flush()
        self.writer.close()
        return self, training_time

    @torch.no_grad()
    def predict(self,first_words_indices, device=None, num_of_words_generate=20):

        # return words_indices
        generated_indices = []

        # Convert prompt to integer sequence
        pattern = first_words_indices

        self.eval()
        # Generate text
        with torch.no_grad():
            for i in range(num_of_words_generate):
                # Format input array of int into PyTorch tensor

                lyrics_sequence = torch.tensor(pattern).long()
                # Generate logits as output from the model
                output, h = self.forward(lyrics_sequence)

                distribution = OneHotCategorical(logits=output, probs=None,validate_args=False)
                probs = distribution.probs

                output_np = probs.squeeze().numpy()     # create shape of voc_size and numpy array [1,2,...VOC_SIZE] of probs

                # Generate array of word indices
                word_index_array = np.arange(len(output_np))        # create array of indices to use in random.choice

                # Select a word based on the predicted probabilities
                chosen_index = np.random.choice(random.choices(word_index_array, k=5, weights=output_np))
                generated_indices.append(chosen_index)
                pattern = np.concatenate((pattern, [[chosen_index]]), axis=1)       # add

                pattern = pattern[:, 1:]    # Remove the first item from the list at index 0

        return generated_indices

    def evaluate(self, original_lyrics, generated_lyrics, word2vec_dict):

        cos_sim_no_order = calculate_cosine_similarity(original_lyrics, generated_lyrics, word2vec_dict)

        cos_sim_with_order = calculate_cosine_similiraity_between_parallel_words(original_lyrics, generated_lyrics,
                                                                                 word2vec_dict)

        cos_sim_3_gram = compute_cosine_similarity_between_ngrams(original_lyrics, generated_lyrics, n=3,
                                                                  word2vec=word2vec_dict)
        cos_sim_5_gram = compute_cosine_similarity_between_ngrams(original_lyrics, generated_lyrics, n=5,
                                                                  word2vec=word2vec_dict)
        cos_sim_7_gram = compute_cosine_similarity_between_ngrams(original_lyrics, generated_lyrics, n=7,
                                                                  word2vec=word2vec_dict)

        bleu_score = calculate_bleu_score(original_lyrics, generated_lyrics)

        subjectivity, polarity = calculate_subjectivity_polarity(generated_lyrics)

        return {
            'loss val': self.val_loss,
            'cos_sim_no_order': cos_sim_no_order,
            'cos_sim_with_order': cos_sim_with_order,
            'cos_sim_3_gram': cos_sim_3_gram,
            'cos_sim_5_gram': cos_sim_5_gram,
            'cos_sim_7_gram': cos_sim_7_gram,
            "bleu score": bleu_score,
            "subjectivity": subjectivity,
            "polarity": polarity
        }


#######################

class GRUMelodies(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size,word2vec_matrix,experiment_num,num_layers=2,
                 dropout=0.3):
        super(GRUMelodies, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word2vec_matrix), freeze=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.writer = SummaryWriter(f"runs/experiment_2")

    def forward(self, x, input_features, h=None):
        x = self.embedding(x)
        x = x.float()
        # Concatenate input features with embedded input
        x = torch.cat((x, input_features), dim=-1)

        if h is None:
            h = torch.zeros((self.num_layers, x.size(0), self.hidden_size))

        out, h = self.gru(x, h)
        out = self.dropout(out[:, -1, :])
        logits = self.linear(out)

        return logits, h

    def fit(self, x_train,y_train, x_val, y_val, melodies_train, melodies_val,epochs, batch_size,learning_rate,device=None):

        self.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        melodies_train_dataset = TensorDataset(torch.tensor(melodies_train,dtype=torch.float32))
        melodies_train_loader = DataLoader(melodies_train_dataset, batch_size=batch_size, shuffle=False)
        melodies_val_dataset = TensorDataset(torch.tensor(melodies_val,dtype=torch.float32))
        melodies_val_loader = DataLoader(melodies_val_dataset, batch_size=batch_size, shuffle=False)

        batch_total = 0
        total_train_batches = len(train_loader)
        total_val_batches = len(val_loader)
        start_time = time.time()
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (batch_words,batch_melodies) in enumerate(zip(train_loader, melodies_train_loader)):

                x = batch_words[0].long()  #.to(device)
                features = batch_melodies[0]    # because it stored in a list
                y = batch_words[1].float()    #.to(device)
                preds,h = self.forward(x,features)

                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (batch_idx) % 10 == 0:
                    print(
                        f'Training - Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')
                train_loss += loss.item()
                self.writer.add_scalar('Train_Loss/Batch', loss.item(), batch_total)

                batch_total += 1

            avg_train_loss = train_loss / total_train_batches

            # Validation phase
            self.eval()

            with torch.no_grad():
                loss_per_epoch = 0
                for batch_idx, (batch_words,batch_melodies) in enumerate(zip(val_loader, melodies_val_loader)):
                    x = batch_words[0].long()  # .to(device)
                    features = batch_melodies[0]
                    y = batch_words[1].float()  # .to(device)

                    preds, h = self.forward(x,features)
                    loss = criterion(preds, y)
                    loss_per_epoch += loss.item()
                avg_val_loss = loss_per_epoch/total_val_batches
            self.writer.add_scalars('Loss_Epochs',
                                   {"Train": avg_train_loss,"Validation":avg_val_loss},
                                    epoch)
            self.val_loss = avg_val_loss
        end_time = time.time()
        training_time = end_time- start_time
        self.writer.flush()
        return self, training_time

    @torch.no_grad()
    def predict(self,first_words_indices,melody_features,device=None, num_of_words_generate=20):
        # return words_indices
        generated_indices = []
        sequence_len = first_words_indices.shape[1]
        # Convert prompt to integer sequence
        pattern = first_words_indices
        melody_features=torch.tensor(melody_features, dtype=torch.float32)
        melody_features = melody_features.unsqueeze(dim=0)
        # Set model to evaluation mode

        self.eval()
        # Generate text
        with torch.no_grad():
            for i in range(num_of_words_generate):
                # Format input array of int into PyTorch tensor
                lyrics_sequence = torch.tensor(pattern).long()

                # Generate logits as output from the model
                output, h = self.forward(lyrics_sequence,melody_features[:,i:i+sequence_len,:])

                # Randomly select one of the top k indices
                distribution = OneHotCategorical(logits=output, probs=None,validate_args=False)
                probs = distribution.probs

                output_np = probs.squeeze().numpy()     # create shape of voc_size and numpy array [1,2,...VOC_SIZE] of probs

                # Generate array of word indices
                word_index_array = np.arange(len(output_np))        # create array of indices to use in random.choice

                # Select a word based on the predicted probabilities
                chosen_index = np.random.choice(random.choices(word_index_array, k=10, weights=output_np))
                generated_indices.append(chosen_index)
                pattern = np.concatenate((pattern, [[chosen_index]]), axis=1)       # add

                pattern = pattern[:, 1:]    # Remove the first item from the list at index 0

        return generated_indices

    def evaluate(self, original_lyrics, generated_lyrics, word2vec_dict):

        cos_sim_no_order = calculate_cosine_similarity(original_lyrics, generated_lyrics, word2vec_dict)

        cos_sim_with_order = calculate_cosine_similiraity_between_parallel_words(original_lyrics, generated_lyrics, word2vec_dict)

        cos_sim_3_gram = compute_cosine_similarity_between_ngrams(original_lyrics, generated_lyrics, n=3,
                                                                  word2vec=word2vec_dict)
        cos_sim_5_gram = compute_cosine_similarity_between_ngrams(original_lyrics, generated_lyrics, n=5,
                                                                  word2vec=word2vec_dict)
        cos_sim_7_gram = compute_cosine_similarity_between_ngrams(original_lyrics, generated_lyrics, n=7,
                                                                  word2vec=word2vec_dict)

        bleu_score = calculate_bleu_score(original_lyrics, generated_lyrics)

        subjectivity, polarity = calculate_subjectivity_polarity(generated_lyrics)

        return {
            'loss val': self.val_loss,
            'cos_sim_no_order': cos_sim_no_order,
            'cos_sim_with_order':cos_sim_with_order,
            'cos_sim_3_gram': cos_sim_3_gram,
            'cos_sim_5_gram': cos_sim_5_gram,
            'cos_sim_7_gram': cos_sim_7_gram,
            "bleu score": bleu_score,
            "subjectivity" : subjectivity,
            "polarity": polarity
        }


