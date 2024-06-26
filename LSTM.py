import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import consts as const

# Priprema podataka za LSTM
# - word_to_index_dict treba da se dobije iz vektorizera, ona imena sto je on vec izvukao treba prosediti ovde i onda mi tehnicki
# vec imamo odradjen recnik za taj tekst

def prepare_sequences(texts, word_to_index_dict):
    sequences = []
    for text in texts:
        sequence = [word_to_index_dict.get(word, 0) for word in text.split()]  
        sequence = sequence[:const.LSTM_max_text_len] + [0]*(const.LSTM_max_text_len - len(sequence))
        sequences.append(sequence)
    return np.array(sequences)

def LSTM_data_prep(relevant_data_dict, vectorizer):
    #----------------
    word_to_index = {word: i+1 for i, word in enumerate(vectorizer.get_feature_names_out())}    #ovo treba da se dobavi od spolja!!
                                                                                                #[preko recnika]

    #------------------
    # ova 3 dela za data treba izvuci iz loader.py -> ili proslediti kako god   [napravio relevant_columns_dict]
    X_train_seq = prepare_sequences(relevant_data_dict['train_data_relevant'].values, word_to_index, const.LSTM_max_text_len)
    X_val_seq = prepare_sequences(relevant_data_dict['val_data_relevant'].values, word_to_index, const.LSTM_max_text_len)
    X_test_seq = prepare_sequences(relevant_data_dict['test_data_relevant'].values, word_to_index, const.LSTM_max_text_len)

    #-------
    #ovo ima dve verzije jednu sa y drugu bez nje...zavisi da li cemo 'ciljnu vrednost dati'[nadgledano ucenje]/'necemo dati'[nenadgledano]
    #po mojoj logici ovo dole gde koristimo nenadgledano ucenje moze ali onda mi ni na jedan nacin ne spajamo lstm sa cnb i rf, sto mi
    #nikako nema logike....
    # train_data = TensorDataset(torch.tensor(X_train_seq, dtype=torch.long), torch.tensor(y_train.values, dtype=torch.long))
    # val_data = TensorDataset(torch.tensor(X_val_seq, dtype=torch.long), torch.tensor(y_val.values, dtype=torch.long))
    # test_data = TensorDataset(torch.tensor(X_test_seq, dtype=torch.long), torch.tensor(y_test.values, dtype=torch.long))

    #zasada koristi ovo ali proveri sa Lukom
    train_data = TensorDataset(torch.tensor(X_train_seq, dtype=torch.long))
    val_data = TensorDataset(torch.tensor(X_val_seq, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(X_test_seq, dtype=torch.long))

    #---------------------

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    LSTM_loader_dict = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'word_to_index': word_to_index
    }
    return LSTM_loader_dict





# Definicija LSTM modela
class RecipeLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RecipeLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out


def createLSTM(LSTM_loader_dict):
    vocab_size = len(LSTM_loader_dict['word_to_index']) + 1
    embedding_dim = 128
    hidden_dim = 256
    # output_dim = len(data['Recipe Name'].unique())

    model = RecipeLSTM(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    lstm_dict = {
        "lstm": model,
        "criterion": criterion,
        "optimizer": optimizer
    }
    return lstm_dict


def trainLSTM(model, optimizer, criterion, LSTM_loader_dict):
    # Treniranje LSTM modela
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for inputs, labels in LSTM_loader_dict['train_loader']:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in LSTM_loader_dict['val_loader']:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss/len(LSTM_loader_dict['val_loader'])}')

# # Evaluacija LSTM modela
# model.eval()
# y_val_pred = []
# with torch.no_grad():
#     for inputs, _ in val_loader:
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         y_val_pred.extend(preds.tolist())

# print('LSTM Accuracy:', accuracy_score(y_val, y_val_pred))
# print('LSTM F1 Score:', f1_score(y_val, y_val_pred, average='weighted'))
# print('LSTM Confusion Matrix:\n', confusion_matrix(y_val, y_val_pred))


# Generisanje novih recepata [namesti samo da prima dict umesto konkretnih stvari, da bi onda iz dict vadio u fji]
def generate_recipe(model, start_text, word_to_index, index_to_word, max_len):
    model.eval()
    words = start_text.split()
    state_h, state_c = model.lstm.init_state(len(words))
    
    for _ in range(max_len - len(words)):
        x = torch.tensor([[word_to_index.get(w, 0) for w in words[-max_len:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(index_to_word[word_index])
    
    return ' '.join(words)

