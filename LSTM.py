import os
import glob
import torch
import numpy as np
from torch import device
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import time
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import corpus_bleu
from joblib import Parallel, delayed

#posebni tokeni koji oznacavaju pocetak generisanja naslova, sastojaka i instrukcija
UNIQUE_TOKEN_TITLE = "<TIT>"
UNIQUE_TOKEN_INGREDIENTS = "<ING>"
UNIQUE_TOKEN_INSTRUCTIONS = "<INS>"
UNIQUE_TOKEN_END = "*" 
MAX_RECIPE_LEN = 2000   #maximalna duzina recepta koju cemo uzeti -> dobija se preko matplotliba

class RecipeDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx][:-1]), torch.tensor(self.sequences[idx][1:])
    
# Model Definition
class RecipeLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(RecipeLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.4)  # Add dropout layer
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out


#spaja sve u listu[dict{title:string, ingredients:[string, string..], instructions:string}]
def process_json(dict): 
    ingredients_list = []
    for ingredient in dict['ingredients']:
        ingredients_list.append(ingredient['text'])
    
    dict['ingredients'] = ingredients_list

    manual = ""
    for instruction in dict['instructions']:
        manual += instruction['text'] + "\n"
    
    dict['instructions'] = manual[:-1]

#otvara .json fajl, ucita iz fajla, preprocesuje svaki recept iz fajla [process_json] i za kraj sve to strpa u listu 
def load_dataset():
    recipes_dataset = []

    with open('data/recipes-s.json', 'r') as f:
        recipes_list = json.load(f)
        recipes_keys = [key for key in recipes_list[0]]
        recipes_keys.sort()

        for recipe in recipes_list:
            process_json(recipe)
            recipes_dataset.append(recipe)
    
    return recipes_dataset

#pretvara recept u jedinstven string kako bi to sve trebalo otprilike da izgleda a pritom i uvodi u taj string posebne tokene od gore
def merge_recipe_string(recipe):    #TODO: OVU FUNKCIJU POSLE AKO MOZES MALO SAMO OVAJ FORMATI IZMENI I ZBOG PLAGIJATA I INACE ODVRATNO JE
    title_format = recipe['title']
    ingredients = recipe['ingredients']
    instructions = recipe['instructions'].split('\n')

    ingredients_format = ""
    for ingredient in ingredients:
        ingredients_format += " - " + ingredient + "\n"
    
    instruction_format = ""
    for instruction in instructions:
        instruction_format += " - " + instruction + "\n"
    
    return UNIQUE_TOKEN_TITLE + " " + title_format + "\n\n" + UNIQUE_TOKEN_INGREDIENTS + "\n\n" + ingredients_format + "\n" + UNIQUE_TOKEN_INSTRUCTIONS + "\n\n" + instruction_format

#filtrira iz dataseta recepte (prethodno su pretvoreni u 1 string) koji su predugacki i vraca [stringRecept, stringRecept, stringRecept]
def filter_long_recepies(dataset):
    filtered_dataset = []
    for recipe in dataset:
        if len(recipe) <= MAX_RECIPE_LEN:
            filtered_dataset.append(recipe)
    
    return filtered_dataset

#svaki recept ceo(postali su stringovi) splitujem po razmaku i vratim jednu veliku listu tokena koja ce tokenizovati sve recepte
def tokenize_recepies(dataset):
    tokens = []
    for recipe in dataset:
        words = recipe.split()
        recipe_tokens = tokenize(words)
        tokens.extend(recipe_tokens)
    return tokens

#pomocna fja koja prima recnik koji se splitovao pre toga i treba unutra da handluje manjak ' ', '\n' i ostale slucajeve
def tokenize(recipe_words):
    tokens = []
    for i in range(len(recipe_words)):
        if recipe_words[i] == "-":
            tokens.pop(-1)
            tokens.append("\n")
            tokens.append(" ")
        elif recipe_words[i] == "<ING>" or recipe_words[i] == "<INS>":
            tokens.pop(-1)
            tokens.append("\n")
            tokens.append(recipe_words[i])
            tokens.append("\n")
            tokens.append(" ")
            continue
        
        tokens.append(recipe_words[i])
        tokens.append(" ")
    tokens.pop(-1)
    return tokens

#pravi se recnik {'char': broj ponavljanja, 'char': broj ponavljanja} [br ponavljanja u celom dokumentu]
def build_vocab(dataset):
    tokens = tokenize_recepies(dataset)
    counter = Counter()
    for word in tokens:
        counter.update(word)
    counter.update(UNIQUE_TOKEN_END)
    return {word: idx for idx, (word, _) in enumerate(counter.items(), 1)} 

#uzece recept i pretvorice ga uz recnik u [1 ,5, 57...]
def translate_to_nums(recipe, vocabulary):
    sequnce = []
    for char in recipe:
        if char in vocabulary:
            sequnce.append(vocabulary[char])
    return sequnce

#uzece listu integera i uz pomoc recnika vratiti ga u string tekst
def translate_to_chars(sequence, vocabulary):
    vocabulary_inverse = {index: word for word, index in vocabulary.items()}
    recipe = ""
    for num in sequence:
        if num in vocabulary_inverse:
            recipe += vocabulary_inverse[num]
    return recipe

#uzece recept koji je vektorizovanog oblika i na njegov kraj dodati UNIQUE_END_TOKEN dok ne dopuni do neke duzine
def pad_vectorized_recepies(dataset_vectorized, vocabulary):
    end_token_int = vocabulary[UNIQUE_TOKEN_END]
    for recipe in dataset_vectorized:
        while len(recipe) < MAX_RECIPE_LEN + 1:
            recipe.append(end_token_int)
    return dataset_vectorized

#uz pomoc numpy stvara sekvence tipa Dusa -> usan
def make_sequences_np(translated_recipes, sequence_len=100):
    sequences = []
    for ints in translated_recipes:
        ints_array = np.array(ints)
        num_sequences = len(ints) - sequence_len
        if num_sequences > 0:
            sequences.extend([ints_array[i:i + sequence_len + 1] for i in range(num_sequences)])
    return sequences

#HUMANS:
#tokenizuj input ljudski (izdeli ga i dodaj razmake sta sve treba)
def preprocess_human_input(input):
    tokens = input.split()
    tokens = tokenize(tokens)
    return tokens

#fja koja priprema korisnicki input da ide u model
def handle_user_input(input, vocabulary):
    input_tokens = preprocess_human_input(input)
    translated_input = translate_to_nums(input_tokens, vocabulary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.tensor(translated_input).unsqueeze(0).to(device)
    return inputs

#fja koja dopunjava svaki deo batcha [input i target] do iste duzine
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded.long() 

#fja za evaluiranje
def evaluate(model, dataloader, criterion, vocabulary_size):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)

            output = output.view(-1, output.size(2))
            targets = targets.view(-1) 
                     
            loss = criterion(output, targets)
            total_loss += loss.item()

            _, predicted = torch.max(output, dim=1)  # Get index of maximum value as predicted class
            total_correct += (predicted == targets).sum().item()
            total_examples += targets.numel()

    accuracy = total_correct / total_examples
    return total_loss / len(dataloader), accuracy

#fja za treniranje
def train(model, dataloader, criterion, optimizer, epoch, vocabulary_size):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    start_time = time.time()

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        output = model(inputs)      #stvori [64, 2] iz modela, tehnicki provuci kroz model za predict


        # Reshape output and targets if needed
        output = output.view(-1, output.size(2))
        targets = targets.view(-1)            # Flatten targets if needed

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(output, 1)  # Get index of maximum value as predicted class
        correct = (predicted == targets).sum().item()
        total_correct += correct
        total_examples += targets.size(0)  # Total number of examples in this batch

    stop_time = time.time()

    print(f"Training epoch {epoch}, Loss: {total_loss / len(dataloader):.2f}, Accuracy: {total_correct / total_examples:.2f}")
    print(f"Training epoch lasted: {stop_time - start_time:.2f} seconds")

#fja za generisanje recepta
def generate_recipe(model, initial_sequences, start_text, max_length, vocabulary, temperature=0.7):
    model.eval()
    vocabulary_inverse = {index: word for word, index in vocabulary.items()}
    generated_text = "New recipe:\n" + start_text

    with torch.no_grad():
        for _ in range(max_length):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Forward pass through the model
            output = model(initial_sequences)
            output = output[:, -1, :].squeeze()  

            #Applying temperature
            output = output / temperature
            probabilities = torch.softmax(output, dim=-1)
            top_idx = torch.multinomial(probabilities, 1).item()

            word = vocabulary_inverse.get(top_idx, '<UNK>')
            generated_text += word


            initial_sequences = torch.cat((initial_sequences, torch.tensor([[top_idx]]).to(device)), dim=1)

    return generated_text




def GENERATE(filename, start_text, max_len, temperature):
    dataset = load_dataset()
    dataset = [merge_recipe_string(recipe) for recipe in dataset]
    # print("=========================EXAMPLE========================")
    # print(dataset[0])
    # print("===========================================================")
    dataset = filter_long_recepies(dataset)
    # print("dataset len: ",len(dataset))

    # print("data loaded successfuly")

    vocabulary = build_vocab(dataset)
    
    model = torch.load("lstm_models/" + filename + ".pth")

    user_sequences = handle_user_input(start_text, vocabulary)
    generated_text = generate_recipe(model, user_sequences, start_text, max_len, vocabulary, temperature=temperature)
    print(generated_text)
    return generated_text

def TRAIN(filename):
      
    dataset = load_dataset()
    dataset = [merge_recipe_string(recipe) for recipe in dataset]
    dataset = filter_long_recepies(dataset)

    # print("data loaded successfuly")

    vocabulary = build_vocab(dataset)

    # print("vocabulary has been built")

    dataset_translated = [translate_to_nums(recipe, vocabulary) for recipe in dataset]

    # print("recipes translated")

    dataset_translated_sequences = make_sequences_np(dataset_translated)

    # print("sequences created")

    train_data, test_data = train_test_split(dataset_translated_sequences, test_size=0.2)
    train_data, val_data = train_test_split(train_data, test_size=0.2)

    # print("data splitted")

    dataset_train = RecipeDataset(train_data)
    dataset_val = RecipeDataset(val_data)
    dataset_test = RecipeDataset(test_data)

    # print("datasets created")

    train_batches = DataLoader(dataset_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_batches = DataLoader(dataset_val, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_batches = DataLoader(dataset_test, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # print("dataloaders created")

    print(f"Train size: {len(dataset_train)}, Val size: {len(dataset_val)}, Test size: {len(dataset_test)}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary_size = len(vocabulary) + 1
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2

    model = RecipeLSTM(vocabulary_size, embedding_dim, hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("training and evaluating on val dataset is beggining....")
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, train_batches, criterion, optimizer, epoch, vocabulary_size)
        val_loss, val_accuracy = evaluate(model, val_batches, criterion, vocabulary_size)
        print(f"Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}")
    
    torch.save(model, "lstm_models/" + filename + '.pth')
    print("model saved")
    

if __name__ == "__main__":

    dataset = load_dataset()
    dataset = [merge_recipe_string(recipe) for recipe in dataset]
    print("=========================EXAMPLE========================")
    print(dataset[0])
    print("===========================================================")
    dataset = filter_long_recepies(dataset)
    print("dataset len: ",len(dataset))

    print("data loaded successfuly")

    vocabulary = build_vocab(dataset)
    
    model = torch.load('lstm_models/lstm_1.pth')

    user_sequences = handle_user_input("<TIT> \n C", vocabulary)
    # print(len(user_tokens))
    print(generate_recipe(model, user_sequences, "<TIT> C", 500))


# if __name__ == "__main__":

#     if torch.cuda.is_available():
#         print("CUDA is available.")
#         print(f"CUDA version: {torch.version.cuda}")
#         print(f"Number of GPUs: {torch.cuda.device_count()}")
#         print(f"Current GPU: {torch.cuda.current_device()}")
#         print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
#     else:
#         print("CUDA is not available.")

#     dataset = load_dataset()
#     dataset = [merge_recipe_string(recipe) for recipe in dataset]
#     dataset = filter_long_recepies(dataset)
#     print("dataset len: ",len(dataset))

#     print("data loaded successfuly")

#     vocabulary = build_vocab(dataset)

#     print("vocabulary has been built")

#     dataset_translated = [translate_to_nums(recipe, vocabulary) for recipe in dataset]
#     print("dataset translated len:", len(dataset_translated))

#     print("recipes translated")

#     dataset_translated_padded = pad_vectorized_recepies(dataset_translated, vocabulary)

#     print("recepies padded")

#     dataset_translated_sequences = make_sequences_np(dataset_translated)
#     print("dataset translated sequences len: ", len(dataset_translated_sequences))

#     print("sequences created")

#     train_data, test_data = train_test_split(dataset_translated_sequences, test_size=0.2)
#     train_data, val_data = train_test_split(train_data, test_size=0.2)

#     print("data splitted")

#     dataset_train = RecipeDataset(train_data)
#     dataset_val = RecipeDataset(val_data)
#     dataset_test = RecipeDataset(test_data)

#     print("datasets created")

#     train_batches = DataLoader(dataset_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
#     val_batches = DataLoader(dataset_val, batch_size=64, shuffle=False, collate_fn=collate_fn)
#     test_batches = DataLoader(dataset_test, batch_size=64, shuffle=False, collate_fn=collate_fn)

#     print("dataloaders created")

#     print(f"Train size: {len(dataset_train)}, Val size: {len(dataset_val)}, Test size: {len(dataset_test)}")


#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     vocabulary_size = len(vocabulary) + 1
#     embedding_dim = 128
#     hidden_dim = 256
#     num_layers = 2

#     model = RecipeLSTM(vocabulary_size, embedding_dim, hidden_dim, num_layers).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     print("training and evaluating on val dataset is beggining....")
#     num_epochs = 10
#     for epoch in range(1, num_epochs + 1):
#         train(model, train_batches, criterion, optimizer, epoch, vocabulary_size)
#         val_loss, val_accuracy = evaluate(model, val_batches, criterion, vocabulary_size)
#         print(f"Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}")
    

#     torch.save(model, 'recipe_lstm_model_extra_tokens.pth')





# def create_datasets(sequences_dict):
#     train_dataset = RecipeDataset(sequences_dict['train'])
#     test_dataset = RecipeDataset(sequences_dict['test'])
#     val_dataset = RecipeDataset(sequences_dict['val'])

#     dataset_dict = {
#         'train': train_dataset,
#         'test': test_dataset,
#         'val': val_dataset
#     }

#     return dataset_dict

# def make_sequences_parallel(translated_recipes, sequence_len=100, n_jobs=-1):
#     sequences = Parallel(n_jobs=n_jobs)(delayed(make_sequences)(ints, sequence_len) for ints in translated_recipes)
#     # Flatten the list of lists
#     return np.array([seq for sublist in sequences for seq in sublist])


# def make_sequences(translated_recipes, sequence_len = 100):
#     sequences = []
#     for ints in translated_recipes:
#         for i in range(0, len(ints) - sequence_len):
#             sequences.append(ints[i:i + sequence_len + 1])
#     return sequences

# def sequence_padding(batch):
#     # Separate inputs and targets from the batch
#     inputs = [item[0] for item in batch]
#     targets = [item[1] for item in batch]
    
#     # Pad sequences to the same length
#     inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
#     targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
#     return inputs, targets

# def create_dataloaders(dataset_dict):
#     train_loader = DataLoader(dataset_dict['train'], batch_size=64, shuffle=True)  
#     test_loader = DataLoader(dataset_dict['test'], batch_size=64, shuffle=False)
#     val_loader = DataLoader(dataset_dict['val'], batch_size=64, shuffle=False)

#     dataloader_dict = {
#         'train': train_loader,
#         'test': test_loader,
#         'val': val_loader
#     }

#     return dataloader_dict