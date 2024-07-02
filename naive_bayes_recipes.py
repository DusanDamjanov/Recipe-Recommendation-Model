import json
import pandas as pd

# 1. Load JSON data
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 2. Extract features and target values
def extract_data(data):
    ingredients = [' '.join(item['text'] for item in recipe['ingredients']) for recipe in data]
    instructions = [' '.join(step['text'] for step in recipe['instructions']) for recipe in data]
    levels = [recipe['level'] for recipe in data]

    print("ingredients example: ", ingredients[0])
    print("instructions example: ", instructions[0])
    print("level example: ", levels[0])
    
    df = pd.DataFrame({
        'ingredients': ingredients,
        'instructions': instructions,
        'level': levels
    })
    return df

if __name__ == "__main__":
    data = load_json_file("data/modified.json")
    df = extract_data(data)