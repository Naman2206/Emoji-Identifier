import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class EmojiIdentifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self.api_url = "https://emoji-api.com/emojis?access_key=37f2c6d7cf98f794d5076de509b52344084dd6b8"

    def train(self, texts, emoji_labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, emoji_labels)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        predicted_label = self.model.predict(X)[0]
        return self.get_emoji_from_api(predicted_label)

    def get_emoji_from_api(self, text):
        params = {
            "search": text,
            "access_key": '37f2c6d7cf98f794d5076de509b52344084dd6b8'  # Replace with your actual API key
        }
        response = requests.get(self.api_url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            # Check if the response contains data and has the expected structure
            if data and isinstance(data, list) and len(data) > 0 and "character" in data[0]:
                return data[0]["character"]
            else:
                print(f"Warning: No emoji found for '{text}' or unexpected API response format.")
                # You can add a fallback here, e.g., return a default emoji
                return "❓" 
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            # You can add a fallback here, e.g., return a default emoji
            return "❗"

# Example usage
texts = [
    "I love this new feature! :)", 
    "Aw, that's too bad. :(",
    "Haha, that's so funny! :D"
]
emoji_labels = [':)', ':(',':D']

identifier = EmojiIdentifier()
identifier.train(texts, emoji_labels)

print(identifier.predict("I'm feeling happy today!")) 
print(identifier.predict("This is disappointing."))   
print(identifier.predict("That was hilarious!"))
