import openai
import spacy
import os
import time
import requests
from scipy.spatial import distance
import tiktoken

# Cost per token for the respective models (hypothetical costs, check OpenAI's pricing page)
TOKEN_COST_GPT_3_5 = 0.06
TOKEN_COST_ADA = 0.0001


def num_tokens_from_string(string: str, encoding_name: str) -> int:
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  return num_tokens


# Load spaCy's English NLP model for embeddings
nlp = spacy.load("en_core_web_md")


positive_sentiment_base = "Positive:  Joyful Elated Ecstatic Content Jubilant Optimistic Serene Euphoric Radiant Thrilled Positive Bliss Elation Jubilation Serenity Triumph Delight Exuberance Harmony Reverie Zenith"
negative_sentiment_base = "Negative:  Despondent Morose Disheartened Forlorn Melancholic Pessimistic Dismayed Frustrated Anguished Apprehensive Negative Despair Gloom Dismay Angst Malaise Turmoil Woe Heartbreak Affliction Abyss"

# Embed the words "positive" and "negative" using spaCy
positive_embedding_spacy = nlp(positive_sentiment_base).vector
negative_embedding_spacy = nlp(negative_sentiment_base).vector

# Set up the OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]


# Function to get embeddings using OpenAI's API
def get_ada_embedding(text):
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}"
  }
  data = {"input": text, "model": "text-embedding-ada-002"}
  response = requests.post("https://api.openai.com/v1/embeddings",
                           headers=headers,
                           json=data)
  return response.json()["data"][0]["embedding"]





# Get embeddings for "positive" and "negative" using ADA
positive_embedding_ada = get_ada_embedding(positive_sentiment_base)
negative_embedding_ada = get_ada_embedding(negative_sentiment_base)

# Example sentences
sentences = [
  "Tears streamed down her face as she read the letter from home.",
  "In the eerie silence of the haunted house, every creak echoed.",
  "His heart swelled with pride watching his daughter's graduation.",
  "The intensity of the storm outside mirrored her turbulent emotions.",
  "He sighed with relief as he found his lost keys in the couch.",
  "In the glow of the campfire, they exchanged stories of their childhood.",
  "The grim news on the radio cast a shadow over their celebrations.",
  "Her infectious laughter turned a mundane day into a memorable one.",
  "At the summit, overwhelmed by the view, he felt insignificantly small.",
  "The suspenseful climax of the movie had everyone on the edge of their seats.",
  "Lost in the maze of alleyways, panic began to set in.",
  "The first rays of sunrise brought hope to the weary travelers.",
  "A shiver ran down his spine as he recalled the ghostly tale.",
  "The festive atmosphere in the market was contagious, uplifting everyone's spirits.",
  "His eyes brimmed with unshed tears as he bid farewell to his family.",
  "The aroma of her grandmother's cookies evoked a flood of memories.",
  "Their argument escalated, filling the room with tension.",
  "In the tranquil meadow, she felt a deep connection to nature.",
  "The haunting melody of the song stirred something deep within him.",
  "The chaotic aftermath of the earthquake was a sight of devastation.",
  "With bated breath, they awaited the verdict of the competition.",
  "Amidst the ruins, he found a photo that spoke of happier times.",
  "The spontaneous dance in the rain rejuvenated her spirit.",
  "In the dark alley, the menacing shadow loomed closer.",
  "The comedian's joke had the entire hall erupting in laughter.",
  "She felt a twinge of envy seeing her friend's success.",
  "Amidst the hustle and bustle of the city, the park was an oasis.",
  "The weight of his guilt was palpable in the room.",
  "The beauty of the meteor shower left them in awestruck wonder.",
  "The news of the accident cast a pall of gloom over the village.",
  "His gentle reassurance dispelled her fears.",
  "The scandal sent shockwaves throughout the community.",
  "In the golden hue of sunset, the world seemed to pause.",
  "The emptiness of the house echoed her loneliness.",
  "His surprise proposal at the concert was met with cheers and applause.",
  "The foreboding forest hid secrets in its shadows.",
  "Her joy at reuniting with her long-lost friend was palpable.",
  "The tense standoff at the border was a geopolitical powder keg.",
  "The gentle hum of the bees was a lullaby to her afternoon nap.",
  "In the heat of the moment, words were exchanged that couldn't be taken back.",
  "The unveiling of the monument was a moment of national pride.",
  "As the ship sank, a somber silence gripped the onlookers.",
  "She blushed as the room burst into applause for her performance.",
  "The inexplicable events at the mansion were the talk of the town.",
  "The joyride took a dark turn as the storm clouds gathered.",
  "In the glow of the lanterns, the festival came alive.",
  "The sudden betrayal was a bitter pill to swallow.",
  "The symphony of nature at dawn was pure magic.",
  "His anxiety was evident as he paced the waiting room.",
  "The mural spoke of a bygone era, filled with romance and rebellion."
]

results = []

for sentence in sentences:
  print(sentence)
  start_time = time.time()

  # Classification using ChatCompletion API
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{
      "role":
      "system",
      "content":
      "You are a helpful assistant that only responds as 'positive' or 'negative'."
    }, {
      "role":
      "user",
      "content":
      f"Is the following sentence positive or negative (options:positive/negative)?\n\n{sentence}"
    }])
  davinci_duration = time.time() - start_time
  davinci_classification = response.choices[0].message['content'].strip(
  ).lower()
  print(davinci_classification)

  start_time = time.time()
  # Classification using spaCy embeddings
  sentence_embedding = nlp(sentence).vector
  pos_distance_spacy = distance.cosine(sentence_embedding, positive_embedding_spacy)
  neg_distance_spacy = distance.cosine(sentence_embedding, negative_embedding_spacy)
  embedding_duration_spacy = time.time() - start_time
  if pos_distance_spacy < neg_distance_spacy:
    embedding_classification_spacy = "positive"
  else:
    embedding_classification_spacy = "negative"
  print(embedding_classification_spacy)
  # Classification using ADA embeddings
  start_time = time.time()
  sentence_embedding_ada = get_ada_embedding(sentence)
  pos_distance_ada = distance.cosine(sentence_embedding_ada, positive_embedding_ada)
  neg_distance_ada = distance.cosine(sentence_embedding_ada, negative_embedding_ada)
  embedding_duration_ada = time.time() - start_time
  if pos_distance_ada < neg_distance_ada:
    embedding_classification_ada = "positive"
  else:
    embedding_classification_ada = "negative"
  print(embedding_classification_ada)

  tokens_gpt_3_5 = num_tokens_from_string(sentence, "cl100k_base")
  tokens_ada = num_tokens_from_string(sentence,"cl100k_base")  # Assuming the same tokenization method

  cost_gpt_3_5 = tokens_gpt_3_5 * TOKEN_COST_GPT_3_5
  cost_ada = tokens_ada * TOKEN_COST_ADA

  # Storing results
  results.append(
      (davinci_classification, davinci_duration, tokens_gpt_3_5, cost_gpt_3_5,
       embedding_classification_spacy, pos_distance_spacy, neg_distance_spacy, embedding_duration_spacy,
       embedding_classification_ada, pos_distance_ada, neg_distance_ada, embedding_duration_ada, tokens_ada, cost_ada)
  )

# Print results
header = "| Sentence | gpt-3.5 Class. | gpt-3.5 Time | gpt-3.5 Tokens | gpt-3.5 Cost | spaCy Class. | spaCy Dist. Pos. | spaCy Dist. Neg. | spaCy Time | ada-002 Class. | ada Dist. Pos. | ada Dist. Neg. | ada-002 Time | ada-002 Tokens | ada-002 Cost |"
separator = "|----------|---------------|--------------|---------------|--------------|--------------|-----------------|-----------------|------------|----------------|----------------|----------------|--------------|---------------|--------------|"

print("")
print(header)
print(separator)
for i, (davinci_classification, davinci_duration, davinci_tokens, cost_gpt_3_5,
        embedding_classification_spacy, pos_distance_spacy, neg_distance_spacy, embedding_duration_spacy,
        embedding_classification_ada, pos_distance_ada, neg_distance_ada, embedding_duration_ada, ada_tokens, cost_ada) in enumerate(results):

    davinci_color = '\033[92m' if 'positive' in davinci_classification else '\033[91m'
    spacy_color = '\033[92m' if 'positive' in embedding_classification_spacy else '\033[91m'
    ada_color = '\033[92m' if 'positive' in embedding_classification_ada else '\033[91m'

    print(
        f"| {sentences[i][:30]}.. | {davinci_color}{davinci_classification}\033[0m | {davinci_duration:.2f} sec | {davinci_tokens} tokens | ${cost_gpt_3_5:.2f} | {spacy_color}{embedding_classification_spacy}\033[0m | {pos_distance_spacy:.3f} | {neg_distance_spacy:.3f} | {embedding_duration_spacy:.2f} sec | {ada_color}{embedding_classification_ada}\033[0m | {pos_distance_ada:.3f} | {neg_distance_ada:.3f} | {embedding_duration_ada:.2f} sec | {ada_tokens} tokens | ${cost_ada:.2f} |"
    )

# Average results
avg_davinci_time = sum([res[1] for res in results]) / len(results)
avg_spacy_time = sum([res[7] for res in results]) / len(results)
avg_ada_time = sum([res[11] for res in results]) / len(results)
print("")
# Average costs
avg_gpt_3_5_cost = sum([res[3] for res in results]) / len(results)
avg_ada_cost = sum([res[13] for res in results]) / len(results)

# ...

print("\nAverage gpt-3.5 Time:", avg_davinci_time)
print("Average gpt-3.5 Cost: $",avg_gpt_3_5_cost)
print("Average spaCy Time:", avg_spacy_time)
print("Average ada-002 Time:", avg_ada_time)
print("Average ada-002 Cost: $",avg_ada_cost)
