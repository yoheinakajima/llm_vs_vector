#This is a variation for testing multiple options for classification, using movies and genres as an example. It's only gpt-3.5 and ada-002 for now

import openai
import time
import requests
import tiktoken
import numpy as np

# Cost per token for the respective models (hypothetical costs, check OpenAI's pricing page)
TOKEN_COST_GPT_3_5 = 0.002
TOKEN_COST_ADA = 0.0001

openai.api_key = os.environ["OPENAI_API_KEY"]


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def cosine_similarity(A, B):
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    return dot / (norma * normb)


def get_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    data = {"input": text, "model": "text-embedding-ada-002"}
    response = requests.post("https://api.openai.com/v1/embeddings",
                             headers=headers,
                             json=data)
    return np.array(response.json()["data"][0]["embedding"])


genres = ["action film", "comedy film", "romance film", "horror film", "sci-fi film","drama film"]
genre_embeddings = {genre: get_embedding(genre) for genre in genres}

# Example movies and their descriptions
movies = {
    "Die Hard": "An NYPD officer tries to save his wife and several others taken hostage by German terrorists during a Christmas party.",
    "When Harry Met Sally": "Two good friends with opposite relationship problems find themselves single at the same time.",
    "Alien": "The crew of a spaceship encounter a deadly alien which begins to kill them off one by one.",
    "The Shawshank Redemption": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
    "The Godfather": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
    "The Dark Knight": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
    "Pulp Fiction": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
    "Forrest Gump": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.",
    "Inception": "A thief who enters the dreams of others to steal ideas tries to have the perfect crime by implanting an idea into someone's mind.",
    "Titanic": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
    "Gladiator": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.",
    "Jurassic Park": "A wealthy entrepreneur secretly creates a theme park featuring living dinosaurs drawn from prehistoric DNA.",
    "The Matrix": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
    "Toy Story": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
    "The Lion King": "A young lion prince flees his kingdom only to learn the true meaning of responsibility and bravery.",
    "Schindler's List": "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution by the Nazis.",
    "Star Wars: A New Hope": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire's world-destroying battle station, while also attempting to rescue Princess Leia from the mysterious Darth Vader.",
    "Casablanca": "A cynical American expatriate struggles to decide whether or not he should help his former lover and her fugitive husband escape French Morocco.",
    "The Silence of the Lambs": "A young FBI cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims.",
    "Avatar": "On the lush alien world of Pandora live the Na'vi, beings who appear primitive but are highly evolved. A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.",
    "Saving Private Ryan": "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action.",
    "Jaws": "A giant great white shark arrives on the shores of a New England beach resort and wreaks havoc with bloody attacks on swimmers.",
    "La La Land": "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.",
    "Slumdog Millionaire": "A Mumbai teenager reflects on his life after being accused of cheating on the Indian version of 'Who Wants to be a Millionaire?'.",
    "Terminator 2: Judgment Day": "A cyborg, identical to the one who failed to kill Sarah Connor, must now protect her ten-year-old son, John, from a more advanced and powerful cyborg."
}

results = []

for movie, description in movies.items():
    print(movie+": "+description)

    # Classification using gpt-3.5-turbo
    start_time = time.time()
    genre_options = ', '.join(genres[:-1]) + ' or ' + genres[-1]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": f"You are a helpful assistant that classifies movie descriptions into genres: {genre_options}."
        }, {
            "role": "user",
            "content": f"Classify the following movie description into a genre ({genre_options}), and only respond with the genre: {description}"
        }])
    davinci_duration = time.time() - start_time
    davinci_classification = response.choices[0].message['content'].strip().lower()

    # Classification using ADA embeddings
    start_time = time.time()
    description_embedding = get_embedding(description)
    similarities = {genre: cosine_similarity(description_embedding, embedding) for genre, embedding in genre_embeddings.items()}
    embedding_classification = max(similarities, key=similarities.get)
    embedding_duration = time.time() - start_time

    tokens_gpt_3_5 = num_tokens_from_string(description, "cl100k_base")
    tokens_ada = tokens_gpt_3_5  # Assuming the same tokenization method for simplicity

    cost_gpt_3_5 = tokens_gpt_3_5 * TOKEN_COST_GPT_3_5
    cost_ada = tokens_ada * TOKEN_COST_ADA

    # Storing results
    results.append((davinci_classification, davinci_duration, tokens_gpt_3_5, cost_gpt_3_5, embedding_classification, embedding_duration, tokens_ada, cost_ada))

# Define some ANSI escape codes for colors
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'

# Display results
header = f"| {Colors.YELLOW}Movie{Colors.END} | {Colors.GREEN}gpt-3.5 Genre{Colors.END} | {Colors.GREEN}gpt-3.5 Time{Colors.END} | {Colors.GREEN}gpt-3.5 Tokens{Colors.END} | {Colors.GREEN}gpt-3.5 Cost{Colors.END} | {Colors.BLUE}ada-002 Genre{Colors.END} | {Colors.BLUE}ada-002 Time{Colors.END} | {Colors.BLUE}ada-002 Tokens{Colors.END} | {Colors.BLUE}ada-002 Cost{Colors.END} |"
separator = "|-------|--------------|--------------|---------------|--------------|--------------|--------------|---------------|--------------|"
print()
print(header)
print(separator)
for i, (davinci_classification, davinci_duration, davinci_tokens, cost_gpt_3_5, embedding_classification, embedding_duration, ada_tokens, cost_ada) in enumerate(results):
    print(f"| {Colors.YELLOW}{list(movies.keys())[i]}{Colors.END} | {Colors.GREEN}{davinci_classification}{Colors.END} | {Colors.GREEN}{davinci_duration:.2f} sec{Colors.END} | {Colors.GREEN}{davinci_tokens} tokens{Colors.END} | {Colors.GREEN}${cost_gpt_3_5:.2f}{Colors.END} | {Colors.BLUE}{embedding_classification}{Colors.END} | {Colors.BLUE}{embedding_duration:.2f} sec{Colors.END} | {Colors.BLUE}{ada_tokens} tokens{Colors.END} | {Colors.BLUE}${cost_ada:.2f}{Colors.END} |")


avg_davinci_time = sum([res[1] for res in results]) / len(results)
avg_ada_time = sum([res[5] for res in results]) / len(results)
avg_gpt_3_5_cost = sum([res[3] for res in results]) / len(results)
avg_ada_cost = sum([res[7] for res in results]) / len(results)

# Colorize average results:
print(f"\n{Colors.GREEN}Average gpt-3.5 Time:{Colors.END}", avg_davinci_time)
print(f"{Colors.GREEN}Average gpt-3.5 Cost:{Colors.END} $", avg_gpt_3_5_cost)
print(f"{Colors.BLUE}Average ada-002 Time:{Colors.END}", avg_ada_time)
print(f"{Colors.BLUE}Average ada-002 Cost:{Colors.END} $", avg_ada_cost)
