import random
import string

def random_word(min_len=3, max_len=8):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_letters, k=length))

def generate_sentence():
    base_sentences = [
        "Once upon a time in a quiet village nestled between rolling hills and lush forests there lived a young girl named Elara",
        "She was known for her curiosity and kindness always eager to explore the world beyond her small home",
        "One day while wandering near the edge of the forest Elara discovered an ancient hidden path that seemed to glow faintly under the dappled sunlight",
        "Intrigued she followed the path deeper into the woods where she encountered a wise old owl perched on a twisted branch",
        "The owl spoke in riddles hinting at a secret treasure buried beneath the roots of the oldest tree in the forest",
        "Determined to uncover the mystery Elara embarked on a journey filled with challenges and new friendships",
        "Along the way she met a clever fox who guided her through tricky terrain a gentle deer who shared stories of the forests past and a mischievous squirrel who taught her the value of laughter",
        "Together they faced storms crossed rivers and climbed steep hills each step bringing Elara closer to the treasure and to understanding the true magic of the forest",
        "In the end the treasure was not gold or jewels but the bonds of friendship and the wisdom gained from the journey",
        "Elara returned to her village her heart full of stories and her spirit forever changed by the adventure"
    ]
    sentence = random.choice(base_sentences)
    words = sentence.split()
    insert_pos = random.randint(0, len(words))
    words.insert(insert_pos, random_word())
    return ' '.join(words)

lines = [generate_sentence() for _ in range(20000)]

with open("alphabetical_dataset.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")
