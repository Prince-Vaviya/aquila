from src.tokenizer import tokenize
from src.index import InvertedIndex
from src.search import retrieve_candidates

index = InvertedIndex()

for i in range(1, 7):
    with open(f"data/{i}.txt", "r") as content:
        file_contents = content.read()

    index.add_documents(i, tokenize(file_contents))

for query in ["energy", "solar", "solar energy", "human brain", "network protocol" ,"xyz"]:
    print(retrieve_candidates(query, index))