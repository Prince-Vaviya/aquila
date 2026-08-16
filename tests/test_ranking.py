from src.tokenizer import tokenize
from src.index import InvertedIndex
from src.search import retrieve_candidates
from src.ranking import TFIDFRanker

index = InvertedIndex()
ranker = TFIDFRanker(index, 6)

for i in range(1, 7):
    with open(f"data/{i}.txt", "r") as content:
        file_contents = content.read()

    index.add_documents(i, tokenize(file_contents))

for query in ["energy", "solar", "solar energy", "human brain", "network protocol" ,"xyz"]:
    candidates = retrieve_candidates(query, index)
    results = ranker.rank(query, candidates)
    print(query, " ---> ", results)