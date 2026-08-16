from src.tokenizer import tokenize

for i in range(1, 7):
    with open(f"data/{i}.txt", "r") as content:
        file_content = content.read()
    print("\n")
    print(tokenize(file_content), "\n")