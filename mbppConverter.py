import json

with open("mbpp.jsonl", "r") as f_in, open("mbppProcessed.txt", "w") as f_out:
    for line in f_in:
        data = json.loads(line)
        # Extract just the text prompt, removing JSON formatting
        f_out.write(data["text"].replace("\n", " ") + "\n")  # Remove newlines if needed
