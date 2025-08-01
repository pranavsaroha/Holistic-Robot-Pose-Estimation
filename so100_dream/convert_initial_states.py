input_file = "initial-episode-states.csv"
output_file = "initial-episode-states-converted.csv"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    next(fin)  # skip header
    for line in fin:
        cleaned = line.strip().replace('[', '').replace(']', '').replace('"', '').replace(', ', ',')
        if cleaned:
            fout.write(cleaned + '\n')