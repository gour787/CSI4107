# src/convert_qrels.py
import csv
import sys

if len(sys.argv) != 3:
    print("Usage: python convert_qrels.py <test.tsv> <qrels.txt>")
    sys.exit(1)

in_path = sys.argv[1]
out_path = sys.argv[2]

with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
    reader = csv.reader(f_in, delimiter="\t")
    header = next(reader, None)  # skip header
    for row in reader:
        if len(row) < 3:
            continue
        qid, docid, rel = row[0].strip(), row[1].strip(), row[2].strip()
        f_out.write(f"{qid} 0 {docid} {rel}\n")

print(f"[SUCCESS] qrels written to {out_path}")
