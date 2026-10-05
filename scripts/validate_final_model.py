from pathlib import Path

from src.data.physionet import load_split
from src.inference import predict_records


RAW_DIR = Path("data/raw")
MODEL_PATH = Path("artifacts/gru_physionet2012.pt")

records = load_split(RAW_DIR, split="set-b")[:32]
assert all(record.label is None for record in records)

output = predict_records(
    records,
    MODEL_PATH,
    device="cpu",
    batch_size=32,
)

print("records:", len(output["record_id"]))
print("score range:", float(output["score"].min()), float(output["score"].max()))
print("positive predictions:", int(output["prediction"].sum()))
print("first five:")

for rid, score, pred in zip(
    output["record_id"][:5],
    output["score"][:5],
    output["prediction"][:5],
):
    print(rid, f"{score:.4f}", int(pred))
