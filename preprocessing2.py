import pandas as pd

CSV_PATH = "NeoJaundice/NeoJaundice/chd_jaundice_published_2.csv"

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# remove duplicates
df = df.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)

# keep only required columns
df = df[["image_idx", "blood(mg/dL)"]]

df.to_csv("processed_data.csv", index=False)

print("✅ Data preprocessing done")