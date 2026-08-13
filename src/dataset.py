"""
GenRec MovieLens Dataset Loader & Preprocessor
Downloads MovieLens 100k / 1M dataset and constructs user history sequences with contextual signals.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Any
try:
    from src.verbalizer import GenRecVerbalizer
except ModuleNotFoundError:
    from verbalizer import GenRecVerbalizer

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

class MovieLensGenRecDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "./data",
        max_history_len: int = 10,
        download: bool = True
    ):
        self.data_dir = data_dir
        self.max_history_len = max_history_len
        self.verbalizer = GenRecVerbalizer(max_history_length=max_history_len)

        if download:
            self._download_and_extract()

        self._load_and_process_data()

    def _download_and_extract(self):
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, "ml-100k.zip")
        ml_dir = os.path.join(self.data_dir, "ml-100k")

        if not os.path.exists(ml_dir):
            print(f"📥 Downloading MovieLens dataset to {zip_path}...")
            urllib.request.urlretrieve(MOVIELENS_100K_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("✅ MovieLens dataset downloaded and extracted!")

    def _load_and_process_data(self):
        ml_dir = os.path.join(self.data_dir, "ml-100k")
        
        # Load Ratings: user_id, item_id, rating, timestamp
        ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
        ratings_df = pd.read_csv(
            os.path.join(ml_dir, "u.data"),
            sep="\t",
            names=ratings_cols,
            encoding="latin-1"
        )

        # Load Items: item_id, title, release_date, video_release_date, IMDb_URL, genres...
        genre_cols = [
            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols
        items_df = pd.read_csv(
            os.path.join(ml_dir, "u.item"),
            sep="|",
            names=item_cols,
            encoding="latin-1"
        )

        # Format genres into string list for each movie
        def extract_genres(row):
            active_genres = [g for g in genre_cols if row[g] == 1]
            return "|".join(active_genres) if active_genres else "General"

        items_df["genres"] = items_df.apply(extract_genres, axis=1)
        item_dict = items_df.set_index("item_id")[["title", "genres"]].to_dict("index")

        # Map item IDs to contiguous 0-indexed integer IDs for embedding table
        unique_items = sorted(items_df["item_id"].unique())
        self.item2idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.idx2item = {idx: item_id for item_id, idx in self.item2idx.items()}
        self.num_items = len(unique_items)

        # Sort ratings chronologically per user to create realistic interaction sequences
        ratings_df = ratings_df.sort_values(by=["user_id", "timestamp"])

        self.samples = []
        devices = ["Smart TV", "Mobile Phone", "Tablet", "Desktop Web"]
        times = ["Morning", "Afternoon", "Evening", "Late Night"]

        print("⚙️ Processing user interaction sequences into GenRec prompts...")
        for user_id, group in ratings_df.groupby("user_id"):
            history = []
            for _, row in group.iterrows():
                item_id = int(row["item_id"])
                rating = int(row["rating"])
                item_info = item_dict.get(item_id, {"title": "Unknown", "genres": "General"})
                
                # Each target item must have preceding watch history
                if len(history) >= 2:
                    # Synthetic context allocation based on timestamp hash
                    ctx_device = devices[int(row["timestamp"]) % len(devices)]
                    ctx_time = times[(int(row["timestamp"]) // 3600) % len(times)]
                    
                    context = {
                        "device": ctx_device,
                        "time_of_day": ctx_time,
                        "locale": "US"
                    }

                    # High-signal filter (Only target high ratings >= 3★ as positive engagements)
                    if rating >= 3:
                        prompt_str = self.verbalizer.verbalize(history, context)
                        target_idx = self.item2idx[item_id]
                        
                        self.samples.append({
                            "user_id": user_id,
                            "prompt": prompt_str,
                            "target_item_idx": target_idx,
                            "rating": rating
                        })

                # Append current item to user's rolling history
                history.append({
                    "title": item_info["title"],
                    "genres": item_info["genres"],
                    "rating": rating
                })

        print(f"✅ Created {len(self.samples)} training samples across {self.num_items} unique items!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    dataset = MovieLensGenRecDataset()
    print(f"\n--- SAMPLE 0 FROM DATASET ---")
    sample = dataset[0]
    print("PROMPT:")
    print(sample["prompt"])
    print(f"\nTARGET ITEM INDEX: {sample['target_item_idx']} (Movie: {dataset.idx2item[sample['target_item_idx']]})")
