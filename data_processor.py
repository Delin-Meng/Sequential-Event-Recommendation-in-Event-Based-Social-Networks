import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from mamba_ssm import Mamba

def train_processor(dataset_dir, preprocessed_dir):
    df = pd.read_csv(dataset_dir / 'train.csv')
    # # Remove rows with both 0 in 'interested' and 'not_interested' columns
    # df = df[~((df['interested'] == 0) & (df['not_interested'] == 0))]
    # Remove unnecessary columns
    df = df.drop(columns=['invited', 'not_interested'])
    # Convert 'timestamp' to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert(None)
    df['timestamp'] = df['timestamp'].apply(lambda x: int(x.timestamp()))
    # Sort by 'timestamp' in ascending order and reset index
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
    # Divide the train set and test set by timestamp
    split_index = int(len(df) * 0.8)
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]
    df_train.to_csv(preprocessed_dir / 'train.csv', index=False)
    df_test.to_csv(preprocessed_dir / 'test.csv', index=False)

def event_preprocessor(dataset_dir, preprocessed_dir):
    df = pd.read_csv(dataset_dir / 'events.csv')
    # Remove unnecessary columns
    df = df.drop(columns=['user_id', 'start_time', 'city', 'state', 'zip', 'country', 'lat', 'lng'])
    # Keep event only existing in the train set and test set
    train_df = pd.read_csv(preprocessed_dir / 'train.csv')
    test_df = pd.read_csv(preprocessed_dir / 'test.csv')
    event_ids = set(train_df['event']).union(set(test_df['event']))
    df = df[df['event_id'].isin(event_ids)]
    df.to_csv(preprocessed_dir / 'events.csv', index=False)

def event_processor(preprocessed_dir, fit_stats: Dict[str, np.ndarray] | None = None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    df = pd.read_csv(preprocessed_dir / 'events.csv')
    count_cols: List[str] = [f"c_{i}" for i in range(1, 101)] + ["c_other"]
    count_log = np.log1p(df[count_cols].to_numpy(dtype=np.float32))
    if fit_stats is None:
        mean = count_log.mean(axis=0)
        std = count_log.std(axis=0)
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    else:
        mean = fit_stats["mean"]
        std = fit_stats["std"]
        std = np.where(std < 1e-8, 1.0, std)
        stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    count_scaled = (count_log - mean) / std
    count_scaled = np.clip(count_scaled, -5, 5)
    processed_cols = [f"{col}_proc" for col in count_cols]
    processed_df = pd.DataFrame(count_scaled, columns=processed_cols, index=df.index)
    events_processed = pd.concat([df[["event_id"]].copy(), processed_df], axis=1)

    return events_processed, stats

def load_seq_data(preprocessed_dir, window_size=5):
    # 1. Load preprocessed train / test / events
    train_df = pd.read_csv(preprocessed_dir / 'train.csv')
    test_df = pd.read_csv(preprocessed_dir / 'test.csv')
    events_df = pd.read_csv(preprocessed_dir / 'events_processed.csv')

    # 2. Mark source split
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    # 3. Concatenate and sort globally by user, timestamp
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    full_df = full_df.sort_values(["user", "timestamp"]).reset_index(drop=True)

    # 4. Filter out users with only one total interaction in the full timeline
    user_counts = full_df["user"].value_counts()
    valid_users = set(user_counts[user_counts > 1].index)
    full_df = full_df[full_df["user"].isin(valid_users)].reset_index(drop=True)

    # 5. Unified sample generation:
    #    For each user, walk through the full timeline in chronological order.
    #    History for each sample = all previous interactions before current one.
    #    Then assign the sample to train/test according to current row's split.
    def generate_samples_from_full_timeline(df, window_size=5):
        train_X_hist_e, train_X_hist_f, train_X_target_u, train_X_target_e, train_y = [], [], [], [], []
        test_X_hist_e, test_X_hist_f, test_X_target_u, test_X_target_e, test_y = [], [], [], [], []
        for user_id, group in df.groupby("user"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            event_ids = group["event"].tolist()
            labels = group["interested"].tolist()
            splits = group["split"].tolist()
            n = len(event_ids)
            # start from index 1, because index 0 has no previous history
            for i in range(1, n):
                target_event = event_ids[i]
                target_label = labels[i]
                target_split = splits[i]
                hist_events = event_ids[:i]
                hist_feedback = labels[:i]
                # keep only the latest window_size history
                if len(hist_events) < window_size:
                    padded_hist_events = [0] * (window_size - len(hist_events)) + hist_events
                    padded_hist_feedback = [0] * (window_size - len(hist_feedback)) + hist_feedback
                else:
                    padded_hist_events = hist_events[-window_size:]
                    padded_hist_feedback = hist_feedback[-window_size:]
                # assign sample by current target row's split
                if target_split == "train":
                    train_X_hist_e.append(padded_hist_events)
                    train_X_hist_f.append(padded_hist_feedback)
                    train_X_target_u.append(user_id)
                    train_X_target_e.append(target_event)
                    train_y.append(target_label)
                else:
                    test_X_hist_e.append(padded_hist_events)
                    test_X_hist_f.append(padded_hist_feedback)
                    test_X_target_u.append(user_id)
                    test_X_target_e.append(target_event)
                    test_y.append(target_label)
        return np.array(train_X_hist_e), np.array(train_X_hist_f), np.array(train_X_target_u), np.array(train_X_target_e), np.array(train_y), np.array(test_X_hist_e), np.array(test_X_hist_f), np.array(test_X_target_u), np.array(test_X_target_e), np.array(test_y)


    X_train_hist_e, X_train_hist_f, X_train_target_u, X_train_target_e, y_train, X_test_hist_e, X_test_hist_f, X_test_target_u, X_test_target_e, y_test = generate_samples_from_full_timeline(full_df, window_size=window_size)

    return  X_train_hist_e, X_train_hist_f, X_train_target_u, X_train_target_e, y_train, X_test_hist_e, X_test_hist_f, X_test_target_u, X_test_target_e, y_test, events_df

def event_to_vec(events_df):
    event_vec_dict = {}
    feat_cols = [col for col in events_df.columns if col != "event_id"]
    for row in events_df.itertuples(index=False):
        event_id = int(row.event_id)
        feat = np.asarray(row[1:], dtype=np.float32)
        event_vec_dict[event_id] = feat
    # Add a zero vector for padding event_id 0
    event_vec_dict[0] = np.zeros(len(feat_cols), dtype=np.float32)

    return event_vec_dict

def hist_seq_to_vec(X_hist, event_vec_dict):
    num_samples, seq_len = X_hist.shape
    feat_dim = len(next(iter(event_vec_dict.values())))
    X_hist_feat = np.zeros((num_samples, seq_len, feat_dim), dtype=np.float32)
    for i in range(num_samples):
        for t in range(seq_len):
            event_id = int(X_hist[i, t])
            X_hist_feat[i, t] = event_vec_dict.get(event_id, event_vec_dict[0])

    return X_hist_feat

def target_to_vec(X_target_e, event_vec_dict):
    num_samples = X_target_e.shape[0]
    feat_dim = len(next(iter(event_vec_dict.values())))
    X_target_feat = np.zeros((num_samples, feat_dim), dtype=np.float32)
    for i in range(num_samples):
        event_id = int(X_target_e[i])
        X_target_feat[i] = event_vec_dict.get(event_id, event_vec_dict[0])

    return X_target_feat

class Data2torch(Dataset):
    def __init__(self, X_hist_e, X_hist_f, X_target_u, X_target_e, X_target_e_feat, y):
        self.X_hist_e = torch.tensor(X_hist_e, dtype=torch.float32)       # (num_samples, seq_len, dim_feat)
        self.X_hist_f = torch.tensor(X_hist_f, dtype=torch.long)          # (num_samples, seq_len)
        self.X_target_u = torch.tensor(X_target_u, dtype=torch.long)      # (num_samples,)
        self.X_target_e = torch.tensor(X_target_e, dtype=torch.long)      # (num_samples,)
        self.X_target_e_feat = torch.tensor(X_target_e_feat, dtype=torch.float32)  # (num_samples, dim_feat)
        self.y = torch.tensor(y, dtype=torch.float32)                     # (num_samples,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_hist_e[idx], self.X_hist_f[idx], self.X_target_u[idx], self.X_target_e[idx], self.X_target_e_feat[idx], self.y[idx]

class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.05, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d_model), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        h = self.mamba(self.norm1(x))
        x = x + self.drop1(h)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x + self.ffn(self.norm2(x))
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x
    
class MambaEventModel(nn.Module):
    def __init__(self, input_dim=101, seq_len=5, d_model=64, d_state=16, d_conv=4, expand=2, num_layers=3, dropout=0.05, mlp_ratio=2.0, feedback_dim=4, num_users=None, user_emb_dim=64):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.feedback_dim = feedback_dim
        self.user_emb_dim = user_emb_dim

        if num_users is None:
            raise ValueError("num_users must be provided for user embedding.")

        # history feedback embedding: binary {0,1}
        self.feedback_emb = nn.Embedding(2, feedback_dim)
        # learnable user embedding
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=1.0 / math.sqrt(user_emb_dim))
        # history branch input = event feature + feedback embedding
        self.hist_proj = nn.Sequential(nn.Linear(input_dim + feedback_dim, d_model, bias=False), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([ResidualMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        # target event branch
        self.target_mlp = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        # fusion classifier: history + target event + user embedding
        self.classifier = nn.Sequential(nn.Linear(d_model * 2 + user_emb_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x_hist_e, x_hist_f, x_target_u, x_target_e):
        # x_hist_e: (batch_size, seq_len, input_dim)
        # x_hist_f: (batch_size, seq_len) in {0,1}
        # x_target_u: (batch_size,)
        # x_target_e: (batch_size, input_dim)
        hist_mask = (x_hist_e.abs().sum(dim=-1) > 0).float()   # (batch_size, seq_len)
        # history feedback embedding
        f_emb = self.feedback_emb(x_hist_f)                    # (batch_size, seq_len, feedback_dim)
        # concatenate event feature + feedback embedding
        hist_input = torch.cat([x_hist_e, f_emb], dim=-1)      # (batch_size, seq_len, input_dim + feedback_dim)
        # history branch
        h = self.hist_proj(hist_input)
        h = h + self.pos_embed[:, :h.size(1), :]
        h = h * hist_mask.unsqueeze(-1)

        for block in self.blocks:
            h = block(h, hist_mask)

        h = self.final_norm(h)
        h = h * hist_mask.unsqueeze(-1)
        lengths = hist_mask.sum(dim=1).long().clamp(min=1)
        last_idx = lengths - 1
        batch_idx = torch.arange(h.size(0), device=h.device)
        h_last = h[batch_idx, last_idx]
        h_mean = (h * hist_mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        h_hist = 0.5 * h_last + 0.5 * h_mean
        # target event branch
        h_target = self.target_mlp(x_target_e)
        # user embedding branch
        h_user = self.user_emb(x_target_u)   # (batch_size, user_emb_dim)
        # fusion
        h_cat = torch.cat([h_hist, h_target, h_user], dim=-1)
        logit = self.classifier(h_cat).squeeze(-1)

        return logit
    
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    dataset_dir = current_dir / 'dataset'
    preprocessed_dir = dataset_dir / 'preprocessed'
    preprocessed_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_processor(dataset_dir, preprocessed_dir)
    # event_preprocessor(dataset_dir, preprocessed_dir)
    # events_processed, stats = event_processor(preprocessed_dir)
    # events_processed.to_csv(preprocessed_dir / 'events_processed.csv', index=False)
    X_train_hist_e, X_train_hist_f, X_train_target_u, X_train_target_e, y_train, X_test_hist_e, X_test_hist_f, X_test_target_u, X_test_target_e, y_test, events_df = load_seq_data(preprocessed_dir)

    event_vec = event_to_vec(events_df)
    # Build user id mapping for embedding
    all_user_ids = np.unique(np.concatenate([X_train_target_u, X_test_target_u]))
    user2idx = {user_id: idx for idx, user_id in enumerate(all_user_ids)}
    X_train_target_u_idx = np.array([user2idx[u] for u in X_train_target_u], dtype=np.int64)
    X_test_target_u_idx = np.array([user2idx[u] for u in X_test_target_u], dtype=np.int64)
    num_users = len(user2idx)
    print("num_users:", num_users)
    X_train_hist_e_vec = hist_seq_to_vec(X_train_hist_e, event_vec)
    X_train_target_e_vec = target_to_vec(X_train_target_e, event_vec)
    X_test_hist_e_vec = hist_seq_to_vec(X_test_hist_e, event_vec)
    X_test_target_e_vec = target_to_vec(X_test_target_e, event_vec)

    train_dataset = Data2torch(X_train_hist_e_vec, X_train_hist_f, X_train_target_u_idx, X_train_target_e, X_train_target_e_vec, y_train)
    test_dataset = Data2torch(X_test_hist_e_vec, X_test_hist_f, X_test_target_u_idx, X_test_target_e, X_test_target_e_vec, y_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model = MambaEventModel(input_dim=101, d_model=64, num_users=num_users, user_emb_dim=64).to(device)
    # Loss function
    # criterion = nn.BCEWithLogitsLoss()
    pos_weight = torch.tensor([73.17 / 26.83], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(250):
        model.train()
        total_loss = 0.0

        for X_hist_e_batch, X_hist_f_batch, X_target_u_batch, X_target_e_id_batch, X_target_e_batch, y_batch in train_loader:
            X_hist_e_batch = X_hist_e_batch.to(device)
            X_hist_f_batch = X_hist_f_batch.to(device)
            X_target_u_batch = X_target_u_batch.to(device)
            X_target_e_batch = X_target_e_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_hist_e_batch, X_hist_f_batch, X_target_u_batch, X_target_e_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    all_target_users = []
    all_target_events = []

    with torch.no_grad():
        for X_hist_e_batch, X_hist_f_batch, X_target_u_batch, X_target_e_id_batch, X_target_e_batch, y_batch in test_loader:
            X_hist_e_batch = X_hist_e_batch.to(device)
            X_hist_f_batch = X_hist_f_batch.to(device)
            X_target_u_batch = X_target_u_batch.to(device)
            X_target_e_batch = X_target_e_batch.to(device)

            logits = model(X_hist_e_batch, X_hist_f_batch, X_target_u_batch, X_target_e_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(y_batch.int().numpy())
            all_target_users.extend(X_target_u_batch.cpu().numpy())
            all_target_events.extend(X_target_e_id_batch.numpy())
            # store the probabilities, predictions, and true labels for evaluation as csv files
    idx2user = {idx: user_id for user_id, idx in user2idx.items()}
    original_user_ids = [idx2user[idx] for idx in all_target_users]
    results_df = pd.DataFrame({"user_id": original_user_ids, "event_id": all_target_events, "probability": all_probs, "prediction": all_preds, "actual_label": all_labels})

    results_df.to_csv('results_250.csv', index=False)