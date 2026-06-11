from evidently import Dataset, Recsys
from evidently import DataDefinition
from evidently.metrics import MAP, MRR, NDCG, PrecisionTopK, RecallTopK, HitRate
from evidently.metrics import Precision, Recall, F1Score
from evidently import Report
import pandas as pd

def filter_users_for_k(df: pd.DataFrame, k: int, require_positive: bool = False) -> pd.DataFrame:
    cand_counts = df.groupby("user_id")["event_id"].nunique()
    users_with_enough_cands = cand_counts[cand_counts >= k].index

    if require_positive:
        pos_counts = df.groupby("user_id")["actual_label"].sum()
        users_with_pos = pos_counts[pos_counts > 0].index
        keep_users = users_with_enough_cands.intersection(users_with_pos)
    else:
        keep_users = users_with_enough_cands

    return df[df["user_id"].isin(keep_users)].copy()

pred_df = pd.read_csv("results_example.csv")

def evaluation(df: pd.DataFrame, k: int) -> pd.DataFrame:
    before_users = df["user_id"].nunique()
    df = filter_users_for_k(df, k=k, require_positive=False)
    after_users = df["user_id"].nunique()
    print(f"Users before/after filtering for k: {before_users} -> {after_users}")

    eval_data_recsys = Dataset.from_pandas(
        df,
        data_definition=DataDefinition(
            ranking=[Recsys(
            user_id="user_id",
            item_id="event_id",
            prediction="prediction",
            target="actual_label",
        )],
        )
    )

    report_recsys = Report([
        # MAP(k=k, ranking_id="recsys"),
        # MRR(k=k, ranking_id="recsys"),
        # NDCG(k=k, ranking_id="recsys"),
        # PrecisionTopK(k=k, ranking_id="recsys"),
        # RecallTopK(k=k, ranking_id="recsys"),
        # HitRate(k=k, ranking_id="recsys"),
        Precision(name="Overall Precision"),
        Recall(name="Overall Recall"),
        F1Score(name="Overall F1 Score"),
        ])

    recsys_eval = report_recsys.run(eval_data_recsys, None)
    recsys_eval.save_html("recsys_eval_report_cls.html")

evaluation(pred_df, k=1)