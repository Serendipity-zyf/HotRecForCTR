import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from multiprocessing import Pool, cpu_count, Lock


# 创建两个全局锁用于文件写入
train_lock = Lock()
test_lock = Lock()


def process_single_user(user_data):
    """处理单个用户的数据"""
    # 确保按时间戳排序
    user_data = user_data.sort_values(by=["timestamp"], ascending=True)

    # 生成训练和测试数据（文本格式）
    train_text, test_text = generate_ctr_data(user_item_df=user_data, max_seq_length=5)

    # 使用锁安全地写入文件
    if train_text:
        with train_lock and open("./train/train_data.txt", "a") as f:
            f.write(train_text)

    if test_text:
        with test_lock and open("./test/test_data.txt", "a") as f:
            f.write(test_text)

    return 1  # 返回1表示成功处理了一个用户


def generate_ctr_data(user_item_df: pd.DataFrame, max_seq_length: int = None) -> Tuple[str, str]:
    """
    构造CTR预测的训练集和测试集数据，但不写入文件，而是返回文本内容
    这个函数基于您现有的create_ctr_train_test_sets函数，但移除了文件写入部分
    """
    train_data = []
    test_data = []
    user_id = user_item_df["user_id"].iloc[0]
    # 获取用户的所有交互记录
    interactions = user_item_df.reset_index(drop=True)
    n_interactions = len(interactions)

    if n_interactions == 1:
        # 用户只有一条交互记录，只能用于训练，构造不了测试集
        train_data.append(
            {
                "user_id": user_id,
                "target_item_id": interactions.iloc[0]["item_id"],
                "user_behavior_seq": [],
                "label": interactions.iloc[0]["click"],
            }
        )
    else:
        # 用户有多条交互记录，可以构造训练和测试集
        # 训练集：使用前 k-1 个记录预测第 k 个记录（ k 从 1 到 n-1 ）
        for k in range(1, n_interactions):
            # 如果限制序列长度，只取最近的 max_seq_length 个记录
            if max_seq_length is not None and k > max_seq_length:
                prev_items = interactions.iloc[k - max_seq_length : k]["item_id"].tolist()
            else:
                prev_items = interactions.iloc[:k]["item_id"].tolist()

            target_item = interactions.iloc[k]["item_id"]
            label = interactions.iloc[k]["click"]

            train_data.append(
                {
                    "user_id": user_id,
                    "target_item_id": target_item,
                    "user_behavior_seq": prev_items,
                    "label": label,
                }
            )

        # 测试集：使用前 n-1 个记录预测第 n 个记录
        if max_seq_length is not None and n_interactions - 1 > max_seq_length:
            prev_items = interactions.iloc[-max_seq_length - 1 : -1]["item_id"].tolist()
        else:
            prev_items = interactions.iloc[:-1]["item_id"].tolist()

        target_item = interactions.iloc[-1]["item_id"]
        label = interactions.iloc[-1]["click"]

        test_data.append(
            {
                "user_id": user_id,
                "target_item_id": target_item,
                "user_behavior_seq": prev_items,
                "label": label,
            }
        )

    # 将数据转换为文本格式
    train_text = ""
    for item in train_data:
        behavior_seq = "<sep>".join(str(i) for i in item["user_behavior_seq"])
        line = f"{item['user_id']}\t{item['target_item_id']}\t{behavior_seq}\t{item['label']}\n"
        train_text += line

    test_text = ""
    for item in test_data:
        behavior_seq = "<sep>".join(str(i) for i in item["user_behavior_seq"])
        line = f"{item['user_id']}\t{item['target_item_id']}\t{behavior_seq}\t{item['label']}\n"
        test_text += line

    return train_text, test_text


def batch_process_users(review_df, batch_size=1000, n_processes=None):
    """批量处理用户数据"""
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    start_time = time.time()

    # 创建输出目录
    os.makedirs("./train", exist_ok=True)
    os.makedirs("./test", exist_ok=True)

    # 清空输出文件（如果存在）
    open("./train/train_data.txt", "w").close()
    open("./test/test_data.txt", "w").close()

    # 按用户ID分组
    grouped = review_df.groupby("user_id")
    total_users = len(grouped)
    print(f"Total users to process: {total_users}")

    processed_count = 0

    # 按批次处理
    batch_count = 0
    with Pool(processes=n_processes) as pool:
        for i in range(0, total_users, batch_size):
            batch_groups = []
            # 获取当前批次的用户组
            batch_user_ids = list(grouped.groups.keys())[i : min(i + batch_size, total_users)]
            for user_id in batch_user_ids:
                batch_groups.append(grouped.get_group(user_id))

            # 并行处理当前批次
            results = list(
                tqdm(
                    pool.imap(process_single_user, batch_groups),
                    total=len(batch_groups),
                    desc=f"Batch {batch_count+1}/{(total_users+batch_size-1)//batch_size}",
                )
            )

            processed_count += sum(results)
            batch_count += 1

            # 计算并显示进度和预计剩余时间
            elapsed_time = time.time() - start_time
            progress = processed_count / total_users
            estimated_total_time = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total_time - elapsed_time

            print(f"Processed {processed_count}/{total_users} users ({progress:.2%})")
            print(
                f"Elapsed time: {elapsed_time/60:.2f} minutes, Estimated remaining: {remaining_time/60:.2f} minutes"
            )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Processing complete! Total time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    # 读取数据
    review_df = pd.read_parquet("./review/review.parquet")
    # 处理数据
    batch_process_users(review_df, batch_size=1000, n_processes=cpu_count())
