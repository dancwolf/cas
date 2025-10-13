# -*- coding: utf-8 -*-
import json
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# ======================
# 数据库配置
# ======================
DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "bilibili_db"

engine = create_engine(
    "postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}".format(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)
)

# ======================
# 配置
# ======================
SAMPLE_SIZE = 30  # 每次抽取多少条弹幕
OUTPUT_FILE = Path("danmaku_sentiment_dataset.jsonl")

# ======================
# 读取已标注弹幕 ID
# ======================
def get_annotated_danmaku_ids():
    if not OUTPUT_FILE.exists():
        return set()
    annotated_ids = set()
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                did = int(data["danmaku_id"])
                annotated_ids.add(did)
            except Exception:
                continue
    return annotated_ids


# ======================
# 随机获取未标注弹幕
# ======================
def fetch_random_danmaku(n=SAMPLE_SIZE, exclude_ids=set()):
    query = """
    SELECT d.id AS danmaku_id, d.danmaku_text, v.city, v.data
    FROM video_danmaku d
    JOIN videos v ON d.video_id = v.id
    WHERE v.istrue = 1
    ORDER BY RANDOM()
    LIMIT {0}
    """.format(n * 2)
    df = pd.read_sql(query, engine)
    df = df[~df["danmaku_id"].isin(exclude_ids)]
    return df.head(n).to_dict(orient="records")


# ======================
# 人工标注函数
# ======================
def annotate_danmaku(row):
    try:
        data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
    except Exception:
        data = {}
    title = data.get("title", "")
    description = data.get("description", "")
    tags = data.get("tag", "")

    print("\n=========================")
    print("弹幕ID:", row["danmaku_id"])
    print("城市:", row["city"])
    print("标题:", title)
    print("描述:", description)
    print("标签:", tags)
    print("弹幕内容:", row["danmaku_text"])
    print("=========================")

    while True:
        label = input("请输入情感 (0=反对, 1=中立, 2=支持): ").strip()
        if label in {"0", "1", "2"}:
            break
        print("输入无效，请输入 0, 1 或 2")

    return {
        "instruction": "根据视频及弹幕内容判断弹幕情感",
        "input": "城市: {0}\n标题: {1}\n描述: {2}\n标签: {3}\n弹幕: {4}".format(
            row["city"], title, description, tags, row["danmaku_text"]
        ),
        "output": label,
        "danmaku_id": row["danmaku_id"]
    }


# ======================
# 主程序
# ======================
def main():
    annotated_ids = get_annotated_danmaku_ids()
    danmaku_list = fetch_random_danmaku(SAMPLE_SIZE, annotated_ids)

    if not danmaku_list:
        print("没有可标注的弹幕了！")
        return

    records = []
    for row in danmaku_list:
        record = annotate_danmaku(row)
        records.append(record)

    # 保存为 jsonl（追加模式）
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("✅ 已保存 {0} 条弹幕标注到 {1}".format(len(records), OUTPUT_FILE))

    # 统计总条数
    total_count = sum(1 for _ in OUTPUT_FILE.open("r", encoding="utf-8"))
    print("📊 当前总共有 {0} 条记录".format(total_count))


if __name__ == "__main__":
    main()
