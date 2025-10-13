# -*- coding: utf-8 -*-
import json
import torch
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, Float, MetaData, select
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# =======================
# 配置
# =======================
BASE_MODEL_PATH = "./Qwen3-0.6B"
LORA_PATHS = [
    "./lora-comment4-qwen3/final_model",
    "./lora-comment5-qwen3/final_model",
    "./lora-comment6-qwen3/final_model"
]

DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "bilibili_db"

MAX_WORKERS = 3
SAVE_INTERVAL = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# 数据库连接
# =======================
engine = create_engine(
    "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    ),
    pool_pre_ping=True
)
metadata = MetaData()
db_lock = Lock()

# =======================
# 加载模型
# =======================
def load_model(thread_id, lora_path):
    print("🧱 [{}] 开始加载模型 ({})...".format(thread_id, lora_path))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=3,
        dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.to(device)
    model.eval()
    print("✅ [{}] 模型加载完成 ({})".format(thread_id, lora_path))
    return tokenizer, model

# =======================
# 线程执行函数
# =======================
def worker_thread(thread_id, tokenizer, model, data_records):
    table_name = "com_sentiments_lora0_6b_{}".format(thread_id + 1)
    print("🚀 [{}] 使用表：{}".format(thread_id, table_name))

    with db_lock:
        sentiments_table = Table(
            table_name,
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("com_id", Integer, nullable=False, unique=True),
            Column("sentiment", Integer, nullable=False),
            Column("confidence", Float, nullable=False),  # ✅ 新增置信度字段
        )
        metadata.create_all(engine)

    local_conn = engine.connect()

    existing_ids = set()
    try:
        result = local_conn.execute(select(sentiments_table.c.com_id))
        existing_ids = {r[0] for r in result}
        print("🔁 [{}] 已存在 {} 条记录，将跳过这些".format(thread_id, len(existing_ids)))
    except Exception as e:
        print("⚠️ [{}] 查询已有记录失败: {}".format(thread_id, e))

    buffer = []
    skipped = 0

    for i, row in enumerate(data_records, 1):
        com_id = row["id"]
        if com_id in existing_ids:
            skipped += 1
            continue

        city = row["city"]
        coms = row["com"]
        title = row["title"]
        description = row["description"]
        tags = row["tag"]

        prompt = f"""
你是情感分析助手。
请根据以下信息判断评论对城市「{city}」的态度。**重点关注评论本身的内容{coms}**：
评论: {coms}
评论重复一遍以强化其权重: {coms}

视频元数据:
- 标题: {title}
- 描述: {description}
- 标签: {tags}

请输出评论在视频语境下的情感强烈程度：
2 表示支持/积极且表达强烈
1 表示中立或表达一般
0 表示反对/负面且表达强烈
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred].item()  # ✅ 计算置信度
        except Exception as e:
            print("⚠️ [{}] com_id={} 推理出错: {}".format(thread_id, com_id, e))
            pred = 1
            confidence = 0.0

        buffer.append({"com_id": com_id, "sentiment": pred, "confidence": confidence})

        if i % 10 == 0:
            print("🧩 [{}] 已处理 {}/{} (跳过 {})".format(thread_id, i, len(data_records), skipped))

        if len(buffer) >= SAVE_INTERVAL:
            try:
                local_conn.execute(sentiments_table.insert(), buffer)
                local_conn.commit()
                print("📝 [{}] 已保存 {} 条结果".format(thread_id, len(buffer)))
                buffer.clear()
            except Exception as e:
                print("❌ [{}] 保存失败: {}".format(thread_id, e))
                buffer.clear()

    if buffer:
        try:
            local_conn.execute(sentiments_table.insert(), buffer)
            local_conn.commit()
            print("📝 [{}] 最后保存 {} 条结果".format(thread_id, len(buffer)))
        except Exception as e:
            print("❌ [{}] 最后保存失败: {}".format(thread_id, e))

    local_conn.close()
    print("✅ [{}] 完成 (跳过 {} 条)".format(thread_id, skipped))

# =======================
# 主函数
# =======================
def main():
    print("🚀 启动 {} 个线程，每个线程独立模型".format(MAX_WORKERS))

    models = []
    for i in range(MAX_WORKERS):
        tokenizer, model = load_model(i, LORA_PATHS[i])
        models.append((tokenizer, model))

    print("✅ 所有模型加载完成")

    query = """
        SELECT a.id, b.city, b.data->>'title' as title, b.data->>'description' as description, b.data->>'tag' as tag, a.comment_text as com
        FROM video_comments as a left join videos as b on a.video_id = b.id
        WHERE b.istrue = 1
        ORDER BY id
    """
    df = pd.read_sql(query, engine)
    print("📦 主线程读取到 {} 条记录".format(len(df)))

    data_for_threads = df.to_dict(orient="records")

    print("✅ 开始并发推理")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(MAX_WORKERS):
            executor.submit(worker_thread, i, models[i][0], models[i][1], data_for_threads.copy())

    print("🏁 全部线程已启动")

if __name__ == "__main__":
    main()
