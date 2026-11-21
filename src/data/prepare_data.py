"""校验 JSONL 教学样本、去重并按主题可复现地划分数据集。"""
from __future__ import annotations
import argparse, json, random, re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/seed_examples.jsonl"


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def read_and_validate(path: Path):
    rows, errors = [], []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            errors.append(f"line {line_no}: blank line")
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {line_no}: invalid JSON ({e})")
            continue
        if not isinstance(item, dict) or not isinstance(item.get("messages"), list):
            errors.append(f"line {line_no}: expected object with messages list")
            continue
        msgs = item["messages"]
        if len(msgs) < 2 or msgs[0].get("role") != "user" or msgs[-1].get("role") != "assistant":
            errors.append(f"line {line_no}: expected user prompt and assistant answer")
            continue
        if any(m.get("role") not in {"system", "user", "assistant"} or not str(m.get("content", "")).strip() for m in msgs):
            errors.append(f"line {line_no}: unsupported role or empty content")
            continue
        if not str(item.get("topic", "")).strip():
            errors.append(f"line {line_no}: missing topic")
            continue
        rows.append(item)
    return rows, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=RAW)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--preview-per-split", type=int, default=2)
    args = ap.parse_args()
    rows, errors = read_and_validate(args.input)
    if errors:
        raise SystemExit("Data validation failed:\n" + "\n".join(errors))

    # 以用户问题为键去重；同一问题若答案不同也只保留首条并报告冲突。
    unique, seen = [], {}
    conflicts = []
    for row in rows:
        key = norm(next(m["content"] for m in row["messages"] if m["role"] == "user"))
        signature = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        if key in seen:
            if seen[key] != signature:
                conflicts.append(key)
            continue
        seen[key] = signature
        unique.append(row)

    # 检查近似问句；达到阈值仅警告，需人工复核语义是否重复。
    questions = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in unique]
    near = []
    for i, q in enumerate(questions):
        nq = norm(q)
        for j in range(i):
            ratio = SequenceMatcher(None, nq, norm(questions[j])).ratio()
            if ratio >= 0.88:
                near.append((j + 1, i + 1, round(ratio, 3)))

    buckets = defaultdict(list)
    for row in unique:
        buckets[row["topic"]].append(row)
    rng = random.Random(args.seed)
    splits = {"train": [], "validation": [], "test": []}
    for topic in sorted(buckets):
        group = buckets[topic][:]
        rng.shuffle(group)
        n = len(group)
        # 62.5/18.75/18.75，16 条/主题时得到 10/3/3。
        n_train = round(n * 0.625)
        n_val = round(n * 0.1875)
        splits["train"].extend(group[:n_train])
        splits["validation"].extend(group[n_train:n_train+n_val])
        splits["test"].extend(group[n_train+n_val:])
    for name in splits:
        rng.shuffle(splits[name])

    # 所有用户问题在各 split 间必须互斥。
    keys = {name: {norm(next(m["content"] for m in r["messages"] if m["role"] == "user")) for r in group} for name, group in splits.items()}
    assert not (keys["train"] & keys["validation"] or keys["train"] & keys["test"] or keys["validation"] & keys["test"])

    out_dir = ROOT / "data"
    for name, group in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in group:
                f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

    def lengths(group):
        chars = [sum(len(m["content"]) for m in r["messages"]) for r in group]
        return {"min": min(chars), "mean": round(sum(chars)/len(chars), 1), "max": max(chars)} if chars else {}
    report = ["# 数据处理报告", "", f"- 输入文件：`{args.input}`", f"- 随机种子：{args.seed}", f"- 原始样本：{len(rows)}；校验通过：{len(rows)}；去重后：{len(unique)}", f"- 完全重复问句：{len(rows)-len(unique)}；答案冲突：{len(conflicts)}", f"- 近似问句（相似度 ≥ 0.88）：{len(near)}（仅作人工复核提示）", "", "## 划分与长度（字符数，不含模板 token）", "", "| Split | 样本数 | 主题分布 | 问答字符数 min/mean/max |", "|---|---:|---|---:|"]
    for name, group in splits.items():
        dist = Counter(r["topic"] for r in group)
        report.append(f"| {name} | {len(group)} | {', '.join(f'{k}:{v}' for k,v in sorted(dist.items()))} | {lengths(group)['min']}/{lengths(group)['mean']}/{lengths(group)['max']} |")
    report += ["", "## 样本预览", ""]
    for name, group in splits.items():
        report.append(f"### {name}")
        for row in group[:args.preview_per_split]:
            report += ["", f"- 主题：{row['topic']}", f"- User：{row['messages'][0]['content']}", f"- Assistant：{row['messages'][-1]['content']}"]
        report.append("")
    if near:
        report += ["## 近似问句复核提示", "", "相似度较高的条目可能只是共享术语，也可能存在语义重复；请在编辑数据时人工确认。", ""]
        for a,b,score in near[:30]: report.append(f"- 原始样本行 {a} 与 {b}：{score}")
    report_path = ROOT / "reports/data_processing_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    preview = []
    for name, group in splits.items():
        for row in group[:args.preview_per_split]:
            preview.append(json.dumps({"split":name, **row}, ensure_ascii=False))
    (ROOT / "data/processed/sample_preview.jsonl").write_text("\n".join(preview)+"\n", encoding="utf-8")
    print(f"raw={len(rows)} unique={len(unique)} train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])}")
    print(f"exact_duplicates={len(rows)-len(unique)} answer_conflicts={len(conflicts)} near_pairs={len(near)}")
    print(f"report={report_path}")

if __name__ == "__main__":
    main()
