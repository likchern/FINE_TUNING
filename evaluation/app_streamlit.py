"""
校验结果展示前端（Streamlit）

用法（在项目根目录 RL 下）:
  streamlit run evaluation/app_streamlit.py

支持：加载已有 report.json，或选择 jsonl 重新校验（规则评分，不调 API）。
可选加载得分 CSV（auto_scores_llm.csv / auto_scores.csv），通过条目的得分会从 CSV 补全显示。
每条数据均展示：通过则显示十维度+总分，未通过则显示具体问题。
"""

import csv
import json
import sys
from pathlib import Path

import streamlit as st

# 保证可导入同目录下的 validate_dialogue_data
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import validate_dialogue_data as vdd

DIMENSION_NAMES = vdd.DIMENSION_NAMES

st.set_page_config(page_title="对话数据校验结果", layout="wide")

st.title("多轮对话数据校验结果")

# 侧边：数据来源
st.sidebar.header("数据来源")
data_source = st.sidebar.radio(
    "选择方式",
    ["加载已有校验报告 (report.json)", "重新校验（选择 jsonl 文件，规则评分）"],
    index=0,
)

report = None
default_report = _APP_DIR / "report.json"
default_jsonl = _APP_DIR.parent / "data" / "test_dia.jsonl"

if data_source == "加载已有校验报告 (report.json)":
    report_path = st.sidebar.text_input(
        "报告 JSON 路径",
        value=str(default_report),
    )
    p = Path(report_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        st.sidebar.warning(f"文件不存在: {p}")

else:
    jsonl_path = st.sidebar.text_input(
        "待校验 jsonl 路径",
        value=str(default_jsonl),
    )
    if st.sidebar.button("开始校验"):
        p = Path(jsonl_path)
        if not p.exists():
            st.sidebar.error(f"文件不存在: {p}")
        else:
            with st.spinner("校验中（规则评分，不调 API）…"):
                report = vdd.run_validation(
                    p,
                    strict_length=False,
                    strict_safety=False,
                    score_dimensions=True,
                    score_llm=False,
                )
            st.sidebar.success("校验完成")

if report is None:
    st.info("请从左侧选择数据来源并加载报告，或选择 jsonl 后点击「开始校验」。")
    st.stop()


def _safe_int(x):
    try:
        return int(x) if x not in (None, "") else 0
    except (ValueError, TypeError):
        return 0


# 可选：从得分 CSV 补全通过条目的分数（report 里可能没有 dimension_scores）
st.sidebar.header("得分 CSV（可选）")
st.sidebar.caption("通过条目的十维度+总分可来自 report，或由此 CSV 按 dialogue_id 补全")
scores_csv_path = st.sidebar.text_input(
    "得分 CSV 路径",
    value="",
    placeholder="e.g. evaluation/auto_scores_llm.csv",
)
scores_by_id = {}
if scores_csv_path and Path(scores_csv_path).exists():
    try:
        with open(scores_csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                did = row.get("dialogue_id", "").strip()
                if not did:
                    continue
                scores_by_id[did] = {dim: _safe_int(row.get(dim)) for dim in DIMENSION_NAMES}
                scores_by_id[did]["总分"] = _safe_int(row.get("总分"))
        st.sidebar.success(f"已加载 {len(scores_by_id)} 条得分")
    except Exception as e:
        st.sidebar.error(f"加载 CSV 失败: {e}")
else:
    for _ in [Path(_APP_DIR / "auto_scores_llm.csv"), Path(_APP_DIR / "auto_scores.csv")]:
        if _.exists():
            try:
                with open(_, "r", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        did = row.get("dialogue_id", "").strip()
                        if not did:
                            continue
                        scores_by_id[did] = {dim: _safe_int(row.get(dim)) for dim in DIMENSION_NAMES}
                        scores_by_id[did]["总分"] = _safe_int(row.get("总分"))
                st.sidebar.caption(f"已自动加载 {_.name}（{len(scores_by_id)} 条）")
                break
            except Exception:
                pass


# 汇总
details = report.get("details", [])
total = report.get("total_lines", 0)
valid = report.get("valid_count", 0)
invalid = report.get("invalid_count", 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("总条数", total)
c2.metric("通过", valid)
c3.metric("未通过", invalid)
c4.metric("通过率", f"{(100 * valid / total):.1f}%" if total else "—")

if report.get("dimension_avg"):
    st.subheader("十维度均值（仅对通过条目）")
    dim_avg = report["dimension_avg"]
    cols = st.columns(5)
    for i, dim in enumerate(DIMENSION_NAMES):
        cols[i % 5].metric(dim, dim_avg.get(dim, "—"))
    st.metric("总分均值", dim_avg.get("总分", "—"))
    st.caption(f"参与评分条数: {report.get('dimension_scored_count', 0)}")

st.divider()
st.subheader("每条数据明细（通过显示分数，未通过显示问题）")

# 筛选
filter_status = st.selectbox(
    "筛选",
    ["全部", "仅通过", "仅未通过"],
    index=0,
)
if filter_status == "仅通过":
    rows = [d for d in details if d.get("ok")]
elif filter_status == "仅未通过":
    rows = [d for d in details if not d.get("ok")]
else:
    rows = details

for i, d in enumerate(rows):
    line_no = d.get("line_no", i + 1)
    did = d.get("dialogue_id", "—")
    ok = d.get("ok", False)
    label = "通过" if ok else "未通过"
    color = "green" if ok else "red"
    title = f"**{label}** · 第 {line_no} 行 · `{did}`"
    with st.expander(title, expanded=(not ok and len(rows) <= 20)):
        if ok:
            st.write("**十维度得分**")
            scores = d.get("dimension_scores") or {}
            did = d.get("dialogue_id", "")
            if not scores and did and did in scores_by_id:
                scores = scores_by_id[did]
                st.caption("得分来源: 得分 CSV")
            total_score = scores.get("总分", 0)
            cols = st.columns(6)
            for j, dim in enumerate(DIMENSION_NAMES):
                cols[j % 6].metric(dim, scores.get(dim, "—"))
            st.metric("总分", total_score)
            if d.get("stats"):
                st.caption(f"对话轮次: {d['stats'].get('turns', '—')} | 用户轮: {d['stats'].get('user_turns', '—')} | 助手轮: {d['stats'].get('assistant_turns', '—')}")
        else:
            st.write("**未通过原因**")
            for issue in d.get("issues", []):
                st.error(issue)
            if d.get("stats"):
                st.caption(f"对话轮次: {d['stats'].get('turns', '—')}")

st.caption(f"共 {len(rows)} 条 · 数据来源: {data_source}")
