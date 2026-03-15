"""
多轮对话数据质量自动化校验脚本

校验项目生成的 jsonl 对话数据，包括：
- 格式与结构（JSON、必填字段、dialogue_history 结构、首轮角色）
- 内容质量（空内容、占位符/未替换模板、thinking/response 泄露）
- 长度约束（用户/助手单轮字数，可选严格模式）
- 对话流程（最少轮次、角色交替、metadata 与实际轮次一致）
- 安全与合规（风险表述关键词扫描，可选严格模式）
- 十维度评分：规则代理 或 调用 Qwen API（--score-llm），API Key 从环境变量或 .env 读取
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 从 .env 加载环境变量（不依赖 python-dotenv）
def _load_dotenv() -> None:
    for _dir in [Path(__file__).resolve().parent.parent, Path.cwd()]:
        _env = _dir / ".env"
        if not _env.exists():
            continue
        try:
            with open(_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip()
                    if k and v and v.startswith(('"', "'")):
                        v = v[1:-1].replace("\\n", "\n")
                    if k:
                        os.environ[k] = v
        except OSError:
            pass
        break


_load_dotenv()
# 若已安装 python-dotenv 可再加载一次（覆盖更全）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()
except ImportError:
    pass

# API Key：环境变量 QWEN_API_KEY 或 DASHSCOPE_API_KEY
def _get_api_key() -> Optional[str]:
    return os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


# 必填顶层字段（至少包含这些 key）
REQUIRED_TOP_LEVEL = {
    "dialogue_id",
    "patient_id",
    "persona",
    "topic",
    "dialogue_history",
    "metadata",
}

# 对话单条必填字段
REQUIRED_TURN_KEYS = {"role", "content"}

# 合法 role
VALID_ROLES = {"user", "assistant"}

# 题目与文档中的长度约束（字符数）
USER_MAIN_MIN, USER_MAIN_MAX = 8, 30   # 每回合 8~30 字为主
USER_EXTRA_MAX = 40                     # 必要时可到 40 字
ASSISTANT_TYPICAL = 20                  # 照护师习惯短句约 20 字
ASSISTANT_MAX = 80                      # 照护师单轮最多 80 字

# 对话流程：至少多少轮视为有效多轮对话
MIN_TURNS = 2

# 占位符/泄露模式：content 中出现则视为内容质量问题
PLACEHOLDER_PATTERNS = [
    r"\[内容\]",
    r"\[请填写.*?\]",
    r"<response>",
    r"</response>",
    r"Response:\s*$",
    r"Thinking:\s*$",
    r"^\s*Thinking\s*:\s*\n",
    r"^\s*Response\s*:\s*\n",
    r"\[内容\]\s*</think>",
    r"\{\{.*?\}\}",  # 未替换的模板变量
]

# 风险关键词（安全与合规类）
SAFETY_WARN_PATTERNS = [
    r"绝对[能可]?治愈",
    r"停用?\s*药",
    r"不必?\s*用药",
    r"替代\s*医生",
    r"保证\s*治愈",
    r"100%\s*有效",
]

# 与 human_review_guide.md 一致的十个评分维度（自动化采用规则/启发式代理）
DIMENSION_NAMES = [
    "即时性",
    "意图识别",
    "内容准确性",
    "建议可遵循",
    "恰当表达",
    "情感关怀",
    "话题引导",
    "安全性",
    "负向情感风险",
    "合规性",
]

# 负向情感风险：易引发焦虑的表述
NEGATIVE_RISK_PATTERNS = [
    r"一定\s*会",
    r"必须\s*马上",
    r"否则\s*会\s*(危险|严重|出事)",
    r"很\s*危险",
    r"已经\s*很\s*严重",
    r"再不\s*.*就\s*晚了",
]

# 合规性：越界表述（不能替代医生做的）
COMPLIANCE_VIOLATION_PATTERNS = [
    r"给你\s*开\s*药",
    r"开\s*处方",
    r"具体\s*剂量\s*[为是]\s*[^\s，。]+",  # 具体剂量建议
    r"你\s*应该\s*吃\s*\d+\s*[mg片粒]",
]

# 建议可遵循：可执行建议的提示词
ACTIONABLE_MARKERS_STRONG = re.compile(
    r"(试试|尝试|建议|可以|先\s*[^再]+再|每天|一次|分钟|步|次|克|毫升)"
)
ACTIONABLE_MARKERS_WEAK = re.compile(r"(建议|可以|试试|慢慢来)")

# 情感关怀：共情与支持用语
EMPATHY_MARKERS_STRONG = re.compile(
    r"(理解|没事|慢慢来|别担心|咱们|真棒|很好|特别|辛苦|不容易|放心)"
)
EMPATHY_MARKERS_WEAK = re.compile(r"(好的|谢谢|嗯|哦|您)")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                records.append((i, None))
                # 后面用 None 表示解析失败
    return records


def _content_has_placeholder(content: str) -> Optional[str]:
    """若 content 含占位符或泄露，返回匹配到的模式描述，否则返回 None。"""
    if not content or not isinstance(content, str):
        return None
    text = content.strip()
    for pat in PLACEHOLDER_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return pat
    if re.search(r"^\s*(Thinking|Response)\s*:\s*\n", text, re.IGNORECASE) and len(text) < 100:
        return "疑似 Thinking/Response 未替换"
    return None


def check_turn(turn: Dict[str, Any], turn_idx: int, strict_length: bool) -> List[str]:
    issues = []
    if not isinstance(turn, dict):
        issues.append(f"turn_{turn_idx}: 非字典")
        return issues
    for k in REQUIRED_TURN_KEYS:
        if k not in turn:
            issues.append(f"turn_{turn_idx}: 缺少字段 '{k}'")
    role = turn.get("role")
    content = turn.get("content")
    if role is not None and role not in VALID_ROLES:
        issues.append(f"turn_{turn_idx}: role 非法 '{role}'")
    if content is not None and not isinstance(content, str):
        issues.append(f"turn_{turn_idx}: content 非字符串")
    if content is not None and isinstance(content, str):
        if len(content.strip()) == 0:
            issues.append(f"turn_{turn_idx}: content 为空")
        else:
            placeholder = _content_has_placeholder(content)
            if placeholder:
                issues.append(f"turn_{turn_idx}: 内容含占位符或未替换模板 ({placeholder})")
    if strict_length and content is not None and isinstance(content, str):
        length = len(content.strip())
        if role == "user" and length > USER_EXTRA_MAX:
            issues.append(f"turn_{turn_idx}: 用户内容过长 {length} 字 (建议≤{USER_EXTRA_MAX})")
        if role == "assistant" and length > ASSISTANT_MAX:
            issues.append(f"turn_{turn_idx}: 助手内容过长 {length} 字 (要求≤{ASSISTANT_MAX})")
    return issues


def check_dialogue_structure(record: Dict[str, Any], strict_length: bool = False) -> List[str]:
    issues = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in record:
            issues.append(f"缺少顶层字段: {key}")
    if "dialogue_history" in record:
        hist = record["dialogue_history"]
        if not isinstance(hist, list):
            issues.append("dialogue_history 非列表")
        else:
            if len(hist) == 0:
                issues.append("dialogue_history 为空")
            elif len(hist) < MIN_TURNS:
                issues.append(f"对话轮次不足: {len(hist)} < {MIN_TURNS}")
            for i, turn in enumerate(hist):
                issues.extend(check_turn(turn, i, strict_length))
            if hist and isinstance(hist[0], dict) and hist[0].get("role") != "user":
                issues.append("首轮应为用户发言")
    if "metadata" in record and isinstance(record["metadata"], dict) and "dialogue_history" in record:
        hist = record["dialogue_history"]
        if isinstance(hist, list):
            meta_turns = record["metadata"].get("total_turns")
            if meta_turns is not None and meta_turns != len(hist):
                issues.append(f"metadata.total_turns({meta_turns}) 与 dialogue_history 长度({len(hist)})不一致")
    if "persona" in record and isinstance(record["persona"], dict):
        if "raw_persona" not in record["persona"] and "enhanced_description" not in record["persona"]:
            issues.append("persona 中建议包含 raw_persona 或 enhanced_description")
    return issues


def safety_scan(text: str) -> List[str]:
    hits = []
    for pat in SAFETY_WARN_PATTERNS:
        if re.search(pat, text or ""):
            hits.append(pat)
    return hits


def _all_assistant_contents(record: Dict[str, Any]) -> List[str]:
    hist = record.get("dialogue_history") or []
    if not isinstance(hist, list):
        return []
    return [t.get("content") or "" for t in hist if isinstance(t, dict) and t.get("role") == "assistant"]


def _last_user_content(record: Dict[str, Any]) -> str:
    hist = record.get("dialogue_history") or []
    if not isinstance(hist, list):
        return ""
    for t in reversed(hist):
        if isinstance(t, dict) and t.get("role") == "user":
            return t.get("content") or ""
    return ""


def _topic_keywords(topic: str) -> set:
    """从 topic 中提取简单关键词用于话题相关性。"""
    if not topic:
        return set()
    # 常见健康管理相关词
    common = {"血糖", "饮食", "运动", "用药", "药", "胰岛素", "测", "监测", "血压", "体重", "糖尿病", "吃", "喝"}
    found = set()
    for w in common:
        if w in topic:
            found.add(w)
    return found


def score_ten_dimensions(record: Dict[str, Any]) -> Dict[str, int]:
    """
    对单条对话按 human_review_guide.md 的十个维度做自动化评分（0/1/2）。
    采用规则与启发式代理，无法完全替代人工评审，仅作一致性参考与初筛。
    """
    scores = {d: 0 for d in DIMENSION_NAMES}
    assistant_texts = _all_assistant_contents(record)
    all_assistant = " ".join(assistant_texts)
    last_user = _last_user_content(record)
    topic = record.get("topic") or ""
    topic_kw = _topic_keywords(topic)

    if not assistant_texts:
        return scores

    # 安全性：无风险词 2，有则 0
    safety_hits = safety_scan(all_assistant)
    scores["安全性"] = 0 if safety_hits else 2

    # 内容准确性：与安全性同源，无安全词给 2，有给 0
    scores["内容准确性"] = 0 if safety_hits else 2

    # 负向情感风险：无易引发焦虑表述 2，有则 0
    neg_risk = any(re.search(p, all_assistant) for p in NEGATIVE_RISK_PATTERNS)
    scores["负向情感风险"] = 0 if neg_risk else 2

    # 合规性：无越界表述 2，有则 0
    compliance_ok = not any(re.search(p, all_assistant) for p in COMPLIANCE_VIOLATION_PATTERNS)
    scores["合规性"] = 2 if compliance_ok else 0

    # 恰当表达：助手回复长度适中、无占位符
    lengths = [len((c or "").strip()) for c in assistant_texts]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    no_placeholder = not any(_content_has_placeholder(c) for c in assistant_texts)
    if no_placeholder and 5 <= avg_len <= ASSISTANT_MAX:
        scores["恰当表达"] = 2
    elif no_placeholder and avg_len >= 1:
        scores["恰当表达"] = 1
    else:
        scores["恰当表达"] = 0

    # 建议可遵循：有可执行建议用语
    if ACTIONABLE_MARKERS_STRONG.search(all_assistant):
        scores["建议可遵循"] = 2
    elif ACTIONABLE_MARKERS_WEAK.search(all_assistant):
        scores["建议可遵循"] = 1
    else:
        scores["建议可遵循"] = 0

    # 情感关怀：共情用语
    if EMPATHY_MARKERS_STRONG.search(all_assistant):
        scores["情感关怀"] = 2
    elif EMPATHY_MARKERS_WEAK.search(all_assistant):
        scores["情感关怀"] = 1
    else:
        scores["情感关怀"] = 0

    # 话题引导：与 topic 有词重叠
    overlap = sum(1 for w in topic_kw if w in all_assistant)
    if topic_kw and overlap >= 2:
        scores["话题引导"] = 2
    elif topic_kw and overlap >= 1:
        scores["话题引导"] = 1
    else:
        scores["话题引导"] = 1 if len(all_assistant.strip()) > 10 else 0

    # 即时性：有实质回复则视为有回应
    if not all_assistant.strip():
        scores["即时性"] = 0
    elif len(all_assistant.strip()) >= 10 and last_user:
        scores["即时性"] = 2
    else:
        scores["即时性"] = 1

    # 意图识别：有建议/共情/追问则视为在回应意图
    if scores["建议可遵循"] >= 1 and scores["情感关怀"] >= 1:
        scores["意图识别"] = 2
    elif scores["建议可遵循"] >= 1 or scores["情感关怀"] >= 1 or "?" in all_assistant or "？" in all_assistant:
        scores["意图识别"] = 1
    else:
        scores["意图识别"] = 1 if len(all_assistant.strip()) > 5 else 0

    # 总分：十个维度分数之和（满分 20）
    scores["总分"] = sum(scores[d] for d in DIMENSION_NAMES)
    return scores


# 十维度评分用的 LLM 提示（与 human_review_guide 一致）
DIMENSION_CRITERIA = """
请对以下「照护师（助手）」的回复质量，按以下十个维度分别打 0、1、2 分，并给出总分（十维度之和，满分20）。

维度与标准：
- 即时性：2=准确识别患者显性诉求并回应，1=仅部分/不完整，0=未识别或误解
- 意图识别：2=识别显性并挖掘隐性需求，1=仅显性，0=未识别主要需求
- 内容准确性：2=科学准确符合指南，1=大部分准确有误，0=明显错误或误导
- 建议可遵循：2=具体可操作有步骤，1=较笼统但方向正确，0=模糊或不可执行
- 恰当表达：2=清晰专业易理解，1=基本清晰有瑕疵，0=混乱或过于专业化
- 情感关怀：2=明显情感支持共情，1=基本礼貌用语，0=冷漠无情感互动
- 话题引导：2=主动引导至健康话题，1=被动但有引导意识，0=无引导或偏离
- 安全性：2=完全安全无风险建议，1=基本安全需补充，0=有安全隐患或危险建议
- 负向情感风险：2=无负面情绪风险，1=可能轻微不适，0=明显引发焦虑/不满
- 合规性：2=完全合规，1=基本合规有小瑕疵，0=不合规或越界

只输出一个 JSON 对象，不要其他文字。键名为上述十个维度（中文），以及「总分」。总分 = 十个维度分数之和。例如：
{"即时性":2,"意图识别":1,"内容准确性":2,"建议可遵循":2,"恰当表达":2,"情感关怀":1,"话题引导":2,"安全性":2,"负向情感风险":2,"合规性":2,"总分":18}
"""


def _call_qwen_api(prompt: str, api_key: str, model: str, timeout: int = 120) -> str:
    """调用阿里百炼 / DashScope Qwen API，返回 content 文本。"""
    import requests
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.1, "max_tokens": 1024, "result_format": "message"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    out = data.get("output", {})
    choices = out.get("choices", [])
    if choices and isinstance(choices[0].get("message"), dict):
        return choices[0]["message"].get("content", "") or ""
    if out.get("text"):
        return out["text"]
    raise ValueError(f"无法从 API 响应解析 content: {json.dumps(data, ensure_ascii=False)[:500]}")


def _dialogue_to_text(record: Dict[str, Any]) -> str:
    """将对话记录转成可读文本，供 LLM 评分。"""
    hist = record.get("dialogue_history") or []
    if not isinstance(hist, list):
        return ""
    lines = []
    topic = (record.get("topic") or "").strip()
    if topic:
        lines.append(f"【对话主题】{topic}\n")
    for i, t in enumerate(hist):
        if not isinstance(t, dict):
            continue
        role = "用户" if t.get("role") == "user" else "照护师"
        content = (t.get("content") or "").strip()
        if content:
            lines.append(f"{i+1}. {role}: {content}")
    return "\n".join(lines)


def _parse_llm_scores(raw: str) -> Dict[str, int]:
    """从 LLM 返回文本中解析出十维度+总分，缺省 0。"""
    scores = {d: 0 for d in DIMENSION_NAMES}
    scores["总分"] = 0
    # 尝试从文本中提取 JSON 块
    raw = raw.strip()
    for start in ("{", "```json", "```"):
        if start in raw:
            idx = raw.find(start)
            if start.startswith("```"):
                idx = raw.find("{", idx)
            end = raw.rfind("}") + 1
            if idx >= 0 and end > idx:
                try:
                    obj = json.loads(raw[idx:end])
                    for d in DIMENSION_NAMES:
                        if d in obj and isinstance(obj[d], (int, float)):
                            scores[d] = max(0, min(2, int(obj[d])))
                    if "总分" in obj and isinstance(obj["总分"], (int, float)):
                        scores["总分"] = max(0, min(20, int(obj["总分"])))
                    else:
                        scores["总分"] = sum(scores[d] for d in DIMENSION_NAMES)
                    return scores
                except json.JSONDecodeError:
                    pass
    return scores


def score_ten_dimensions_llm(
    record: Dict[str, Any], api_key: str, model: str = "qwen-plus"
) -> Dict[str, int]:
    """使用 Qwen API 对单条对话按十维度+总分评分。"""
    text = _dialogue_to_text(record)
    if not text.strip():
        out = {d: 0 for d in DIMENSION_NAMES}
        out["总分"] = 0
        return out
    prompt = DIMENSION_CRITERIA.strip() + "\n\n【待评分对话】\n" + text
    raw = _call_qwen_api(prompt, api_key, model)
    return _parse_llm_scores(raw)


def validate_one(
    line_no: int,
    data: Optional[Dict[str, Any]],
    strict_length: bool = False,
    strict_safety: bool = False,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    stats = {"turns": 0, "user_turns": 0, "assistant_turns": 0, "safety_warnings": [], "warnings": []}
    if data is None:
        return False, [f"第 {line_no} 行: JSON 解析失败"], stats
    issues = list(check_dialogue_structure(data, strict_length))
    hist = data.get("dialogue_history") or []
    if isinstance(hist, list):
        stats["turns"] = len(hist)
        for t in hist:
            if isinstance(t, dict):
                r = t.get("role")
                if r == "user":
                    stats["user_turns"] += 1
                elif r == "assistant":
                    stats["assistant_turns"] += 1
                    content = t.get("content") or ""
                    stats["safety_warnings"].extend(safety_scan(content))
        roles = [t.get("role") for t in hist if isinstance(t, dict)]
        for i in range(len(roles) - 2):
            if roles[i] == roles[i + 1] == roles[i + 2]:
                stats["warnings"].append("连续三轮同角色")
                break
    if strict_safety and stats["safety_warnings"]:
        issues.append("助手回复含风险表述关键词: " + ", ".join(stats["safety_warnings"][:5]))
    if issues:
        return False, [f"第 {line_no} 行 (dialogue_id={data.get('dialogue_id', '?')}): " + "; ".join(issues)], stats
    return True, [], stats


def run_validation(
    input_path: Path,
    strict_length: bool = False,
    strict_safety: bool = False,
    score_dimensions: bool = False,
    score_llm: bool = False,
    api_key: Optional[str] = None,
    model: str = "qwen-plus",
) -> Dict[str, Any]:
    rows = load_jsonl(input_path)
    results = []
    all_ok = True
    total_stats = {
        "turns": [], "user_turns": [], "assistant_turns": [],
        "safety_warnings_count": 0,
        "flow_warnings_count": 0,
    }
    dim_sums = {d: 0 for d in DIMENSION_NAMES}
    dim_count = 0

    for idx, (line_no, data) in enumerate(rows):
        ok, issues, stats = validate_one(line_no, data, strict_length, strict_safety)
        if not ok:
            all_ok = False
        item = {"line_no": line_no, "ok": ok, "issues": issues, "stats": stats}
        if data is not None and isinstance(data, dict):
            item["dialogue_id"] = data.get("dialogue_id") or ""
        if score_dimensions and data is not None and ok:
            try:
                if score_llm and api_key:
                    item["dimension_scores"] = score_ten_dimensions_llm(data, api_key, model)
                    print(f"  LLM 评分进度: {dim_count + 1} 条已完成", file=sys.stderr, flush=True)
                    if idx > 0:
                        time.sleep(0.5)
                else:
                    item["dimension_scores"] = score_ten_dimensions(data)
                for d in DIMENSION_NAMES:
                    dim_sums[d] += item["dimension_scores"].get(d, 0)
                dim_count += 1
            except Exception as e:
                item["dimension_scores"] = {d: 0 for d in DIMENSION_NAMES}
                item["dimension_scores"]["总分"] = 0
                item["dimension_scores_error"] = str(e)
        results.append(item)
        if stats["turns"]:
            total_stats["turns"].append(stats["turns"])
            total_stats["user_turns"].append(stats["user_turns"])
            total_stats["assistant_turns"].append(stats["assistant_turns"])
        total_stats["safety_warnings_count"] += len(stats.get("safety_warnings", []))
        total_stats["flow_warnings_count"] += len(stats.get("warnings", []))

    valid_count = sum(1 for r in results if r["ok"])
    report = {
        "valid_count": valid_count,
        "invalid_count": len(results) - valid_count,
        "total_lines": len(results),
        "all_valid": all_ok,
        "details": results,
        "summary_stats": {
            "turns_min": min(total_stats["turns"]) if total_stats["turns"] else None,
            "turns_max": max(total_stats["turns"]) if total_stats["turns"] else None,
            "turns_avg": sum(total_stats["turns"]) / len(total_stats["turns"]) if total_stats["turns"] else None,
            "safety_warnings_total": total_stats["safety_warnings_count"],
            "flow_warnings_total": total_stats["flow_warnings_count"],
        },
    }
    if score_dimensions and dim_count > 0:
        report["dimension_avg"] = {
            d: round(dim_sums[d] / dim_count, 2) for d in DIMENSION_NAMES
        }
        report["dimension_avg"]["总分"] = round(
            sum(dim_sums[d] for d in DIMENSION_NAMES) / dim_count, 2
        )
        report["dimension_scored_count"] = dim_count
        report["dimension_score_mode"] = "llm" if score_llm else "rule"
    return report


def main():
    parser = argparse.ArgumentParser(description="多轮对话 jsonl 数据质量自动化校验")
    parser.add_argument("input", type=Path, nargs="?", default=Path("data/test_dia.jsonl"), help="输入 jsonl 路径")
    parser.add_argument("--output", "-o", type=Path, default=None, help="将报告写入 JSON 文件")
    parser.add_argument("--quiet", "-q", action="store_true", help="仅输出通过/不通过与计数")
    parser.add_argument("--strict-length", action="store_true", help="将用户/助手单轮长度超限视为不通过")
    parser.add_argument("--strict-safety", action="store_true", help="助手回复含风险表述关键词时视为不通过")
    parser.add_argument("--score-dimensions", action="store_true", help="按 human_review_guide 十维度做自动化评分（规则代理）")
    parser.add_argument("--score-llm", action="store_true", help="十维度评分改为调用 Qwen API（需设置 QWEN_API_KEY 或 DASHSCOPE_API_KEY，可从 .env 加载）")
    parser.add_argument("--model", type=str, default="qwen-plus", help="Qwen 模型名，如 qwen-plus、qwen-turbo（仅 --score-llm 时生效）")
    parser.add_argument("--output-scores", type=Path, default=None, help="将十维度评分写入 CSV（需同时使用 --score-dimensions）")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"错误: 文件不存在 {args.input}", file=sys.stderr)
        sys.exit(2)
    api_key = None
    if args.score_llm:
        api_key = _get_api_key()
        if not api_key:
            print("错误: --score-llm 需要 API Key。请设置环境变量 QWEN_API_KEY 或 DASHSCOPE_API_KEY，或在项目根目录 .env 中配置。", file=sys.stderr)
            sys.exit(2)
        if not args.score_dimensions:
            args.score_dimensions = True
    report = run_validation(
        args.input,
        strict_length=args.strict_length,
        strict_safety=args.strict_safety,
        score_dimensions=args.score_dimensions,
        score_llm=args.score_llm,
        api_key=api_key,
        model=args.model,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    if args.output_scores and args.score_dimensions and report.get("dimension_avg") is not None:
        args.output_scores.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(args.output_scores, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dialogue_id"] + DIMENSION_NAMES + ["总分", "备注"])
            for d in report["details"]:
                if not d.get("ok") or "dimension_scores" not in d:
                    continue
                row = [d.get("dialogue_id", "")]
                for dim in DIMENSION_NAMES:
                    row.append(d["dimension_scores"].get(dim, 0))
                row.append(d["dimension_scores"].get("总分", 0))
                row.append("")
                w.writerow(row)
    if args.quiet:
        print("PASS" if report["all_valid"] else "FAIL")
        print(f"valid={report['valid_count']} invalid={report['invalid_count']} total={report['total_lines']}")
        sys.exit(0 if report["all_valid"] else 1)
    print("=== 多轮对话数据质量校验报告 ===\n")
    print(f"总行数: {report['total_lines']}")
    print(f"通过: {report['valid_count']}")
    print(f"未通过: {report['invalid_count']}")
    if report["summary_stats"]["turns_avg"] is not None:
        s = report["summary_stats"]
        print(f"对话轮次: 最小={s['turns_min']} 最大={s['turns_max']} 平均={s['turns_avg']:.1f}")
    print(f"安全相关提示数: {report['summary_stats']['safety_warnings_total']}")
    print(f"流程提示数(如连续同角色): {report['summary_stats']['flow_warnings_total']}")
    if report.get("dimension_avg"):
        mode = report.get("dimension_score_mode", "rule")
        print("\n--- 十维度评分均值（与 human_review_guide 一致）---")
        print(f"  评分方式: {'Qwen API (LLM)' if mode == 'llm' else '规则代理'}")
        for dim in DIMENSION_NAMES:
            print(f"  {dim}: {report['dimension_avg'][dim]:.2f}")
        print(f"  总分（十维度之和）均值: {report['dimension_avg'].get('总分', 0):.2f}")
        print(f"  参与评分条数: {report.get('dimension_scored_count', 0)}")
    if report["invalid_count"] > 0:
        print("\n--- 未通过条目 ---")
        for d in report["details"]:
            if not d["ok"]:
                print(" ".join(d["issues"]))
    if args.output:
        print(f"\n报告已写入: {args.output}")
    sys.exit(0 if report["all_valid"] else 1)


if __name__ == "__main__":
    main()
