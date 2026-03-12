import streamlit as st
import json
import os
import io
import re
import pandas as pd
from docx import Document
from openai import OpenAI

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        default = {"openai_api_key": "", "openai_model": "gpt-4o", "papers": {}}
        save_config(default)
        return default
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs)


def detect_paper(text: str, papers: dict) -> tuple[str | None, str | None]:
    """Return (paper_id, paper_name) or (None, None)."""
    lower_text = text.lower()
    for pid, paper in papers.items():
        keyword = paper.get("detection_keyword", "")
        if keyword and keyword.lower() in lower_text:
            return pid, paper.get("name", pid)
    return None, None


def extract_answer(text: str, start_text: str, end_text: str) -> tuple[str, str | None]:
    """
    Extract the substring between start_text and end_text.
    Returns (extracted_answer, warning_message_or_None).
    """
    if not start_text:
        return "", "No start_text defined for this question."

    lower_text = text.lower()
    start_lower = start_text.lower()
    start_idx = lower_text.find(start_lower)

    if start_idx == -1:
        return "", f"Start text not found: '{start_text}'"

    answer_start = start_idx + len(start_text)

    if end_text:
        end_lower = end_text.lower()
        end_idx = lower_text.find(end_lower, answer_start)
        if end_idx == -1:
            answer = text[answer_start:].strip()
            return answer, f"End text not found: '{end_text}'. Extracted to end of document."
        answer = text[answer_start:end_idx].strip()
    else:
        answer = text[answer_start:].strip()

    return answer, None


# ---------------------------------------------------------------------------
# OpenAI marking
# ---------------------------------------------------------------------------

def mark_answer(client: OpenAI, model: str, rubric: str, max_score: int, answer: str) -> dict:
    """Call OpenAI and return {"score": int, "justification": str}."""
    if not answer.strip():
        return {"score": 0, "justification": "No answer found."}

    prompt = (
        "You are an exam marker. Mark the following candidate answer.\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Max score: {max_score}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        'Respond in JSON format only, with no extra text:\n'
        '{"score": <integer>, "justification": "<brief explanation>"}'
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        score = int(result.get("score", 0))
        score = max(0, min(score, max_score))
        justification = str(result.get("justification", ""))
        return {"score": score, "justification": justification}
    except (json.JSONDecodeError, ValueError):
        return {"score": 0, "justification": f"Could not parse OpenAI response: {raw}"}


def mark_document(client: OpenAI, model: str, text: str, paper: dict) -> list[dict]:
    """
    Mark all questions in a paper for a given document text.
    Returns a list of question result dicts.
    """
    results = []
    for qid, qdata in paper.get("questions", {}).items():
        label = qdata.get("label", qid)
        start_text = qdata.get("start_text", "")
        end_text = qdata.get("end_text", "")
        max_score = int(qdata.get("max_score", 1))
        weight = float(qdata.get("weight", 1.0))
        rubric = qdata.get("rubric", "")

        answer, warning = extract_answer(text, start_text, end_text)
        marking = mark_answer(client, model, rubric, max_score, answer)

        results.append({
            "qid": qid,
            "label": label,
            "answer": answer,
            "warning": warning,
            "score": marking["score"],
            "justification": marking["justification"],
            "max_score": max_score,
            "weight": weight,
        })
    return results


def compute_weighted_total(question_results: list[dict]) -> float | None:
    """Return percentage 0-100 rounded to 1 dp, or None if no questions."""
    denom = sum(q["max_score"] * q["weight"] for q in question_results)
    if denom == 0:
        return None
    numer = sum(q["score"] * q["weight"] for q in question_results)
    return round(numer / denom * 100, 1)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def score_color_widget(score: int, max_score: int, justification: str, label: str) -> None:
    """Display a question result with colour coding."""
    if max_score == 0:
        ratio = 0.0
    else:
        ratio = score / max_score

    header = f"**{label}** — {score}/{max_score}"

    if ratio >= 1.0:
        st.success(f"{header}\n\n{justification}")
    elif ratio > 0:
        st.warning(f"{header}\n\n{justification}")
    else:
        st.error(f"{header}\n\n{justification}")


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------

def init_session_state():
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "marking_results" not in st.session_state:
        st.session_state.marking_results = []


# ---------------------------------------------------------------------------
# Tab 1: Settings
# ---------------------------------------------------------------------------

def render_settings_tab():
    st.header("Settings")
    cfg = st.session_state.config

    api_key = st.text_input(
        "OpenAI API Key",
        value=cfg.get("openai_api_key", ""),
        type="password",
        key="settings_api_key",
    )

    model = st.selectbox(
        "OpenAI Model",
        options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"].index(
            cfg.get("openai_model", "gpt-4o")
        )
        if cfg.get("openai_model", "gpt-4o") in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        else 0,
        key="settings_model",
    )

    if st.button("Save Settings", key="save_settings"):
        st.session_state.config["openai_api_key"] = api_key
        st.session_state.config["openai_model"] = model
        save_config(st.session_state.config)
        st.success("Settings saved.")


# ---------------------------------------------------------------------------
# Tab 2: Paper & Rubric Config
# ---------------------------------------------------------------------------

def render_config_tab():
    st.header("Paper & Rubric Configuration")
    cfg = st.session_state.config
    papers = cfg.setdefault("papers", {})

    # --- Paper selector ---
    paper_options = ["— Add New Paper —"] + [
        f"{pid}: {p.get('name', pid)}" for pid, p in papers.items()
    ]
    selected_option = st.selectbox("Select Paper", paper_options, key="paper_select")

    is_new_paper = selected_option == "— Add New Paper —"

    if is_new_paper:
        st.subheader("New Paper")
        new_paper_name = st.text_input("Paper Name", key="new_paper_name")
        new_paper_keyword = st.text_input(
            "Detection Keyword",
            help="If the document contains this text (case-insensitive), it is identified as this paper.",
            key="new_paper_keyword",
        )
        if st.button("Create Paper", key="create_paper"):
            if not new_paper_name.strip():
                st.error("Paper name cannot be empty.")
            else:
                import uuid
                pid = "paper_" + uuid.uuid4().hex[:8]
                papers[pid] = {
                    "name": new_paper_name.strip(),
                    "detection_keyword": new_paper_keyword.strip(),
                    "questions": {},
                }
                save_config(cfg)
                st.success(f"Paper '{new_paper_name}' created.")
                st.rerun()
        return

    # Extract paper id from selection
    selected_pid = selected_option.split(":")[0].strip()
    paper = papers[selected_pid]

    st.subheader(f"Paper: {paper.get('name', selected_pid)}")

    col1, col2 = st.columns(2)
    with col1:
        paper_name_val = st.text_input("Paper Name", value=paper.get("name", ""), key=f"pname_{selected_pid}")
    with col2:
        paper_keyword_val = st.text_input(
            "Detection Keyword",
            value=paper.get("detection_keyword", ""),
            key=f"pkw_{selected_pid}",
        )

    col_save, col_delete = st.columns([1, 1])
    with col_save:
        if st.button("Save Paper Details", key=f"save_paper_{selected_pid}"):
            paper["name"] = paper_name_val.strip()
            paper["detection_keyword"] = paper_keyword_val.strip()
            save_config(cfg)
            st.success("Paper details saved.")
            st.rerun()
    with col_delete:
        if st.button("Delete Paper", key=f"del_paper_{selected_pid}", type="secondary"):
            del papers[selected_pid]
            save_config(cfg)
            st.warning(f"Paper '{paper.get('name', selected_pid)}' deleted.")
            st.rerun()

    st.divider()
    st.subheader("Questions")

    questions = paper.setdefault("questions", {})

    # Display existing questions
    questions_to_delete = []
    for qid, qdata in list(questions.items()):
        with st.expander(f"Question: {qdata.get('label', qid)}", expanded=False):
            q_label = st.text_input("Label", value=qdata.get("label", ""), key=f"qlabel_{selected_pid}_{qid}")
            q_start = st.text_input(
                "Start Text",
                value=qdata.get("start_text", ""),
                help="Text that marks the beginning of the answer region.",
                key=f"qstart_{selected_pid}_{qid}",
            )
            q_end = st.text_input(
                "End Text",
                value=qdata.get("end_text", ""),
                help="Text that marks the end of the answer region. Leave blank to extract to end of document.",
                key=f"qend_{selected_pid}_{qid}",
            )
            col_score, col_weight = st.columns(2)
            with col_score:
                q_max_score = st.number_input(
                    "Max Score",
                    min_value=0,
                    value=int(qdata.get("max_score", 1)),
                    step=1,
                    key=f"qmax_{selected_pid}_{qid}",
                )
            with col_weight:
                q_weight = st.number_input(
                    "Weight",
                    min_value=0.0,
                    value=float(qdata.get("weight", 1.0)),
                    step=0.1,
                    format="%.2f",
                    key=f"qweight_{selected_pid}_{qid}",
                )
            q_rubric = st.text_area(
                "Rubric",
                value=qdata.get("rubric", ""),
                height=120,
                key=f"qrubric_{selected_pid}_{qid}",
            )

            # Stage updates
            questions[qid]["label"] = q_label
            questions[qid]["start_text"] = q_start
            questions[qid]["end_text"] = q_end
            questions[qid]["max_score"] = q_max_score
            questions[qid]["weight"] = q_weight
            questions[qid]["rubric"] = q_rubric

            if st.button("Delete this Question", key=f"del_q_{selected_pid}_{qid}"):
                questions_to_delete.append(qid)

    for qid in questions_to_delete:
        del questions[qid]
    if questions_to_delete:
        save_config(cfg)
        st.rerun()

    # Add Question
    st.divider()
    if st.button("Add Question", key=f"add_q_{selected_pid}"):
        import uuid
        new_qid = "q_" + uuid.uuid4().hex[:6]
        questions[new_qid] = {
            "label": f"Question {len(questions) + 1}",
            "start_text": "",
            "end_text": "",
            "max_score": 1,
            "weight": 1.0,
            "rubric": "",
        }
        save_config(cfg)
        st.rerun()

    if st.button("Save All Questions", key=f"save_all_q_{selected_pid}", type="primary"):
        save_config(cfg)
        st.success("All questions saved.")


# ---------------------------------------------------------------------------
# Tab 3: Mark Papers
# ---------------------------------------------------------------------------

def render_marking_tab():
    st.header("Mark Papers")
    cfg = st.session_state.config

    api_key = cfg.get("openai_api_key", "").strip()
    model = cfg.get("openai_model", "gpt-4o")
    papers = cfg.get("papers", {})

    uploaded_files = st.file_uploader(
        "Upload candidate .docx files",
        type=["docx"],
        accept_multiple_files=True,
        key="docx_uploader",
    )

    mark_btn = st.button("Mark All", type="primary", disabled=not uploaded_files)

    if mark_btn:
        if not api_key:
            st.error("Please set your OpenAI API key in the Settings tab.")
            return
        if not papers:
            st.error("No papers configured. Please add papers in the Paper & Rubric Config tab.")
            return

        client = OpenAI(api_key=api_key)
        all_results = []

        with st.spinner("Marking papers…"):
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                text = extract_text_from_docx(file_bytes)
                pid, paper_name = detect_paper(text, papers)

                candidate_result = {
                    "filename": uploaded_file.name,
                    "paper_id": pid,
                    "paper_name": paper_name or "Unknown",
                    "question_results": [],
                    "weighted_total": None,
                }

                if pid is not None:
                    paper = papers[pid]
                    try:
                        q_results = mark_document(client, model, text, paper)
                        candidate_result["question_results"] = q_results
                        candidate_result["weighted_total"] = compute_weighted_total(q_results)
                    except Exception as e:
                        candidate_result["error"] = str(e)

                all_results.append(candidate_result)

        st.session_state.marking_results = all_results
        st.success("Marking complete.")

    results = st.session_state.marking_results

    if not results:
        return

    # --- Summary table ---
    st.subheader("Summary")
    summary_rows = []
    for r in results:
        summary_rows.append({
            "Filename": r["filename"],
            "Paper": r["paper_name"],
            "Weighted Score": f"{r['weighted_total']}%" if r["weighted_total"] is not None else "N/A",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.divider()

    # --- Per-candidate accordions ---
    for r in results:
        wt = r["weighted_total"]
        score_label = f"{wt}% (weighted)" if wt is not None else "Unknown paper"
        expander_title = f"📄 {r['filename']} — {r['paper_name']} — {score_label}"

        with st.expander(expander_title, expanded=False):
            if r.get("error"):
                st.error(f"Error during marking: {r['error']}")
                continue

            if r["paper_id"] is None:
                st.warning("Paper type could not be detected. No detection keyword matched.")
                continue

            st.markdown(f"**Paper detected:** {r['paper_name']}")

            if not r["question_results"]:
                st.info("No questions defined for this paper.")
                continue

            for qr in r["question_results"]:
                st.markdown(f"---\n**{qr['label']}**")

                if qr.get("warning"):
                    st.warning(f"Extraction warning: {qr['warning']}")

                with st.expander("Extracted Answer", expanded=False):
                    if qr["answer"]:
                        st.text(qr["answer"])
                    else:
                        st.caption("_(no answer extracted)_")

                score_color_widget(
                    qr["score"],
                    qr["max_score"],
                    qr["justification"],
                    f"Score",
                )

    # --- CSV Export ---
    st.divider()
    if st.button("Export CSV", key="export_csv"):
        csv_data = build_csv(results)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="marking_results.csv",
            mime="text/csv",
            key="download_csv",
        )


def build_csv(results: list[dict]) -> bytes:
    """Build a CSV from marking results."""
    # Collect all question labels across all results to create consistent columns
    all_q_labels: list[str] = []
    seen: set[str] = set()
    for r in results:
        for qr in r.get("question_results", []):
            key = qr["label"]
            if key not in seen:
                all_q_labels.append(key)
                seen.add(key)

    rows = []
    for r in results:
        row: dict = {
            "Filename": r["filename"],
            "PaperType": r["paper_name"],
        }
        # Build a lookup from label to result for this candidate
        q_lookup = {qr["label"]: qr for qr in r.get("question_results", [])}
        for qlabel in all_q_labels:
            if qlabel in q_lookup:
                qr = q_lookup[qlabel]
                row[f"{qlabel}_Score"] = qr["score"]
                row[f"{qlabel}_Justification"] = qr["justification"]
            else:
                row[f"{qlabel}_Score"] = ""
                row[f"{qlabel}_Justification"] = ""
        row["TotalWeightedScore"] = (
            f"{r['weighted_total']}%" if r["weighted_total"] is not None else "N/A"
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Exam Marking Assistant",
        page_icon="📝",
        layout="wide",
    )
    st.title("📝 Exam Marking Assistant")

    init_session_state()

    tab_settings, tab_config, tab_marking = st.tabs(
        ["⚙️ Settings", "📋 Paper & Rubric Config", "✅ Mark Papers"]
    )

    with tab_settings:
        render_settings_tab()

    with tab_config:
        render_config_tab()

    with tab_marking:
        render_marking_tab()


if __name__ == "__main__":
    main()
