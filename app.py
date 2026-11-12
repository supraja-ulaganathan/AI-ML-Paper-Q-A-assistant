import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import gradio as gr

from answer import answer
from nlp_tools import keywords_for_paper, summarize_paper

PROCESSED = Path("data/processed")


def paper_choices():
    items = []
    for f in sorted(PROCESSED.glob("*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        items.append((p.get("title", p["id"])[:90], p["id"]))
    return items


def ask(question):
    if not question or not question.strip():
        return "Please type a question.", ""
    out = answer(question.strip())
    if not out["sources"]:
        return out["answer"], ""
    lines = []
    for i, s in enumerate(out["sources"], 1):
        link = f'[{s["title"]}]({s["url"]})' if s["url"] else s["title"]
        lines.append(f'**[{i}]** {link}, page {s["page"]} (match {s["score"]})')
    return out["answer"], "\n\n".join(lines)


def paper_info(paper_id):
    if not paper_id:
        return "Please choose a paper.", ""
    kws = ", ".join(keywords_for_paper(paper_id))
    summ = summarize_paper(paper_id)
    return kws, summ


with gr.Blocks(title="AI/ML Paper Q&A Assistant") as demo:
    gr.Markdown(
        "# AI/ML Paper Q&A Assistant\n"
        "Ask questions about a collection of arXiv papers on RAG and LLMs. "
        "Answers come only from the papers, with citations. "
        "If the papers don't contain the answer, it says so."
    )

    with gr.Tab("Ask a question"):
        question = gr.Textbox(label="Your question", placeholder="What is retrieval augmented generation?")
        ask_btn = gr.Button("Get answer", variant="primary")
        answer_box = gr.Markdown(label="Answer")
        gr.Markdown("### Sources")
        sources_box = gr.Markdown()
        gr.Examples(
            examples=[
                "What is retrieval augmented generation?",
                "How do language models hallucinate?",
                "How can RAG systems be evaluated?",
            ],
            inputs=question,
        )
        ask_btn.click(ask, inputs=question, outputs=[answer_box, sources_box])
        question.submit(ask, inputs=question, outputs=[answer_box, sources_box])

    with gr.Tab("Paper summary and keywords"):
        paper = gr.Dropdown(choices=paper_choices(), label="Choose a paper")
        info_btn = gr.Button("Analyze paper", variant="primary")
        kw_box = gr.Textbox(label="Keywords")
        sum_box = gr.Textbox(label="Summary", lines=5)
        info_btn.click(paper_info, inputs=paper, outputs=[kw_box, sum_box])

if __name__ == "__main__":
    demo.launch()