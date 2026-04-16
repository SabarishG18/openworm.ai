"""
OpenWorm.ai Streamlit Demo
Calls the backend API, which wraps the RAG pipeline and MCP tools.
"""

import base64
import json
import os
import re
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
VS_CONFIG = os.getenv("GEN_RAG_VS_CONFIG", str(REPO_ROOT / "vector-stores.json"))
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8005")

# Models from our RAG evaluation benchmarks (current-03 branch,
# openworm_ai/quiz/scores/rag_full_logprob/), ordered by size.
AVAILABLE_MODELS = {
    # -- Qwen --
    "Qwen 2.5 72B Instruct": "huggingface:Qwen/Qwen2.5-72B-Instruct",
    "Qwen 2.5 32B Instruct": "huggingface:Qwen/Qwen2.5-32B-Instruct",
    "Qwen 3 32B": "huggingface:Qwen/Qwen3-32B",
    "Qwen 2.5 14B Instruct": "huggingface:Qwen/Qwen2.5-14B-Instruct",
    "Qwen 3 8B": "huggingface:Qwen/Qwen3-8B",
    "Qwen 2.5 7B Instruct": "huggingface:Qwen/Qwen2.5-7B-Instruct",
    # -- Meta Llama --
    "Llama 3.3 70B Instruct": "huggingface:meta-llama/Llama-3.3-70B-Instruct",
    "Llama 3.1 8B Instruct": "huggingface:meta-llama/Llama-3.1-8B-Instruct",
    "Llama 3.2 1B Instruct": "huggingface:meta-llama/Llama-3.2-1B-Instruct",
    # -- Mistral --
    "Mixtral 8x22B Instruct": "huggingface:mistralai/Mixtral-8x22B-Instruct-v0.1",
    "Mistral 7B Instruct v0.2": "huggingface:mistralai/Mistral-7B-Instruct-v0.2",
    # -- Google --
    "Gemma 3 27B": "huggingface:google/gemma-3-27b-it",
    "Gemma 2 9B": "huggingface:google/gemma-2-9b-it",
    # -- Microsoft --
    "Phi-3 Medium 14B": "huggingface:microsoft/Phi-3-medium-4k-instruct",
    "Phi-3 Mini 3.8B": "huggingface:microsoft/Phi-3-mini-4k-instruct",
    # -- DeepSeek --
    "DeepSeek R1": "huggingface:deepseek-ai/DeepSeek-R1",
    "DeepSeek R1 Distill Qwen 32B": "huggingface:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # -- Cohere --
    "Aya Expanse 32B": "huggingface:CohereLabs/aya-expanse-32b",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_source_document(source_doc: str):
    """Parse the 'source document' metadata string into components.

    Corpus formats:
      'Publication: [Author2012, p.1, para 3](https://...)'
      'WormAtlas Handbook: [Topic, Section X](https://...)'

    Returns (source_type, doc_name, detail, url).
      source_type: "Publication" or "WormAtlas Handbook"
      doc_name:    e.g. "Boyle,Berri,Cohen2012" or "Alimentary System"
      detail:      e.g. "p.1, para 3" or "Section 1) Overview"
      url:         the link, or None
    """
    md_match = re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", source_doc)
    if md_match:
        bracket_text = md_match.group(1).strip()
        url = md_match.group(2).strip()
        prefix = source_doc[: md_match.start()].strip().rstrip(":")

        parts = [p.strip() for p in bracket_text.split(",")]
        doc_name = parts[0] if parts else bracket_text
        detail = ", ".join(parts[1:]) if len(parts) > 1 else ""

        return prefix or "Source", doc_name, detail, url

    bare_match = re.search(r"(https?://\S+)", source_doc)
    if bare_match:
        url = bare_match.group(1).strip()
        title = source_doc.replace(url, "").strip() or source_doc
        return "Source", title, "", url

    return "Source", source_doc, "", None


def extract_references(reference_material: dict) -> list[dict]:
    """Group retrieved chunks by source document."""
    groups: dict[tuple, dict] = {}
    for _query, doc_score_pairs in reference_material.items():
        for doc, score in doc_score_pairs:
            source = doc.metadata.get("source document", "")
            source_type, doc_name, detail, url = parse_source_document(source)
            key = (doc_name, url or "")
            if key not in groups:
                groups[key] = {
                    "source_type": source_type,
                    "doc_name": doc_name,
                    "url": url,
                    "best_score": score,
                    "chunks": [],
                }
            g = groups[key]
            g["best_score"] = max(g["best_score"], score)
            preview = (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            )
            g["chunks"].append(
                {"detail": detail, "score": round(score, 3), "preview": preview}
            )

    refs = sorted(groups.values(), key=lambda r: r["best_score"], reverse=True)
    return refs


def split_answer_and_llm_refs(answer: str) -> tuple[str, str]:
    """Split the LLM answer into body and its generated References section."""
    pattern = re.compile(
        r"\n\s*(?:#{1,3}\s*)?(?:References|Sources|Bibliography)\s*:?\s*\n",
        re.IGNORECASE,
    )
    match = pattern.search(answer)
    if match:
        body = answer[: match.start()].rstrip()
        llm_refs = answer[match.end() :].strip()
        return body, llm_refs
    return answer, ""


def call_backend_chat(query: str, chat_model: str, messages: list[dict]):
    resp = requests.post(
        f"{API_BASE_URL}/chat",
        json={
            "query": query,
            "chat_model": chat_model,
            "messages": messages,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def call_backend_tool(tool_name: str, params: dict):
    resp = requests.post(
        f"{API_BASE_URL}/tool/run",
        json={
            "tool_name": tool_name,
            "params": params,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["output"]


def render_refs(refs):
    """Render grouped references."""
    for i, ref in enumerate(refs):
        n = len(ref["chunks"])
        label = ref["doc_name"]
        if ref["url"]:
            label = f"[{ref['doc_name']}]({ref['url']})"
        st.markdown(
            f"**{i + 1}. {ref['source_type']}**: {label} "
            f"&mdash; {n} chunk{'s' if n != 1 else ''}"
        )
        details = [c["detail"] for c in ref["chunks"] if c["detail"]]
        if details:
            st.caption("&nbsp;&nbsp;&nbsp;&nbsp;" + " | ".join(details))
        with st.expander("Relevance scores", expanded=False):
            for c in ref["chunks"]:
                where = c["detail"] or "chunk"
                st.text(f"  {where}: {c['score']}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="OpenWorm.ai", page_icon="🪱", layout="wide")
st.title("OpenWorm.ai")

with st.sidebar:
    st.header("Settings")
    selected_model_key = st.selectbox(
        "LLM model",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Select the language model for the assistant. "
        "Changing the model will reload the assistant.",
    )
    CHAT_MODEL = AVAILABLE_MODELS[selected_model_key]

    allow_fallback = st.toggle(
        "Allow parametric fallback",
        value=True,
        help="When ON, questions outside the corpus are answered from the "
        "model's training data. When OFF, only retrieval-backed answers "
        "are returned.",
    )
    show_refs = st.toggle("Show references", value=True)
    show_debug = st.toggle("Show debug info", value=False)

st.caption(
    f"Model: `{CHAT_MODEL.split(':',1)[-1]}` "
    f"&nbsp;|&nbsp; Vector store: `{VS_CONFIG}` "
    f"&nbsp;|&nbsp; API: `{API_BASE_URL}`"
)

tab_chat, tab_tools = st.tabs(["Chat", "MCP Tools"])

# ======================== TAB 1: RAG CHAT ========================
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("plot_base64") or msg.get("plot_data"):
                with st.expander("📈 Voltage trace", expanded=False):
                    if msg.get("plot_base64"):
                        st.image(
                            base64.b64decode(msg["plot_base64"]),
                            caption="Hodgkin-Huxley simulation voltage trace",
                            use_container_width=True,
                        )
                    elif msg.get("plot_data"):
                        _pd = msg["plot_data"]
                        if _pd.get("t_ms") and _pd.get("v_mv"):
                            import pandas as pd
                            _df = pd.DataFrame({
                                "Time (ms)": _pd["t_ms"],
                                "Voltage (mV)": _pd["v_mv"],
                            })
                            st.line_chart(_df, x="Time (ms)", y="Voltage (mV)")
            if msg.get("refs") and show_refs:
                with st.expander(f"📚 Retrieved sources ({len(msg['refs'])})"):
                    render_refs(msg["refs"])
            if msg.get("llm_refs"):
                with st.expander("📝 Citations found within retrieved sources"):
                    st.caption(
                        "*These citations appear within the retrieved documents "
                        "but are not directly verified from our corpus.*"
                    )
                    st.markdown(msg["llm_refs"])
            if msg.get("source_tag"):
                st.caption(msg["source_tag"])

    if prompt := st.chat_input("Ask about C. elegans..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                ]
                final_state = call_backend_chat(prompt, CHAT_MODEL, history)

            answer = final_state.get("message_for_user", "I was unable to answer.")
            ref_material = final_state.get("reference_material", {})
            query_type = final_state.get("query_type", None)
            plot_b64 = final_state.get("plot_base64", "")
            plot_data = final_state.get("plot_data", {})

            has_retrieval = bool(ref_material)
            is_tool = query_type == "task"

            if is_tool:
                source_tag = "🔧 Answer from MCP tool"
            elif has_retrieval:
                source_tag = "📚 Answer from corpus retrieval"
            else:
                if allow_fallback:
                    source_tag = "🧠 Answer from model knowledge (no corpus match)"
                else:
                    answer = (
                        "I couldn't find relevant information in the C. elegans corpus "
                        "for this query. Try rephrasing, or enable parametric fallback "
                        "in the sidebar."
                    )
                    source_tag = "❌ No corpus match (parametric fallback disabled)"

            answer_body, llm_refs_text = split_answer_and_llm_refs(answer)

            st.markdown(answer_body)

            if plot_b64:
                st.image(
                    base64.b64decode(plot_b64),
                    caption="Hodgkin-Huxley simulation voltage trace",
                    use_container_width=True,
                )
            elif plot_data and plot_data.get("t_ms") and plot_data.get("v_mv"):
                import pandas as pd
                df = pd.DataFrame({
                    "Time (ms)": plot_data["t_ms"],
                    "Voltage (mV)": plot_data["v_mv"],
                })
                st.line_chart(df, x="Time (ms)", y="Voltage (mV)")

            st.caption(source_tag)

            refs = extract_references(ref_material) if has_retrieval else []
            if refs and show_refs:
                with st.expander(f"📚 Retrieved sources ({len(refs)})"):
                    render_refs(refs)

            if llm_refs_text:
                with st.expander("📝 Citations found within retrieved sources"):
                    st.caption(
                        "*These citations appear within the retrieved documents "
                        "but are not directly verified from our corpus.*"
                    )
                    st.markdown(llm_refs_text)

            if show_debug:
                with st.expander("Debug"):
                    st.json(
                        {
                            "query_domain": query_domain,
                            "query_type": query_type,
                            "has_retrieval": has_retrieval,
                            "num_refs": len(refs),
                            "has_plot": bool(plot_b64 or plot_data),
                            "eval": str(final_state.get("text_response_eval", "N/A")),
                        }
                    )

            msg_entry = {
                "role": "assistant",
                "content": answer_body,
                "refs": refs,
                "llm_refs": llm_refs_text,
                "source_tag": source_tag,
            }
            if plot_b64:
                msg_entry["plot_base64"] = plot_b64
            if plot_data:
                msg_entry["plot_data"] = plot_data
            st.session_state.messages.append(msg_entry)

# ======================== TAB 2: MCP TOOLS ========================
with tab_tools:
    st.markdown(
        "**Test MCP tools directly** via backend API "
        f"`{API_BASE_URL}`"
    )

    tool_choice = st.selectbox(
        "Tool",
        ["HH Simulator", "WormBase: Gene", "WormBase: Neuron",
         "WormBase: Phenotype", "WormBase: Search"],
    )

    if tool_choice == "HH Simulator":
        st.markdown("Run a Hodgkin-Huxley squid axon simulation (scipy, no NEURON needed).")
        col1, col2 = st.columns(2)
        with col1:
            hh_current = st.number_input("Current injection (µA/cm²)", value=10.0, step=0.5)
            hh_duration = st.number_input("Duration (ms)", value=100.0, step=10.0)
        with col2:
            hh_delay = st.number_input("Delay before injection (ms)", value=5.0, step=5.0)
            hh_temp = st.number_input("Temperature (°C)", value=6.3, step=1.0)

        if st.button("Run simulation", key="hh_run"):
            with st.spinner("Running HH simulation..."):
                try:
                    raw = call_backend_tool("run_hh_simulation_tool", {
                        "current_injection": hh_current,
                        "duration": hh_duration,
                        "delay": hh_delay,
                        "temperature": hh_temp,
                    })
                    data = json.loads(raw)
                    returncode = data.get("returncode", -1)
                    stderr = data.get("stderr", "")

                    if returncode != 0:
                        st.error(f"Simulation failed (return code {returncode})")
                        if stderr:
                            st.code(stderr)
                    else:
                        stdout = json.loads(data.get("stdout", "{}"))
                        st.success("Simulation complete")

                        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                        mcol1.metric("Action potentials", stdout.get("num_action_potentials", "?"))
                        mcol2.metric("Firing rate", f"{stdout.get('firing_rate_hz', '?')} Hz")
                        mcol3.metric("Peak voltage", f"{stdout.get('peak_voltage_mv', '?'):.1f} mV")
                        mcol4.metric("Resting voltage", f"{stdout.get('resting_voltage_mv', '?'):.1f} mV")

                        trace = stdout.get("voltage_trace", {})
                        t_ms = trace.get("t_ms", [])
                        v_mv = trace.get("v_mv", [])
                        if t_ms and v_mv:
                            import pandas as pd
                            df = pd.DataFrame({"Time (ms)": t_ms, "Voltage (mV)": v_mv})
                            st.line_chart(df, x="Time (ms)", y="Voltage (mV)")

                        with st.expander("Raw output"):
                            st.json(stdout)

                except Exception as e:
                    st.error(f"Error: {e}")

    elif tool_choice == "WormBase: Gene":
        st.markdown("Query WormBase for gene information. Accepts gene names (eat-4) or WBGene IDs.")
        gene_id = st.text_input("Gene name or WBGene ID", value="eat-4")
        gene_field = st.selectbox("Field", [
            "overview", "phenotype", "expression", "function",
            "genetics", "homology", "interactions", "references",
        ])
        if st.button("Query gene", key="gene_run"):
            with st.spinner(f"Querying WormBase for {gene_id}..."):
                try:
                    raw = call_backend_tool("query_wormbase_gene_tool", {
                        "gene_id": gene_id, "field": gene_field,
                    })
                    data = json.loads(raw)
                    if "error" in data:
                        st.error(f"WormBase error: {data['error']}")
                        if data.get("resolved_id"):
                            st.info(f"Gene resolved to: {data['resolved_id']}")
                    else:
                        fields = data.get("fields", data)
                        name_data = fields.get("name", {}).get("data", {})
                        if isinstance(name_data, dict):
                            st.subheader(f"{name_data.get('label', gene_id)} ({name_data.get('id', '')})")
                        conc = fields.get("concise_description", {}).get("data", {})
                        if isinstance(conc, dict) and conc.get("text"):
                            st.markdown(conc["text"])
                        elif isinstance(conc, str):
                            st.markdown(conc)
                    with st.expander("Raw JSON"):
                        st.json(data)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif tool_choice == "WormBase: Neuron":
        st.markdown("Query WormBase for neuron/anatomy info. Uses the `anatomy_term` endpoint.")
        st.caption("Note: WormBase REST API neuron endpoints may return 500 errors — this is a known server-side issue.")
        neuron_name = st.text_input("Neuron name", value="AVAL")
        neuron_field = st.selectbox("Field", [
            "overview", "anatomy_function", "expressed_in", "innervates", "references",
        ])
        if st.button("Query neuron", key="neuron_run"):
            with st.spinner(f"Querying WormBase for {neuron_name}..."):
                try:
                    raw = call_backend_tool("query_wormbase_neuron_tool", {
                        "neuron_name": neuron_name, "field": neuron_field,
                    })
                    data = json.loads(raw)
                    if "error" in data:
                        st.error(f"WormBase error: {data['error']}")
                    else:
                        fields = data.get("fields", data)
                        name_data = fields.get("name", {}).get("data", {})
                        if isinstance(name_data, dict):
                            st.subheader(name_data.get("label", neuron_name))
                        defn = fields.get("definition", {}).get("data", "")
                        if defn:
                            st.markdown(str(defn))
                    with st.expander("Raw JSON"):
                        st.json(data)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif tool_choice == "WormBase: Phenotype":
        st.markdown("Query WormBase for phenotype info. Use WBPhenotype IDs for best results.")
        st.caption("Example: `WBPhenotype:0000643` (uncoordinated)")
        phenotype_id = st.text_input("Phenotype ID", value="WBPhenotype:0000643")
        if st.button("Query phenotype", key="pheno_run"):
            with st.spinner(f"Querying WormBase for {phenotype_id}..."):
                try:
                    raw = call_backend_tool("query_wormbase_phenotype_tool", {
                        "phenotype_id": phenotype_id,
                    })
                    data = json.loads(raw)
                    if "error" in data:
                        st.error(f"WormBase error: {data['error']}")
                    else:
                        st.json(data)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif tool_choice == "WormBase: Search":
        st.markdown("Search WormBase by name or keyword.")
        search_query = st.text_input("Search term", value="unc-17")
        search_type = st.selectbox("Entity type", [
            "gene", "anatomy", "phenotype", "protein", "variation",
        ])
        search_limit = st.slider("Max results", 1, 20, 5)
        if st.button("Search", key="search_run"):
            with st.spinner(f"Searching WormBase for '{search_query}'..."):
                try:
                    raw = call_backend_tool("search_wormbase_tool", {
                        "query": search_query,
                        "entity_type": search_type,
                        "limit": search_limit,
                    })
                    data = json.loads(raw)
                    if "error" in data:
                        st.error(f"WormBase error: {data['error']}")
                    hits = data.get("hits", [])
                    if hits:
                        st.success(f"Found {len(hits)} result(s)")
                        for h in hits:
                            st.markdown(f"- **{h.get('name', '?')}** — `{h.get('id', '?')}` ({h.get('class', '')})")
                    elif "error" not in data:
                        st.warning("No results found")
                    with st.expander("Raw JSON"):
                        st.json(data)
                except Exception as e:
                    st.error(f"Error: {e}")