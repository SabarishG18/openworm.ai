import json
import time
import random
import datetime
import glob
from typing import List, Tuple

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

# ruff: noqa: F401
from openworm_ai.utils.llms import (
    LLM_HF_QWEN25_72B,
    LLM_HF_QWEN25_14B,
    LLM_HF_QWEN25_32B,
    LLM_HF_LLAMA31_8B,
    LLM_HF_GEMMA_2_9B,
    LLM_HF_LLAMA32_1b,
    LLM_HF_COHERE_AYA_32B,
    LLM_HF_MISTRAL_7B,
    LLM_GPT4o,
    LLM_GEMINI_2F,
    LLM_CLAUDE37,
    LLM_GPT35,
    ask_question_get_response,
)

from openworm_ai.quiz.Templates import ASK_Q

from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
field = "corpus_rag"
iteration_per_day = 1
current_date = datetime.datetime.now().strftime("%d-%m-%y")
SOURCE_QUESTIONS_FILE = "openworm_ai/quiz/samples/huggingface_Qwen_Qwen2.5-72B-Instruct_100questions_celegans_corpus.json"
OUTPUT_FILENAME = f"llm_scores_{field}_{current_date}_{iteration_per_day}.json"
SAVE_DIRECTORY = f"openworm_ai/quiz/scores/{field}"
TITLE = "Performance of LLMs on Corpus-Based C. elegans Quiz WITH RAG (retrieval-augmented answering)"

CORPUS_GLOB = "processed/json/papers/*.json"
RETRIEVAL_TOP_K = 3  # How many relevant passages to retrieve per question

indexing = ["A", "B", "C", "D"]


def get_embed_model_for_llm(llm_ver: str):
    """Get embedding model (same as QuizMasterCorpus)."""
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")

    if llm_ver.startswith("huggingface:") or not openai_key:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

    return OpenAIEmbedding()


def load_corpus_sections(papers_glob: str = CORPUS_GLOB) -> List[dict]:
    """Load sections from all processed paper JSONs (same as QuizMasterCorpus)."""
    json_inputs = glob.glob(papers_glob)
    sections: List[dict] = []

    if not json_inputs:
        print(f"! Warning: no JSON papers found under {papers_glob}")
        return sections

    for json_file in json_inputs:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"! Error reading {json_file}: {e}")
            continue

        for title, doc_contents in data.items():
            src_page = doc_contents.get("source", json_file)
            for section_name, details in doc_contents.get("sections", {}).items():
                sec_name_lower = section_name.lower()

                # Skip non-content sections
                if any(
                    key in sec_name_lower
                    for key in [
                        "reference",
                        "bibliograph",
                        "supplementary",
                        "acknowledg",
                        "funding",
                        "author contributions",
                        "materials and methods",
                    ]
                ):
                    continue

                paragraphs = details.get("paragraphs", [])
                text = " ".join(p.get("contents", "") for p in paragraphs).strip()

                if len(text.split()) < 30:
                    continue

                lower_text = text.lower()
                if "doi.org" in lower_text or "doi:" in lower_text:
                    continue

                src_info = f"{os.path.basename(json_file)}: [{title}, Section {section_name}]({src_page})"
                sections.append({"text": text, "source": src_info})

    print(f"[RAG] Loaded {len(sections)} sections from corpus papers")
    return sections


def build_corpus_index(
    llm_ver: str, papers_glob: str = CORPUS_GLOB
) -> Tuple[VectorStoreIndex, List[Document]]:
    """Build RAG index over corpus."""
    sections = load_corpus_sections(papers_glob=papers_glob)
    if not sections:
        raise ValueError("No sections found for corpus index.")

    docs: List[Document] = []
    for sid, sec in enumerate(sections):
        docs.append(
            Document(
                text=sec["text"],
                metadata={"source": sec["source"], "sid": sid},
            )
        )

    embed_model = get_embed_model_for_llm(llm_ver)
    print("[RAG] Building VectorStoreIndex for corpus...")

    import nest_asyncio

    nest_asyncio.apply()

    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    print(f"[RAG] Built index over {len(docs)} documents")
    return index, docs


def retrieve_context_for_question(
    question: str, index: VectorStoreIndex, top_k: int = RETRIEVAL_TOP_K
) -> str:
    """Retrieve relevant context from corpus for a given question."""
    retriever = index.as_retriever(similarity_top_k=top_k)

    try:
        results = retriever.retrieve(question)
    except Exception as e:
        print(f"! RAG retrieval failed: {e}")
        return ""

    context_parts = []
    for r in results:
        try:
            text = r.get_content()
            context_parts.append(text)
        except Exception:
            node = getattr(r, "node", None)
            if node:
                text = getattr(node, "text", "")
                if text:
                    context_parts.append(text)

    return "\n\n---\n\n".join(context_parts)


def load_llms():
    """Loads only the selected LLMs: Ollama Llama3 and GPT-3.5."""
    llms = [
        # LLM_OLLAMA_LLAMA32_1B,
        # LLM_OLLAMA_LLAMA32_3B,
        LLM_GPT4o,
        #####LLM_GEMINI,
        LLM_CLAUDE37,
        LLM_GPT35,
        # LLM_OLLAMA_PHI4, - cant run local ones on my laptop
        # LLM_OLLAMA_GEMMA2,
        # LLM_OLLAMA_DEEPSEEK - unable to answer A-D(too few params?),
        # LLM_OLLAMA_GEMMA,
        # LLM_OLLAMA_QWEN,
        # LLM_OLLAMA_TINYLLAMA,
        # LLM_OLLAMA_FALCON2 - 'only an assistant with no acess to external resources',
        # LLM_OLLAMA_CODELLAMA - understands only a fraction of questions, doesnt understand prompts
        # Small models (1-7B)
        LLM_HF_LLAMA32_1b,  # 1B - baseline
        LLM_HF_MISTRAL_7B,  # 14B - good small model
        # Mid-small (7-10B)
        LLM_HF_GEMMA_2_9B,  # 9B
        # Mid-range (10-30B) - ADD THESE:
        LLM_HF_QWEN25_14B,  # 14B - Qwen family comparison
        LLM_HF_QWEN25_32B,  # 32B - Best price/performance
        # or
        LLM_HF_LLAMA31_8B,  # 8B - Meta's latest
        # Large (70B+)
        LLM_HF_QWEN25_72B,  # 72B - Your quiz generator
        LLM_HF_COHERE_AYA_32B,  # 32B - Cohere model
    ]

    return llms


def load_questions_from_json(filename):
    """Loads a structured quiz JSON file and extracts questions and answers."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "questions" not in data or not isinstance(data["questions"], list):
            raise ValueError(
                "Invalid JSON format: Missing or malformed 'questions' list."
            )

        questions = []
        for q in data["questions"]:
            if "question" in q and isinstance(q["question"], str) and "answers" in q:
                formatted_answers = [
                    {"ans": ans["ans"], "correct": ans["correct"]}
                    for ans in q["answers"]
                    if "ans" in ans and "correct" in ans
                ]
                if formatted_answers:
                    questions.append(
                        {"question": q["question"], "answers": formatted_answers}
                    )
        if len(questions) == 0:
            raise ValueError("Error: No valid questions found in the JSON file.")

        return questions

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except json.JSONDecodeError:
        print(
            f"Error: Failed to decode JSON. Check that '{filename}' is properly formatted."
        )
        return []
    except ValueError as e:
        print(f"Error: {e}")
        return []


def format_rag_prompt(question_text: str, answers: str, context: str) -> str:
    """Format the prompt with RAG context."""
    if context.strip():
        # WITH RAG: Provide context
        rag_prompt = f"""You are answering a multiple-choice question about C. elegans research.

**CONTEXT FROM RESEARCH PAPERS:**
{context}

**QUESTION:**
{question_text}

**OPTIONS:**
{answers}

Based on the context provided above, select the correct answer (A, B, C, or D).
Answer with just the letter, nothing else."""
    else:
        # WITHOUT RAG: Fallback to normal prompt
        rag_prompt = ASK_Q.replace("<QUESTION>", question_text).replace(
            "<ANSWERS>", answers
        )

    return rag_prompt


def evaluate_llm_with_rag(llm, questions, index: VectorStoreIndex, temperature=0):
    """Evaluate LLM with RAG-augmented prompts."""
    results = {
        "LLM": llm,
        "Total Questions": len(questions),
        "Correct Answers": 0,
        "Correct with RAG": 0,
        "Correct without RAG": 0,
        "Questions with Context": 0,
        "Response Times": [],
    }

    for question_data in questions:
        question_text = question_data["question"]
        answers = question_data["answers"]

        # Shuffle answers
        random.shuffle(answers)

        # Assign answer labels
        presented_answers = {}
        correct_answer = None

        for idx, answer in enumerate(answers):
            ref = indexing[idx]
            formatted_answer = f"{ref}: {answer['ans']}"
            presented_answers[ref] = formatted_answer
            if answer["correct"]:
                correct_answer = ref

        answers_text = "\n".join(presented_answers.values())

        # Retrieve RAG context
        context = retrieve_context_for_question(
            question_text, index, top_k=RETRIEVAL_TOP_K
        )
        has_context = bool(context.strip())

        if has_context:
            results["Questions with Context"] += 1

        # Format prompt with/without RAG
        full_question = format_rag_prompt(question_text, answers_text, context)

        # Ask the LLM
        start_time = time.time()
        response = ask_question_get_response(
            full_question, llm, temperature, print_question=False
        ).strip()
        response_time = time.time() - start_time

        # Process response
        guess = response.split(":")[0].strip()
        if " " in guess:
            guess = guess[0]

        correct_guess = guess == correct_answer
        if correct_guess:
            results["Correct Answers"] += 1
            if has_context:
                results["Correct with RAG"] += 1
            else:
                results["Correct without RAG"] += 1

        results["Response Times"].append(response_time)

        rag_indicator = "✓ RAG" if has_context else "✗ No RAG"
        print(
            f" >> [{rag_indicator}] LLM ({llm}) - Q: {question_text[:50]}... | "
            f"Guess: {guess} | Correct: {correct_answer} | {correct_guess}"
        )

    # Compute stats
    results["Accuracy (%)"] = round(
        100 * results["Correct Answers"] / results["Total Questions"], 2
    )
    results["Accuracy with RAG (%)"] = (
        round(
            100
            * results["Correct with RAG"]
            / max(1, results["Questions with Context"]),
            2,
        )
        if results["Questions with Context"] > 0
        else 0.0
    )
    results["Accuracy without RAG (%)"] = (
        round(
            100
            * results["Correct without RAG"]
            / max(1, results["Total Questions"] - results["Questions with Context"]),
            2,
        )
        if (results["Total Questions"] - results["Questions with Context"]) > 0
        else 0.0
    )
    results["Avg Response Time (s)"] = round(
        sum(results["Response Times"]) / results["Total Questions"], 3
    )
    del results["Response Times"]

    return results


def iterate_over_llms(questions, index: VectorStoreIndex, temperature=0):
    """Iterate over all selected LLMs with RAG."""
    llms = load_llms()
    evaluation_results = []

    for llm in llms:
        print(f"\n{'=' * 80}")
        print(f"Evaluating LLM with RAG: {llm}")
        print(f"{'=' * 80}")
        llm_results = evaluate_llm_with_rag(llm, questions, index, temperature)
        evaluation_results.append(llm_results)
        print(f"\n✓ {llm} completed:")
        print(f"  Overall: {llm_results['Accuracy (%)']}%")
        print(f"  With RAG: {llm_results['Accuracy with RAG (%)']}%")
        print(f"  Without RAG: {llm_results['Accuracy without RAG (%)']}%")

    return evaluation_results


def save_results_to_json(
    results, filename=OUTPUT_FILENAME, save_path=SAVE_DIRECTORY, title=TITLE
):
    """Save results to JSON."""
    if save_path:
        file_path = f"{save_path}/{filename}"
    else:
        file_path = f"openworm_ai/quiz/scores/{filename}"

    current_datetime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    output_data = {
        "Title": title,
        "Quiz Type": "Corpus-Based with RAG (retrieval-augmented answering)",
        "RAG Configuration": {
            "Top-K Retrieval": RETRIEVAL_TOP_K,
            "Corpus": CORPUS_GLOB,
        },
        "Date of Testing": current_datetime,
        "Source Quiz": SOURCE_QUESTIONS_FILE,
        "Results": results,
    }

    try:
        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"\n{'=' * 80}")
        print(f"Results saved to: {file_path}")
        print(f"{'=' * 80}")
    except FileNotFoundError:
        print(
            f"Error: Directory '{save_path or 'openworm_ai/quiz/scores'}' does not exist."
        )
    except Exception as e:
        print(f"Error saving JSON file: {e}")


def print_summary(results):
    """Print summary comparison table."""
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY - CORPUS-BASED QUIZ WITH RAG")
    print(f"{'=' * 80}")
    print(f"{'LLM':<40} {'Overall':<12} {'With RAG':<12} {'No RAG':<12}")
    print(f"{'-' * 80}")

    sorted_results = sorted(results, key=lambda x: x["Accuracy (%)"], reverse=True)

    for r in sorted_results:
        llm_name = r["LLM"].split(":")[-1] if ":" in r["LLM"] else r["LLM"]
        llm_name = llm_name[:38]
        print(
            f"{llm_name:<40} {r['Accuracy (%)']}%{'':<7} {r['Accuracy with RAG (%)']}%{'':<7} {r['Accuracy without RAG (%)']}%"
        )

    print(f"{'=' * 80}\n")


def main():
    """Main execution function."""
    print(f"\n{'=' * 80}")
    print(f"{TITLE}")
    print(f"{'=' * 80}")
    print(f"Source Quiz: {SOURCE_QUESTIONS_FILE}")
    print(f"RAG Configuration: Top-{RETRIEVAL_TOP_K} from {CORPUS_GLOB}")
    print(f"{'=' * 80}\n")

    # Load questions
    questions = load_questions_from_json(SOURCE_QUESTIONS_FILE)
    if not questions:
        print("No valid questions to process. Exiting...")
        return

    print(f"Loaded {len(questions)} questions from corpus-based quiz\n")

    # Build RAG index (use first LLM's embedding model)
    llms = load_llms()
    first_llm = llms[0]
    print(f"Building RAG index using embeddings for: {first_llm}\n")

    try:
        index, docs = build_corpus_index(first_llm, papers_glob=CORPUS_GLOB)
    except Exception as e:
        print(f"! Error building corpus index: {e}")
        print("Exiting...")
        return

    # Evaluate LLMs
    results = iterate_over_llms(questions, index)

    # Print summary
    print_summary(results)

    # Save results
    save_results_to_json(results, OUTPUT_FILENAME, SAVE_DIRECTORY, TITLE)


if __name__ == "__main__":
    main()
