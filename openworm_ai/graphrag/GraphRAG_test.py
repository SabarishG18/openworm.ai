# Based on https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/

import glob
import json
import os
import sys

from dotenv import load_dotenv
from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore

# HF embeddings fallback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LLMs - Native LlamaIndex (no LangChain dependency)
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from openworm_ai import print_
from openworm_ai.utils.llms import get_llm_from_argv

load_dotenv()

STORE_DIR = "store"
SOURCE_DOCUMENT = "source document"

Settings.chunk_size = 3000
Settings.chunk_overlap = 50

SOURCE_REGISTRY_PATH = Path("corpus/papers/source_registry.json")


def load_source_registry(path: Path):
    if not path.exists():
        return {"papers": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def make_chunk_id(
    paper_ref: str, section_title: str, para_index: int, text: str
) -> str:
    base = f"{paper_ref}|{section_title}|{para_index}|{text[:80]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _has_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    return bool(key and key.strip())


def _select_embed_model():
    """
    Prefer OpenAI embeddings if a key is present, otherwise fall back to HF BGE small.
    Also fall back to HF if OpenAI embedding init fails for any reason.
    """
    if _has_openai_key():
        try:
            _ = Settings.embed_model
            print_("Embedding model: OpenAI (default)")
            return Settings.embed_model
        except Exception as e:
            print_(
                f"! OpenAI embeddings unavailable ({type(e).__name__}: {e}) -> falling back to HF BGE-small."
            )

    hf = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print_("Embedding model: HuggingFace BAAI/bge-small-en-v1.5")
    return hf


EMBED_MODEL = _select_embed_model()
Settings.embed_model = EMBED_MODEL


def _get_embedding_folder_name():
    if EMBED_MODEL.__class__.__name__.lower().startswith("openai"):
        return "embed_openai"

    if hasattr(EMBED_MODEL, "model_name"):
        name = EMBED_MODEL.model_name
        return "embed_" + name.replace("/", "_").replace(":", "_")

    return "embed_unknown"


def _make_llamaindex_llm(model: str):
    """
    Create a LlamaIndex LLM with smart fallbacks:
    1. Try OpenAI if key exists
    2. Try HuggingFace if token exists
    3. Fall back to Ollama (local, always works)
    """
    if isinstance(model, str) and model.startswith("ollama:"):
        local_name = model.split(":", 1)[1]
        print_(f"Using Ollama: {local_name}")
        return Ollama(model=local_name, request_timeout=60.0)

    if model.startswith("huggingface:"):
        hf_model = model.split(":", 1)[1]
        print_(f"Using HuggingFace Inference: {hf_model}")
        return HuggingFaceInferenceAPI(model_name=hf_model, token=os.getenv("HF_TOKEN"))

    if _has_openai_key():
        print_(f"Using OpenAI: {model}")
        return OpenAI(model=model)

    if os.getenv("HF_TOKEN"):
        print_(f"No OpenAI key, using HuggingFace default model instead of {model}")
        return HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-7B-Instruct")

    print_(f"No OpenAI/HF keys, using Ollama llama3.2 instead of {model}")
    return Ollama(model="llama3.2", request_timeout=60.0)


def get_embedding_model(model: str):
    """
    Get embedding model. Priority: HuggingFace API > Ollama > OpenAI default
    """
    # Check for embedding model override in env
    embed_model_env = os.getenv("NML_AI_EMBEDDING_MODEL") or os.getenv(
        "OPENWORM_AI_EMBEDDING_MODEL"
    )

    if embed_model_env and embed_model_env.startswith("huggingface:"):
        hf_model = strip_huggingface_prefix(embed_model_env)
        print_(f"Using HuggingFace API embedding: {hf_model}")
        from llama_index.embeddings.huggingface_api import (
            HuggingFaceInferenceAPIEmbedding,
        )

        return HuggingFaceInferenceAPIEmbedding(
            model_name=hf_model, token=get_hf_token()
        )

    if is_huggingface_model(model):
        hf_embed_model = "BAAI/bge-small-en-v1.5"
        print_(f"Using HuggingFace API embedding: {hf_embed_model}")
        from llama_index.embeddings.huggingface_api import (
            HuggingFaceInferenceAPIEmbedding,
        )

        return HuggingFaceInferenceAPIEmbedding(
            model_name=hf_embed_model, token=get_hf_token()
        )

    if is_ollama_model(model):
        ollama_model = normalize_ollama_model_name(model)
        print_(f"Using Ollama embedding: {ollama_model}")
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(model_name=ollama_model)

    print_("Using default embedding model")
    return None


def get_store_subfolder(model: str) -> str:
    if is_huggingface_model(model):
        hf_model = strip_huggingface_prefix(model)
        return "/" + hf_model.replace("/", "_").replace(":", "_")
    if is_ollama_model(model):
        ollama_model = normalize_ollama_model_name(model)
        return "/" + ollama_model.replace(":", "_")
    return ""


def create_store(model):
    json_inputs = glob.glob("processed/json/*/*.json")
    print_(f"Found {len(json_inputs)} JSON files to process")

    source_registry = load_source_registry(SOURCE_REGISTRY_PATH)
    papers_meta = source_registry.get("papers", {})

    documents = []
    for json_file in json_inputs:
        print_("Adding file to document store: %s" % json_file)

        with open(json_file, encoding="utf-8") as f:
            doc_model = json.load(f)

        for title in doc_model:
            print_("  Processing document: %s" % title)
            doc_contents = doc_model[title]
            src_page = doc_contents["source"]

            src_type = "Publication"
            if "wormatlas" in json_file:
                src_type = "WormAtlas Handbook"

            for section in doc_contents["sections"]:
                if "paragraphs" not in doc_contents["sections"][section]:
                    continue

                paragraphs = doc_contents["sections"][section]["paragraphs"]
                print_(
                    "    Processing section: %s\t(%i paragraphs)"
                    % (section, len(paragraphs))
                )

                # One Document per paragraph — fine-grained chunks for retrieval.
                # Uses paragraph-level metadata when present, falls back to
                # section info for older JSON files without metadata.
                for para_idx, p in enumerate(paragraphs):
                    text = p["contents"].strip()
                    if not text:
                        continue

                    page_num = p.get("page_number")
                    para_index = p.get("paragraph_index", para_idx)

                    if page_num is not None:
                        src_info = (
                            f"{src_type}: [{title}, p.{page_num}, "
                            f"para {para_index}]({src_page})"
                        )
                    else:
                        src_info = (
                            f"{src_type}: [{title}, Section {section}]({src_page})"
                        )

                    doc = Document(text=text, metadata={SOURCE_DOCUMENT: src_info})
                    documents.append(doc)

                    meta = {
                        "paper_ref": paper_ref,
                        "citation_short": citation_short,
                        "source_url": source_url,
                        "doi": doi,
                        "s2_paper_id": s2_paper_id,
                        "source_type": src_type,
                        "section_title": str(section_title),
                        "para_index": para_index,
                        "chunk_id": chunk_id,
                    }

    STORE_SUBFOLDER = "/" + _get_embedding_folder_name()

    index = VectorStoreIndex.from_documents(
        documents, embed_model=EMBED_MODEL, show_progress=True
    )

    print_("Persisting vector store index")
    index.storage_context.persist(persist_dir=STORE_DIR + STORE_SUBFOLDER)


def load_index(model):
    print_("Creating a storage context for %s" % model)

    STORE_SUBFOLDER = "/" + _get_embedding_folder_name()

    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(
            persist_dir=STORE_DIR + store_subfolder
        ),
        vector_store=SimpleVectorStore.from_persist_dir(
            persist_dir=STORE_DIR + store_subfolder
        ),
        index_store=SimpleIndexStore.from_persist_dir(
            persist_dir=STORE_DIR + store_subfolder
        ),
    )

    print_("Reloading index for %s" % model)
    index_reloaded = load_index_from_storage(storage_context)
    return index_reloaded


def get_query_engine(index_reloaded, model, similarity_top_k=4):
    print_("Creating query engine for %s" % model)

    text_qa_template_str = (
        "Context information is below.\n---------------------\n{context_str}\n"
        "---------------------\nUsing both the context information and your own knowledge, "
        "answer the question: {query_str}\nIf the context isn't helpful, you can also answer on your own.\n"
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    refine_template_str = (
        "The original question is: {query_str}\nWe have an existing answer: {existing_answer}\n"
        "We can refine it with more context below.\n------------\n{context_msg}\n------------\n"
        "Using both the new context and your knowledge, update or repeat the existing answer.\n"
    )
    refine_template = PromptTemplate(refine_template_str)

    llm = _make_llamaindex_llm(model)

    query_engine = index_reloaded.as_query_engine(
        llm=llm,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        embed_model=EMBED_MODEL,
    )

    query_engine.retriever.similarity_top_k = similarity_top_k
    return query_engine


def process_query(query, model, query_engine, verbose=False):
    print_("Processing query: %s" % query)
    response = query_engine.query(query)

    response_text = str(response)

    if "<think>" in response_text:
        response_text = (
            response_text[0 : response_text.index("<think>")]
            + response_text[response_text.index("</think>") + 8 :]
        )

    cutoff = 0.2
    files_used = []

    for sn in response.source_nodes:
        if verbose:
            print_("===================================")
            print_(sn.metadata["source document"])
            print_("-------")
            print_(f"Length of selection: {len(sn.text)}")
            print_(sn.text)

        sd = sn.metadata["source document"]
        if sd not in files_used:
            if len(files_used) == 0 or sn.score >= cutoff:
                files_used.append(f"{sd} (score: {sn.score})")

    file_info = ",\n   ".join(files_used)
    print_(
        f"""
===============================================================================
QUERY: {query}
MODEL: {model}
-------------------------------------------------------------------------------
RESPONSE: {response_text}
SOURCES:
   {file_info}
===============================================================================
"""
    )

    return response_text, response.metadata


if __name__ == "__main__":
    llm_ver = get_llm_from_argv(sys.argv)

    if "-test" not in sys.argv:
        if "-q" not in sys.argv:
            create_store(llm_ver)

        index_reloaded = load_index(llm_ver)
        query_engine = get_query_engine(index_reloaded, llm_ver)

        queries = [
            "What are the main differences between NeuroML versions 1 and 2?",
            "What are the main types of cells in the C. elegans pharynx?",
            "Give me 3 facts about the coelomocyte system in C. elegans",
            "Tell me about the neurotransmitter betaine in C. elegans",
            "Tell me about the different locomotory gaits of C. elegans",
        ]

        print_(f"Processing {len(queries)} queries")

        for query in queries:
            process_query(query, llm_ver, query_engine)
