import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

# Define model parameters (in billions)
# ONLY models actually being tested
llm_parameters = {
    # Commercial models (estimated)
    "GPT-4o": 1760,
    "GPT4o": 1760,
    "gpt-4o": 1760,
    "Claude-3.7": 175,
    "Claude 3.7": 175,
    "claude-3.7": 175,
    "GPT-3.5": 20,
    "GPT35": 20,
    "gpt-3.5": 20,
    "Gemini": 150,
    "gemini": 150,
    # Open-source HuggingFace models (actual sizes)
    "Llama-3.2-1B": 1,
    "Llama32_1b": 1,
    "llama-3.2-1b": 1,
    "Mistral-7B": 7,
    "Mistral": 7,
    "mistral-7b": 7,
    "Llama-3.1-8B": 8,
    "Llama31_8B": 8,
    "llama-3.1-8b": 8,
    "Gemma-2-9b": 9,
    "Gemma2": 9,
    "gemma-2-9b": 9,
    "Qwen2.5-14B": 14,
    "Qwen_14": 14,
    "qwen2.5-14b": 14,
    "Cohere-aya-32B": 32,
    "aya-32b": 32,
    "aya_32b": 32,
    "Qwen2.5-32B": 32,
    "Qwen_32": 32,
    "qwen2.5-32b": 32,
    "Qwen2.5-72B": 72,
    "Qwen_72": 72,
    "qwen2.5-72b": 72,
}


def get_model_name_and_params(llm_string):
    """Extract a clean model name and parameter count from the LLM string."""
    llm_lower = llm_string.lower()

    # Check each known model (order matters - check specific before general)

    # Commercial models
    if "gpt-4o" in llm_lower or "gpt4o" in llm_lower:
        return "GPT-4o", 1760
    elif "claude" in llm_lower and "3.7" in llm_lower:
        return "Claude 3.7", 175
    elif "gpt-3.5" in llm_lower or "gpt35" in llm_lower:
        return "GPT-3.5", 20
    elif "gemini" in llm_lower:
        return "Gemini 2.0", 150

    # Qwen family (check specific sizes first)
    elif "qwen2.5-72b" in llm_lower or "qwen_72" in llm_lower.replace(".", "_"):
        return "Qwen 2.5 72B", 72
    elif "qwen2.5-32b" in llm_lower or "qwen_32" in llm_lower.replace(".", "_"):
        return "Qwen 2.5 32B", 32
    elif "qwen2.5-14b" in llm_lower or "qwen_14" in llm_lower.replace(".", "_"):
        return "Qwen 2.5 14B", 14

    # Llama family
    elif "llama-3.2-1b" in llm_lower or "llama32_1b" in llm_lower.replace(".", "_"):
        return "Llama 3.2 1B", 1
    elif "llama-3.1-8b" in llm_lower or "llama31_8b" in llm_lower.replace(".", "_"):
        return "Llama 3.1 8B", 8

    # Other models
    elif "gemma-2-9b" in llm_lower or "gemma2" in llm_lower:
        return "Gemma 2 9B", 9
    elif "mistral-7b" in llm_lower or "mistral" in llm_lower:
        return "Mistral 7B", 7
    elif "aya-32b" in llm_lower or "aya_32b" in llm_lower:
        return "Cohere Aya 32B", 32

    # Fallback - try to extract from parameters dict
    for key, params in llm_parameters.items():
        if key.lower() in llm_lower:
            return key, params

    # Last resort fallback
    clean_name = llm_string.split(":")[-1] if ":" in llm_string else llm_string
    return clean_name, None


def find_latest_quiz_file(category_folder):
    """Find the most recent quiz results file in a category folder."""
    if not os.path.exists(category_folder):
        return None

    json_files = list(Path(category_folder).glob("llm_scores_*.json"))
    if not json_files:
        return None

    # Sort by modification time, return most recent
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(json_files[0])


# File paths for quiz results - now auto-detects latest file
base_scores_dir = "openworm_ai/quiz/scores"

quiz_categories = {
    "General Knowledge": f"{base_scores_dir}/general",
    "Science": f"{base_scores_dir}/science",
    "C. elegans": f"{base_scores_dir}/celegans",
    "C. elegans (Corpus)": f"{base_scores_dir}/rag",
    "C. elegans (Corpus + RAG)": f"{base_scores_dir}/corpus_rag",
}

# Folder to save figures
figures_folder = "openworm_ai/quiz/figures"
os.makedirs(figures_folder, exist_ok=True)

# Color schemes
bar_color = "#4A90E2"  # Professional blue
scatter_color = "#E74C3C"  # Professional red

for category, folder_path in quiz_categories.items():
    # Find latest quiz file
    file_path = find_latest_quiz_file(folder_path)

    if not file_path:
        print(f"! No quiz results found for {category} in {folder_path}")
        continue

    print(f"Processing {category}: {file_path}")

    save_path = f"{figures_folder}/llm_accuracy_vs_parameters_{category.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').lower()}_bar.png"

    # Load JSON data
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except Exception as e:
        print(f"! Error loading {file_path}: {e}")
        continue

    # Extract relevant data
    filtered_results = []
    for result in data.get("Results", []):
        llm_string = result.get("LLM", "")
        accuracy = result.get("Accuracy (%)", 0)

        model_name, params = get_model_name_and_params(llm_string)

        if params is not None:
            filtered_results.append(
                {
                    "LLM": model_name,
                    "Accuracy": accuracy,
                    "Parameters": params,
                }
            )

    if not filtered_results:
        print(f"! No valid results for {category}")
        continue

    # Sort data by parameters
    filtered_results.sort(key=lambda x: x["Parameters"])

    # Extract x (models), y (accuracy), and parameters
    models = [entry["LLM"] for entry in filtered_results]
    y_accuracy = np.array([entry["Accuracy"] for entry in filtered_results])
    y_parameters = np.array([entry["Parameters"] for entry in filtered_results])

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()  # Create a second y-axis

    # Plot bar graph for accuracy
    ax1.bar(
        models,
        y_accuracy,
        color=bar_color,
        alpha=0.7,
        label="Accuracy (%)",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Accuracy (%)", color=bar_color, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="y", labelcolor=bar_color)
    ax1.tick_params(axis="x", rotation=45)

    # Plot scatter for parameters
    ax2.scatter(
        models,
        y_parameters,
        color=scatter_color,
        marker="D",
        s=100,
        label="Parameters (B)",
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )
    ax2.set_ylabel(
        "Number of Parameters (Billions)",
        color=scatter_color,
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_yscale("log")  # Log scale for better visualization
    ax2.tick_params(axis="y", labelcolor=scatter_color)

    # Add gridlines
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Title
    quiz_title = data.get("Title", category)
    plt.title(
        f"{quiz_title}\nAccuracy vs. Model Parameters",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Legends
    ax1.legend(loc="upper left", framealpha=0.9)
    ax2.legend(loc="upper right", framealpha=0.9)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")

    if "-nogui" not in sys.argv:
        plt.show()

    plt.close()

print(f"\n{'=' * 80}")
print(f"All figures saved to: {figures_folder}/")
print(f"{'=' * 80}")
