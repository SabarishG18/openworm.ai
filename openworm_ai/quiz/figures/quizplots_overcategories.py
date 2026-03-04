import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Define model parameters (in billions) - SAME AS QuizPlots.py
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
    
    # Commercial models
    if "gpt-4o" in llm_lower or "gpt4o" in llm_lower:
        return "GPT-4o", 1760
    elif "claude" in llm_lower and "3.7" in llm_lower:
        return "Claude 3.7", 175
    elif "gpt-3.5" in llm_lower or "gpt35" in llm_lower:
        return "GPT-3.5", 20
    elif "gemini" in llm_lower:
        return "Gemini 2.0", 150
    
    # Qwen family
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
    
    # Fallback
    for key, params in llm_parameters.items():
        if key.lower() in llm_lower:
            return key, params
    
    clean_name = llm_string.split(":")[-1] if ":" in llm_string else llm_string
    return clean_name, None


def find_latest_quiz_file(category_folder):
    """Find the most recent quiz results file in a category folder."""
    if not os.path.exists(category_folder):
        return None
    
    json_files = list(Path(category_folder).glob("llm_scores_*.json"))
    if not json_files:
        return None
    
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(json_files[0])


# File paths for quiz results - auto-detects latest file
base_scores_dir = "openworm_ai/quiz/scores"

# Define quiz categories and their difficulty order
quiz_categories = {
    "General Knowledge": f"{base_scores_dir}/general",
    "Science": f"{base_scores_dir}/science",
    "C. elegans": f"{base_scores_dir}/celegans",
    "C. elegans (Corpus)": f"{base_scores_dir}/rag",
    "C. elegans (Corpus+RAG)": f"{base_scores_dir}/corpus_rag",
}

# Folder to save figures
figures_folder = "openworm_ai/quiz/figures"
os.makedirs(figures_folder, exist_ok=True)

# Company color mapping (matching your image)
COMPANY_COLORS = {
    "OpenAI": "#4169E1",      # Blue
    "Google": "#EA4335",      # Red
    "Anthropic": "#D4A574",   # Tan/Gold
    "Microsoft": "#7F39FB",   # Purple
    "Alibaba": "#FF6A00",     # Orange
    "Meta": "#00A8E8",        # Cyan
    "Mistral AI": "#F2A900",  # Gold
    "Cohere": "#39594D",      # Dark green
}

def get_company_from_model(model_name: str) -> str:
    """Determine company from model name."""
    model_lower = model_name.lower()
    
    if "gpt" in model_lower:
        return "OpenAI"
    elif "gemini" in model_lower or "gemma" in model_lower:
        return "Google"
    elif "claude" in model_lower:
        return "Anthropic"
    elif "phi" in model_lower:
        return "Microsoft"
    elif "qwen" in model_lower:
        return "Alibaba"
    elif "llama" in model_lower:
        return "Meta"
    elif "mistral" in model_lower:
        return "Mistral AI"
    elif "aya" in model_lower or "cohere" in model_lower:
        return "Cohere"
    else:
        return "Other"

# Create an empty DataFrame to store results for all categories
df_all = pd.DataFrame()

# Process each quiz category
for category, folder_path in quiz_categories.items():
    file_path = find_latest_quiz_file(folder_path)
    
    if not file_path:
        print(f"! No quiz results found for {category} in {folder_path}")
        continue

    print(f"Processing {category}: {file_path}")
    
    # Load JSON data
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except Exception as e:
        print(f"! Error loading {file_path}: {e}")
        continue

    # Extract results
    category_results = []
    for result in data.get("Results", []):
        llm_string = result.get("LLM", "")
        accuracy = result.get("Accuracy (%)", 0)
        
        model_name, params = get_model_name_and_params(llm_string)
        
        if params is not None:
            category_results.append({
                "Model": model_name,
                "Quiz Category": category,
                "Accuracy (%)": accuracy,
                "Parameters (B)": params,
                "Company": get_company_from_model(model_name),
            })

    # Append results to the main DataFrame
    if category_results:
        df_category = pd.DataFrame(category_results)
        df_all = pd.concat([df_all, df_category], ignore_index=True)
        print(f"  ✓ Added {len(category_results)} results")

if df_all.empty:
    print("! No data to plot. Exiting...")
    sys.exit(1)

# Get the actual categories present in the data
actual_categories = df_all["Quiz Category"].unique()
print(f"\nCategories found: {list(actual_categories)}")

# Ensure quiz category is treated as an ordered category
# Natural progression: General → Science → C. elegans → Corpus (harder) → Corpus+RAG
category_order = []
for cat in ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)", "C. elegans (Corpus+RAG)"]:
    if cat in actual_categories:
        category_order.append(cat)

df_all["Quiz Category"] = pd.Categorical(
    df_all["Quiz Category"],
    categories=category_order,
    ordered=True,
)

# Sort models by parameter size for consistent legend ordering
model_order = df_all.groupby("Model")["Parameters (B)"].first().sort_values().index.tolist()

print(f"\nModels found: {model_order}")

# ============================================================================
# PLOT 1: Accuracy vs. Task Complexity (Line Plot)
# ============================================================================

plt.figure(figsize=(12, 7))

# Use a color palette with enough distinct colors
palette = sns.color_palette("husl", n_colors=len(model_order))

sns.lineplot(
    data=df_all,
    x="Quiz Category",
    y="Accuracy (%)",
    hue="Model",
    hue_order=model_order,  # Order by parameter size
    palette=palette,
    marker="o",
    linewidth=2.5,
    markersize=8,
)

# Improve readability
plt.title("LLM Performance Across Quiz Categories", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Quiz Category", fontsize=13, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=13, fontweight='bold')
plt.ylim(0, 100)
plt.xticks(rotation=25, ha='right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(title="Model (ordered by size)", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

plt.tight_layout()

# Save the figure
plot_path = os.path.join(figures_folder, "llm_performance_vs_task_complexity.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {plot_path}")

if "-nogui" not in sys.argv:
    plt.show()

plt.close()

# ============================================================================
# PLOT 2: Heatmap - Model Performance Across Categories
# ============================================================================

# Create pivot table for heatmap
pivot_df = df_all.pivot_table(
    index="Model",
    columns="Quiz Category",
    values="Accuracy (%)",
    aggfunc="mean"
)

# Reorder rows by parameter size
pivot_df = pivot_df.reindex(model_order)

plt.figure(figsize=(10, 8))

sns.heatmap(
    pivot_df,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Accuracy (%)'},
    linewidths=0.5,
    linecolor='gray'
)

plt.title("LLM Performance Heatmap Across Quiz Categories", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Quiz Category", fontsize=12, fontweight='bold')
plt.ylabel("Model (ordered by size)", fontsize=12, fontweight='bold')
plt.xticks(rotation=25, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

# Save heatmap
heatmap_path = os.path.join(figures_folder, "llm_performance_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {heatmap_path}")

if "-nogui" not in sys.argv:
    plt.show()

plt.close()

# ============================================================================
# PLOT 3: RAG Comparison (if RAG data available)
# ============================================================================

# Check if we have both Corpus and Corpus+RAG data
has_corpus = "C. elegans (Corpus)" in actual_categories
has_rag = "C. elegans (Corpus+RAG)" in actual_categories

if has_corpus and has_rag:
    print("\n📊 Generating RAG comparison plot...")
    
    # Filter for corpus vs corpus+RAG
    df_rag_comparison = df_all[
        df_all["Quiz Category"].isin(["C. elegans (Corpus)", "C. elegans (Corpus+RAG)"])
    ].copy()
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for grouped bar chart
    models = df_rag_comparison["Model"].unique()
    categories = ["C. elegans (Corpus)", "C. elegans (Corpus+RAG)"]
    
    x = np.arange(len(models))
    width = 0.35
    
    corpus_scores = []
    rag_scores = []
    
    for model in models:
        corpus_score = df_rag_comparison[
            (df_rag_comparison["Model"] == model) & 
            (df_rag_comparison["Quiz Category"] == "C. elegans (Corpus)")
        ]["Accuracy (%)"].values
        
        rag_score = df_rag_comparison[
            (df_rag_comparison["Model"] == model) & 
            (df_rag_comparison["Quiz Category"] == "C. elegans (Corpus+RAG)")
        ]["Accuracy (%)"].values
        
        corpus_scores.append(corpus_score[0] if len(corpus_score) > 0 else 0)
        rag_scores.append(rag_score[0] if len(rag_score) > 0 else 0)
    
    bars1 = ax.bar(x - width/2, corpus_scores, width, label='No RAG', color='#4A90E2', alpha=0.8)
    bars2 = ax.bar(x + width/2, rag_scores, width, label='With RAG', color='#2ECC71', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('RAG Impact on C. elegans Quiz Performance', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    rag_comparison_path = os.path.join(figures_folder, "rag_impact_comparison.png")
    plt.savefig(rag_comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {rag_comparison_path}")
    
    if "-nogui" not in sys.argv:
        plt.show()
    
    plt.close()

# ============================================================================
# PLOT 4: Company Comparison Scatter (for each quiz category)
# ============================================================================

print("\n📊 Generating company comparison scatter plots...")

for quiz_category in actual_categories:
    df_quiz = df_all[df_all["Quiz Category"] == quiz_category].copy()
    
    if df_quiz.empty:
        continue
    
    plt.figure(figsize=(10, 7))
    
    # Plot each company with its color
    for company in df_quiz["Company"].unique():
        df_company = df_quiz[df_quiz["Company"] == company]
        color = COMPANY_COLORS.get(company, "#999999")
        
        plt.scatter(
            df_company["Parameters (B)"],
            df_company["Accuracy (%)"],
            c=color,
            label=company,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
        )
        
        # Add model labels
        for _, row in df_company.iterrows():
            plt.annotate(
                row["Model"].replace(" ", "\n"),  # Line break for readability
                (row["Parameters (B)"], row["Accuracy (%)"]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8,
            )
    
    plt.xscale('log')
    plt.xlabel('Model Parameters (B)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(
        f'LLM Accuracy vs. Model Parameters - {quiz_category}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save with quiz category in filename
    safe_name = quiz_category.lower().replace(" ", "_").replace("(", "").replace(")", "")
    scatter_path = os.path.join(
        figures_folder,
        f"llm_accuracy_vs_parameters_{safe_name}.png"
    )
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {scatter_path}")
    
    if "-nogui" not in sys.argv:
        plt.show()
    
    plt.close()

print(f"\n{'='*80}")
print(f"All figures saved to: {figures_folder}/")
print(f"{'='*80}")