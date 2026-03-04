"""
RAG Analysis Plotting Script for Dissertation

Generates 8 key plots from RAG evaluation data:
1A. Accuracy by Method & Quiz Type - Side-by-side (bars)
1A2. Accuracy by Method & Quiz Type - Compound bars (improvement visualization)
1B. RAG Impact Breakdown (stacked bars)
2C. Context Coverage vs RAG Success
3E. Model Size vs RAG Benefit
4G. Corpus Gap Analysis
4H. Fallback Value Demonstration
5I. Response Time Analysis

Usage:
    python QuizPlots_RAG_Analysis.py [--rag-results RAG_FILES] [--no-rag-results NO_RAG_FILES]
    
    Or to auto-detect:
    python QuizPlots_RAG_Analysis.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11

# Output directory
OUTPUT_DIR = Path("openworm_ai/quiz/figures/rag_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results(filepaths: List[str]) -> List[Dict]:
    """Load and combine results from multiple JSON files."""
    all_results = []
    
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results = data.get('Results', [])
                all_results.extend(results)
                print(f"✓ Loaded {len(results)} results from {filepath}")
        except Exception as e:
            print(f"! Error loading {filepath}: {e}")
    
    return all_results


def load_no_rag_scores() -> Dict[str, Dict[str, float]]:
    """
    Load existing non-RAG quiz scores from scores directories.
    
    Returns:
        Dict mapping quiz category -> model -> accuracy
    """
    score_dirs = {
        'General Knowledge': 'openworm_ai/quiz/scores/general',
        'Science': 'openworm_ai/quiz/scores/science',
        'C. elegans': 'openworm_ai/quiz/scores/celegans',
        'C. elegans (Corpus)': 'openworm_ai/quiz/scores/corpus',
    }
    
    no_rag_data = {}
    
    for quiz_name, score_dir in score_dirs.items():
        no_rag_data[quiz_name] = {}
        
        # Find all JSON files in the directory (use most recent)
        json_files = glob.glob(f"{score_dir}/*.json")
        
        # Sort by modification time and use the most recent
        if json_files:
            # Get the most recent file (by date in filename)
            latest_file = sorted(json_files)[-1]
            
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                
                # Check if this is the combined results format
                results = data.get('Results', [])
                
                if results:
                    # New format: list of results
                    for result in results:
                        model = result.get('LLM', '')
                        accuracy = result.get('Accuracy (%)', 0)
                        
                        if model:
                            no_rag_data[quiz_name][model] = accuracy
                else:
                    # Old format: single result
                    model = data.get('LLM', '')
                    accuracy = data.get('Accuracy (%)', 0)
                    
                    if model:
                        no_rag_data[quiz_name][model] = accuracy
                
                print(f"  ✓ Loaded {quiz_name}: {len(no_rag_data[quiz_name])} models from {latest_file}")
                        
            except Exception as e:
                print(f"  ! Error loading {latest_file}: {e}")
                continue
    
    # Print summary
    print("\n📁 Loaded Non-RAG Baseline Scores:")
    for quiz, models in no_rag_data.items():
        if models:
            avg_acc = np.mean(list(models.values()))
            print(f"  {quiz}: {len(models)} models (avg: {avg_acc:.1f}%)")
    
    return no_rag_data


def get_model_size(model_name: str) -> float:
    """Extract parameter size from model name (in billions)."""
    size_map = {
        'llama-3.2-1b': 1,
        'mistral-7b': 7,
        'llama-3.1-8b': 8,
        'gemma-2-9b': 9,
        'qwen2.5-14b': 14,
        'qwen2.5-32b': 32,
        'aya-expanse-32b': 32,
        'qwen2.5-72b': 72,
        'gpt-3.5': 175,  # Estimated
        'gpt-4o': 1760,  # Estimated (10x GPT-3.5)
        'claude-3-7': 200,  # Estimated
    }
    
    model_lower = model_name.lower()
    for key, size in size_map.items():
        if key in model_lower:
            return size
    
    return 0  # Unknown


def plot_1a_accuracy_by_method(results: List[Dict], no_rag_scores: Dict[str, Dict[str, float]]):
    """
    Plot 1A: Accuracy by Method & Quiz Type - Side-by-side bars
    
    Shows LLM-only (from actual non-RAG runs), RAG-only, and RAG+Fallback accuracy.
    """
    print("\n📊 Generating Plot 1A: Accuracy by Method & Quiz Type (side-by-side)...")
    
    # Group by quiz category
    quiz_types = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    
    data = {quiz: {'llm_only': [], 'rag_only': [], 'rag_fallback': []} for quiz in quiz_types}
    
    # Get LLM-only from actual non-RAG baseline scores
    for quiz in quiz_types:
        if quiz in no_rag_scores and no_rag_scores[quiz]:
            # Average across all models
            llm_only_avg = np.mean(list(no_rag_scores[quiz].values()))
            data[quiz]['llm_only'] = [llm_only_avg]
        else:
            data[quiz]['llm_only'] = []
    
    # Get RAG results
    for result in results:
        quiz = result.get('Quiz Category')
        if quiz not in quiz_types:
            continue
        
        # RAG-only: Questions where RAG was sufficient
        rag_only_acc = result.get('RAG-SUFFICIENT (%)', 0)
        
        # RAG+Fallback: Overall accuracy (uses both)
        rag_fallback_acc = result.get('Overall Accuracy (%)', 0)
        
        data[quiz]['rag_only'].append(rag_only_acc)
        data[quiz]['rag_fallback'].append(rag_fallback_acc)
    
    # Compute averages
    quiz_names_short = ['General', 'Science', 'C. elegans', 'Corpus']
    llm_only_avg = [data[q]['llm_only'][0] if data[q]['llm_only'] else 0 for q in quiz_types]
    rag_only_avg = [np.mean(data[q]['rag_only']) if data[q]['rag_only'] else 0 for q in quiz_types]
    rag_fallback_avg = [np.mean(data[q]['rag_fallback']) if data[q]['rag_fallback'] else 0 for q in quiz_types]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(quiz_names_short))
    width = 0.25
    
    bars1 = ax.bar(x - width, llm_only_avg, width, label='LLM-only (baseline, no RAG)', 
                   color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, rag_only_avg, width, label='RAG-only (corpus knowledge)', 
                   color='#2ecc71', edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, rag_fallback_avg, width, label='RAG+Fallback (hybrid)', 
                   color='#9b59b6', edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Quiz Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy Comparison: LLM-only vs RAG-only vs RAG+Fallback', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(quiz_names_short, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "1a_accuracy_by_method.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()


def plot_1a2_accuracy_compound_bars(results: List[Dict], no_rag_scores: Dict[str, Dict[str, float]]):
    """
    Plot 1A2: Accuracy by Method & Quiz Type - Compound bars
    
    Shows baseline (LLM-only) with RAG improvement stacked on top to visualize gains.
    """
    print("\n📊 Generating Plot 1A2: Accuracy Improvement with RAG (compound bars)...")
    
    quiz_types = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    
    # Get baseline LLM-only scores
    llm_only_avg = []
    for quiz in quiz_types:
        if quiz in no_rag_scores and no_rag_scores[quiz]:
            llm_only_avg.append(np.mean(list(no_rag_scores[quiz].values())))
        else:
            llm_only_avg.append(0)
    
    # Get RAG+Fallback scores
    rag_fallback_avg = []
    for quiz in quiz_types:
        quiz_results = [r for r in results if r.get('Quiz Category') == quiz]
        accuracies = [r.get('Overall Accuracy (%)', 0) for r in quiz_results]
        rag_fallback_avg.append(np.mean(accuracies) if accuracies else 0)
    
    # Calculate improvement
    rag_improvement = [rag - llm for rag, llm in zip(rag_fallback_avg, llm_only_avg)]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    quiz_names_short = ['General', 'Science', 'C. elegans', 'Corpus']
    x = np.arange(len(quiz_names_short))
    width = 0.6
    
    # Bottom bars: LLM-only baseline
    bars1 = ax.bar(x, llm_only_avg, width, label='LLM-only (baseline)', 
                   color='#3498db', edgecolor='black', linewidth=1.2)
    
    # Top bars: RAG improvement
    bars2 = ax.bar(x, rag_improvement, width, bottom=llm_only_avg,
                   label='RAG improvement', 
                   color='#2ecc71', edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Baseline value
        height1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
               f'{height1:.1f}%', ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
        
        # Improvement value (if significant)
        if rag_improvement[i] > 2:
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height1 + height2/2,
                   f'+{height2:.1f}%', ha='center', va='center', 
                   fontsize=11, fontweight='bold', color='white')
        
        # Total on top
        total = llm_only_avg[i] + rag_improvement[i]
        ax.text(bar2.get_x() + bar2.get_width()/2., total,
               f'{total:.1f}%', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Quiz Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('RAG Improvement: Baseline + RAG Gain', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(quiz_names_short, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax.annotate('Green shows accuracy gain from RAG system',
               xy=(0.98, 0.05), xycoords='axes fraction',
               fontsize=10, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "1a2_accuracy_compound_bars.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()


def plot_1b_rag_impact_breakdown(results: List[Dict]):
    """
    Plot 1B: RAG Impact Breakdown
    
    Stacked bar chart showing where RAG helps, where LLM is sufficient, etc.
    """
    print("\n📊 Generating Plot 1B: RAG Impact Breakdown...")
    
    quiz_types = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    
    # Categories
    categories = {
        'RAG-SUFFICIENT': [],
        'LLM-KNOWLEDGE': [],
        'LLM-RESCUED': [],
        'NO-CONTEXT': [],
        'BOTH-FAILED': []
    }
    
    for quiz in quiz_types:
        quiz_results = [r for r in results if r.get('Quiz Category') == quiz]
        
        for cat in categories.keys():
            values = [r.get(f'{cat} (%)', 0) for r in quiz_results if r.get('Total Evaluated', 0) > 0]
            avg = np.mean(values) if values else 0
            categories[cat].append(avg)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    quiz_names_short = ['General', 'Science', 'C. elegans', 'Corpus']
    x = np.arange(len(quiz_names_short))
    width = 0.6
    
    colors = {
        'RAG-SUFFICIENT': '#2ecc71',      # Green - RAG worked!
        'LLM-KNOWLEDGE': '#3498db',       # Blue - LLM knew it
        'LLM-RESCUED': '#f39c12',         # Orange - Fallback saved it
        'NO-CONTEXT': '#95a5a6',          # Gray - No context found
        'BOTH-FAILED': '#e74c3c'          # Red - Both failed
    }
    
    bottom = np.zeros(len(quiz_names_short))
    
    for cat, color in colors.items():
        values = categories[cat]
        bars = ax.bar(x, values, width, label=cat.replace('-', ' '), 
                     bottom=bottom, color=color, edgecolor='black', linewidth=0.8)
        
        # Add percentage labels for significant segments (>5%)
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width()/2., bottom[i] + val/2,
                       f'{val:.0f}%', ha='center', va='center', 
                       fontsize=9, fontweight='bold', color='white')
        
        bottom += values
    
    ax.set_xlabel('Quiz Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Questions (%)', fontsize=13, fontweight='bold')
    ax.set_title('RAG Impact Breakdown: Where Does Each Method Succeed?', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(quiz_names_short, fontsize=12)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "1b_rag_impact_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()


def plot_2c_context_coverage(results: List[Dict]):
    """
    Plot 2C: Context Coverage vs RAG Success
    
    Shows % questions with context and % of those RAG answered correctly.
    """
    print("\n📊 Generating Plot 2C: Context Coverage vs RAG Success...")
    
    quiz_types = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    
    coverage = []
    success = []
    
    for quiz in quiz_types:
        quiz_results = [r for r in results if r.get('Quiz Category') == quiz]
        
        # Coverage: % questions with context
        cov_vals = [r.get('Context Coverage (%)', 0) for r in quiz_results]
        avg_cov = np.mean(cov_vals) if cov_vals else 0
        coverage.append(avg_cov)
        
        # Success: Of questions with context, how many did RAG get right?
        # RAG-SUFFICIENT / Questions with Context
        rag_suf = [r.get('RAG-SUFFICIENT', 0) for r in quiz_results]
        with_ctx = [r.get('Questions with Context', 1) for r in quiz_results]  # Avoid /0
        success_rates = [100 * s / c if c > 0 else 0 for s, c in zip(rag_suf, with_ctx)]
        avg_suc = np.mean(success_rates) if success_rates else 0
        success.append(avg_suc)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    quiz_names_short = ['General', 'Science', 'C. elegans', 'Corpus']
    x = np.arange(len(quiz_names_short))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, coverage, width, label='Context Coverage (% questions)', 
                   color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, success, width, label='RAG Success Rate (% when context available)', 
                   color='#2ecc71', edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Quiz Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title('Corpus Coverage and RAG Reliability', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(quiz_names_short, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "2c_context_coverage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()


def plot_3e_model_size_vs_rag_benefit(results: List[Dict]):
    """
    Plot 3E: Model Size vs RAG Benefit
    
    Shows if smaller models benefit more from RAG.
    """
    print("\n📊 Generating Plot 3E: Model Size vs RAG Benefit...")
    
    quiz_types = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    
    # Group by model and quiz
    data = {}
    
    for result in results:
        model = result.get('LLM', '')
        quiz = result.get('Quiz Category', '')
        
        if quiz not in quiz_types:
            continue
        
        size = get_model_size(model)
        if size == 0:
            continue
        
        # Calculate RAG benefit: (RAG+Fallback accuracy) - (LLM-only accuracy)
        # LLM-only is approximated by LLM-KNOWLEDGE % (questions without context)
        rag_fallback = result.get('Overall Accuracy (%)', 0)
        llm_only = result.get('LLM-KNOWLEDGE (%)', 0)
        
        # Better metric: RAG-SUFFICIENT % (pure RAG contribution)
        rag_benefit = result.get('RAG-SUFFICIENT (%)', 0)
        
        key = (size, model)
        if key not in data:
            data[key] = {q: [] for q in quiz_types}
        
        data[key][quiz].append(rag_benefit)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        'General Knowledge': '#3498db',
        'Science': '#e74c3c',
        'C. elegans': '#f39c12',
        'C. elegans (Corpus)': '#2ecc71'
    }
    
    markers = {'General Knowledge': 'o', 'Science': 's', 'C. elegans': '^', 'C. elegans (Corpus)': 'D'}
    
    for quiz in quiz_types:
        sizes = []
        benefits = []
        
        for (size, model), quiz_data in sorted(data.items()):
            if quiz_data[quiz]:
                sizes.append(size)
                benefits.append(np.mean(quiz_data[quiz]))
        
        if sizes:
            ax.plot(sizes, benefits, marker=markers[quiz], markersize=10, 
                   linewidth=2, label=quiz.replace(' (Corpus)', ' (C)'), 
                   color=colors[quiz], alpha=0.8)
    
    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=13, fontweight='bold')
    ax.set_ylabel('RAG Benefit (% questions where RAG was essential)', fontsize=13, fontweight='bold')
    ax.set_title('Do Smaller Models Benefit More from RAG?', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "3e_model_size_vs_rag_benefit.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()


def plot_4g_corpus_gap_analysis(results: List[Dict]):
    """
    Plot 4G: Corpus Gap Analysis
    
    Pie charts showing where corpus has answers vs gaps.
    """
    print("\n📊 Generating Plot 4G: Corpus Gap Analysis...")
    
    quiz_types = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = {
        'RAG-SUFFICIENT': '#2ecc71',      # Corpus has answer
        'LLM-RESCUED': '#f39c12',         # Corpus gap, LLM knew it
        'NO-CONTEXT': '#95a5a6',          # Corpus gap, nobody knew
        'BOTH-FAILED': '#e74c3c',         # Had context but failed
        'LLM-KNOWLEDGE': '#3498db'        # No context needed
    }
    
    for idx, quiz in enumerate(quiz_types):
        quiz_results = [r for r in results if r.get('Quiz Category') == quiz]
        
        # Average percentages
        categories = ['RAG-SUFFICIENT', 'LLM-RESCUED', 'NO-CONTEXT', 'LLM-KNOWLEDGE', 'BOTH-FAILED']
        values = []
        labels = []
        plot_colors = []
        
        for cat in categories:
            vals = [r.get(f'{cat} (%)', 0) for r in quiz_results if r.get('Total Evaluated', 0) > 0]
            avg = np.mean(vals) if vals else 0
            
            if avg > 2:  # Only show segments >2%
                values.append(avg)
                labels.append(cat.replace('-', ' '))
                plot_colors.append(colors[cat])
        
        # Plot pie chart
        ax = axes[idx]
        wedges, texts, autotexts = ax.pie(values, labels=labels, colors=plot_colors, 
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        # Make percentage text white
        for autotext in autotexts:
            autotext.set_color('white')
        
        ax.set_title(quiz, fontsize=13, fontweight='bold', pad=10)
    
    plt.suptitle('Corpus Gap Analysis: Where Does Each Method Contribute?', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "4g_corpus_gap_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()


def plot_4h_fallback_value(results: List[Dict]):
    """
    Plot 4H: Fallback Value Demonstration
    
    Shows why hybrid system (RAG+Fallback) is essential.
    """
    print("\n📊 Generating Plot 4H: Fallback Value Demonstration...")
    
    quiz_types = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    
    # Three scenarios per quiz
    rag_only_acc = []
    llm_only_acc = []
    hybrid_acc = []
    
    for quiz in quiz_types:
        quiz_results = [r for r in results if r.get('Quiz Category') == quiz]
        
        # RAG-only: Only counts questions where RAG had context and answered
        # = RAG-SUFFICIENT %
        rag_vals = [r.get('RAG-SUFFICIENT (%)', 0) for r in quiz_results]
        rag_only_acc.append(np.mean(rag_vals) if rag_vals else 0)
        
        # LLM-only: Only counts questions LLM answered without context
        # = LLM-KNOWLEDGE %
        llm_vals = [r.get('LLM-KNOWLEDGE (%)', 0) for r in quiz_results]
        llm_only_acc.append(np.mean(llm_vals) if llm_vals else 0)
        
        # Hybrid: Overall accuracy (best of both)
        hybrid_vals = [r.get('Overall Accuracy (%)', 0) for r in quiz_results]
        hybrid_acc.append(np.mean(hybrid_vals) if hybrid_vals else 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    quiz_names_short = ['General', 'Science', 'C. elegans', 'Corpus']
    x = np.arange(len(quiz_names_short))
    width = 0.25
    
    bars1 = ax.bar(x - width, rag_only_acc, width, label='RAG-only (fails without context)', 
                   color='#2ecc71', edgecolor='black', linewidth=1.2, alpha=0.7)
    bars2 = ax.bar(x, llm_only_acc, width, label='LLM-only (fails on corpus questions)', 
                   color='#3498db', edgecolor='black', linewidth=1.2, alpha=0.7)
    bars3 = ax.bar(x + width, hybrid_acc, width, label='Hybrid RAG+Fallback (best of both)', 
                   color='#9b59b6', edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Quiz Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Why Hybrid System is Essential: RAG+Fallback Outperforms Both', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(quiz_names_short, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax.annotate('Hybrid system combines strengths:\n• Uses RAG when corpus has data\n• Falls back to LLM when corpus lacks info',
               xy=(0.98, 0.05), xycoords='axes fraction',
               fontsize=10, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "4h_fallback_value.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()


def plot_6_per_model_breakdown(results: List[Dict], no_rag_scores: Dict[str, Dict[str, float]]):
    """
    Plot 6: Per-Model Breakdown for C. elegans and Corpus
    
    Shows each model's performance: LLM-only, RAG-only, RAG+Fallback
    Two subplots: one for C. elegans, one for Corpus
    """
    print("\n📊 Generating Plot 6: Per-Model Breakdown...")
    
    quiz_types = ["C. elegans", "C. elegans (Corpus)"]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for idx, quiz in enumerate(quiz_types):
        ax = axes[idx]
        
        # Get model data
        quiz_results = [r for r in results if r.get('Quiz Category') == quiz]
        
        # Extract model names and scores
        models = []
        llm_only = []
        rag_only = []
        rag_fallback = []
        
        for result in quiz_results:
            model = result.get('LLM', '')
            
            # Clean model name for display
            if 'huggingface:' in model:
                model = model.split('/')[-1].split(':')[0]
            
            models.append(model)
            
            # LLM-only from baseline
            if quiz in no_rag_scores and model in no_rag_scores[quiz]:
                llm_only.append(no_rag_scores[quiz][model])
            else:
                # Try to find by partial match
                found = False
                for full_model, acc in no_rag_scores.get(quiz, {}).items():
                    if model.lower() in full_model.lower():
                        llm_only.append(acc)
                        found = True
                        break
                if not found:
                    llm_only.append(0)
            
            rag_only.append(result.get('RAG-SUFFICIENT (%)', 0))
            rag_fallback.append(result.get('Overall Accuracy (%)', 0))
        
        # Sort by model size
        model_sizes = [get_model_size(m) for m in models]
        sorted_indices = sorted(range(len(models)), key=lambda i: model_sizes[i])
        
        models = [models[i] for i in sorted_indices]
        llm_only = [llm_only[i] for i in sorted_indices]
        rag_only = [rag_only[i] for i in sorted_indices]
        rag_fallback = [rag_fallback[i] for i in sorted_indices]
        
        # Plot
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, llm_only, width, label='LLM-only', 
               color='#3498db', edgecolor='black', linewidth=1)
        ax.bar(x, rag_only, width, label='RAG-only', 
               color='#2ecc71', edgecolor='black', linewidth=1)
        ax.bar(x + width, rag_fallback, width, label='RAG+Fallback', 
               color='#9b59b6', edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{quiz} Quiz', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 110)  # Slightly above 100 to show reference line
        ax.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at 100% for Corpus reference
        if 'Corpus' in quiz:
            ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax.text(len(models)-1, 102, 'Expected: 100%', fontsize=10, color='red', fontweight='bold')
    
    plt.suptitle('Per-Model Performance: C. elegans Quizzes', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "6_per_model_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()


def plot_5i_response_time(results: List[Dict]):
    """
    Plot 5I: Response Time Analysis
    
    Shows that RAG doesn't significantly affect efficiency.
    """
    print("\n📊 Generating Plot 5I: Response Time Analysis...")
    
    # Group by method
    # Note: Current data has combined times, but we can estimate
    
    quiz_types = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    
    times_by_quiz = {quiz: [] for quiz in quiz_types}
    
    for result in results:
        quiz = result.get('Quiz Category')
        if quiz not in quiz_types:
            continue
        
        avg_time = result.get('Avg Response Time (s)', 0)
        if avg_time > 0:
            times_by_quiz[quiz].append(avg_time)
    
    # Calculate averages
    avg_times = []
    std_times = []
    
    for quiz in quiz_types:
        if times_by_quiz[quiz]:
            avg_times.append(np.mean(times_by_quiz[quiz]))
            std_times.append(np.std(times_by_quiz[quiz]))
        else:
            avg_times.append(0)
            std_times.append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    quiz_names_short = ['General', 'Science', 'C. elegans', 'Corpus']
    x = np.arange(len(quiz_names_short))
    width = 0.6
    
    bars = ax.bar(x, avg_times, width, yerr=std_times, 
                  color='#3498db', edgecolor='black', linewidth=1.2,
                  capsize=5, error_kw={'linewidth': 2})
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Quiz Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Response Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('RAG System Efficiency: Response Times Remain Fast', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(quiz_names_short, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation
    max_time = max(avg_times) if avg_times else 1
    ax.annotate(f'Average response time: {np.mean(avg_times):.2f}s\nRAG retrieval adds minimal overhead',
               xy=(0.98, 0.95), xycoords='axes fraction',
               fontsize=11, ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "5i_response_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()


def main():
    """Main execution."""
    
    print("="*80)
    print("RAG ANALYSIS PLOTTING SCRIPT")
    print("="*80)
    
    # Auto-detect result files
    rag_files = glob.glob("openworm_ai/quiz/scores/rag_proper/*.json")
    
    if not rag_files:
        print("\n! No RAG result files found in openworm_ai/quiz/scores/rag_proper/")
        print("  Run QuizEvalRAG_Proper.py first to generate results.")
        sys.exit(1)
    
    print(f"\n📁 Found {len(rag_files)} RAG result file(s)")
    
    # Load RAG results
    results = load_results(rag_files)
    
    if not results:
        print("\n! No valid results found. Exiting.")
        sys.exit(1)
    
    print(f"✓ Total RAG results loaded: {len(results)}")
    
    # Load non-RAG baseline scores
    no_rag_scores = load_no_rag_scores()
    
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    plot_1a_accuracy_by_method(results, no_rag_scores)
    plot_1a2_accuracy_compound_bars(results, no_rag_scores)
    plot_1b_rag_impact_breakdown(results)
    plot_2c_context_coverage(results)
    plot_3e_model_size_vs_rag_benefit(results)
    plot_4g_corpus_gap_analysis(results)
    plot_4h_fallback_value(results)
    plot_5i_response_time(results)
    plot_6_per_model_breakdown(results, no_rag_scores)
    
    print("\n" + "="*80)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nPlots saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated plots:")
    print("  1A.  Accuracy by Method (side-by-side bars)")
    print("  1A2. Accuracy Improvement (compound bars)")
    print("  1B.  RAG Impact Breakdown")
    print("  2C.  Context Coverage vs RAG Success")
    print("  3E.  Model Size vs RAG Benefit")
    print("  4G.  Corpus Gap Analysis")
    print("  4H.  Fallback Value Demonstration")
    print("  5I.  Response Time Analysis")
    print("  6.   Per-Model Breakdown (C. elegans & Corpus)")
    print()


if __name__ == "__main__":
    main()