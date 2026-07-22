import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import clean_text_pipeline

def run_eda(csv_path="GlobalFakeNews_Research2026_v1.csv", output_dir="results/plots"):
    """
    Executes Exploratory Data Analysis and generates plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Starting Exploratory Data Analysis...")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Set aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
    
    # 1. Class Distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='label', data=df, palette='Set2')
    plt.title('Distribution of Real vs Fake News')
    plt.xlabel('Label (0 = Real, 1 = Fake)')
    plt.ylabel('Count')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Text Length Distribution
    df['text_len'] = df['full_text'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='text_len', hue='label', kde=False, bins=10, palette='Set1', multiple='stack')
    plt.title('Article Length Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Sentiment & Clickbait Score Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(ax=axes[0], data=df, x='label', y='sentiment_score', palette='Set3')
    axes[0].set_title('Sentiment Score: Real vs Fake')
    axes[0].set_xlabel('Label (0 = Real, 1 = Fake)')
    axes[0].set_ylabel('Sentiment Score')
    
    sns.boxplot(ax=axes[1], data=df, x='label', y='clickbait_score', palette='Set3')
    axes[1].set_title('Clickbait Score: Real vs Fake')
    axes[1].set_xlabel('Label (0 = Real, 1 = Fake)')
    axes[1].set_ylabel('Clickbait Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scores_comparison.png"), dpi=300)
    plt.close()
    
    # 4. Emotional Word Ratio & Source Trust Score Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.violinplot(ax=axes[0], data=df, x='label', y='emotional_word_ratio', palette='Pastel1', split=True)
    axes[0].set_title('Emotional Word Ratio: Real vs Fake')
    axes[0].set_xlabel('Label (0 = Real, 1 = Fake)')
    axes[0].set_ylabel('Emotional Word Ratio')
    
    sns.boxplot(ax=axes[1], data=df, x='label', y='source_trust_score', palette='Pastel1')
    axes[1].set_title('Source Trust Score: Real vs Fake')
    axes[1].set_xlabel('Label (0 = Real, 1 = Fake)')
    axes[1].set_ylabel('Source Trust Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trust_emotional_comparison.png"), dpi=300)
    plt.close()

    # 5. Category Analysis
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='category', hue='label', palette='Set2')
    plt.title('Category distribution of Real vs Fake News')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_distribution.png"), dpi=300)
    plt.close()

    print(f"EDA successfully completed. Plots saved to '{output_dir}/'")

if __name__ == "__main__":
    run_eda()
