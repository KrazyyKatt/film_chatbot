"""
visualize_embeddings.py — PCA + t-SNE visualization of document embeddings
Run: python visualize_embeddings.py
Saves: embedding_visualization.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rag import get_all_chunks

def plot_embeddings():
    print("Loading chunks and embeddings...")
    texts, vectors = get_all_chunks()

    if len(vectors) < 5:
        print("Not enough vectors to visualize. Build vectorstore first.")
        return

    print(f"Visualizing {len(vectors)} chunks...")

    #  Color by source document 
    keywords = {
        "nolan": "#ff6600",
        "tarantino": "#00b4d8",
        "godfather": "#e63946",
        "inception": "#2a9d8f",
        "interstellar": "#8338ec",
        "back": "#fb8500",
        "pirates": "#06d6a0",
        "breaking": "#ffbe0b",
        "player": "#3a86ff",
        "naked": "#ff006e",
    }

    colors = []
    labels = []
    for text in texts:
        text_lower = text.lower()
        assigned = False
        for kw, color in keywords.items():
            if kw in text_lower:
                colors.append(color)
                labels.append(kw)
                assigned = True
                break
        if not assigned:
            colors.append("#555555")
            labels.append("other")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d0d0d")

    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # PCA 
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(vectors)

    axes[0].scatter(pca_result[:, 0], pca_result[:, 1],
                    c=colors, alpha=0.75, s=40, linewidths=0)
    axes[0].set_title("PCA — Document Embeddings",
                       color="#ff6600", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", color="#888")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", color="#888")

    #  t-SNE 
    perplexity = min(30, len(vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    tsne_result = tsne.fit_transform(vectors)

    axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1],
                    c=colors, alpha=0.75, s=40, linewidths=0)
    axes[1].set_title("t-SNE — Document Embeddings",
                       color="#00b4d8", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Dim 1", color="#888")
    axes[1].set_ylabel("Dim 2", color="#888")

    # Legend
    unique_labels = list(dict.fromkeys(labels))
    color_map = {}
    for text, color, label in zip(texts, colors, labels):
        color_map[label] = color

    patches = [mpatches.Patch(color=color_map[l], label=l.capitalize())
               for l in unique_labels if l in color_map]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               facecolor="#111", edgecolor="#333",
               labelcolor="white", fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("🎬 Film Chatbot — Embedding Space Visualization",
                 color="white", fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig("embedding_visualization.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    print("Saved: embedding_visualization.png")
    plt.show()


if __name__ == "__main__":
    plot_embeddings()
