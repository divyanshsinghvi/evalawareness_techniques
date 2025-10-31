"""
Interpretability and error analysis for evaluation awareness probes.

This module provides:
1. Feature importance analysis (which layers matter most)
2. Error analysis (what kinds of sentences are misclassified)
3. Decision boundary visualization
4. Comparison with LLM judge and keyword baselines
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd


class ProbeAnalyzer:
    """
    Analyzer for probe interpretability and error patterns.
    """

    def __init__(self, probe, output_dir: str = "probe/analysis"):
        """
        Initialize analyzer.

        Args:
            probe: Trained EvalAwarenessProbe instance
            output_dir: Directory to save analysis results
        """
        self.probe = probe
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        save_name: str = "feature_importance.png",
        top_k: int = 20
    ):
        """
        Analyze and visualize feature importance from probe coefficients.

        Args:
            feature_names: Optional list of feature names
            save_name: Filename to save plot
            top_k: Number of top features to display
        """
        # Get coefficients
        coef = self.probe.get_feature_importance()

        if coef.ndim > 1:
            # Multi-class: use mean absolute coefficient across classes
            coef_magnitude = np.abs(coef).mean(axis=0)
        else:
            # Binary: use absolute coefficients
            coef_magnitude = np.abs(coef)

        # Limit top_k to actual number of features
        top_k = min(top_k, len(coef_magnitude))

        # Sort by magnitude
        sorted_indices = np.argsort(coef_magnitude)[::-1][:top_k]
        sorted_coef = coef_magnitude[sorted_indices]

        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(coef_magnitude))]

        sorted_names = [feature_names[i] for i in sorted_indices]

        # Plot
        plt.figure(figsize=(10, max(6, top_k * 0.3)))
        colors = plt.cm.viridis(sorted_coef / sorted_coef.max())
        plt.barh(range(top_k), sorted_coef, color=colors)
        plt.yticks(range(top_k), sorted_names)
        plt.xlabel('Coefficient Magnitude', fontsize=12)
        plt.title(f'Top {top_k} Feature Importance', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3, axis='x')

        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Feature importance plot saved to {save_path}")

        # Return top features
        return [(sorted_names[i], sorted_coef[i]) for i in range(top_k)]

    def analyze_layer_contributions(
        self,
        steering_vectors: Dict[int, np.ndarray],
        save_name: str = "layer_contributions.png"
    ):
        """
        Analyze which layers contribute most to awareness detection.

        Assumes features are projections onto steering vectors,
        ordered by layer index.

        Args:
            steering_vectors: Dictionary of layer_idx -> steering vector
            save_name: Filename to save plot
        """
        coef = self.probe.get_feature_importance()

        if coef.ndim > 1:
            coef_magnitude = np.abs(coef).mean(axis=0)
        else:
            coef_magnitude = np.abs(coef)

        # Assume features are ordered by layer
        layer_indices = sorted(steering_vectors.keys())
        n_layers = len(layer_indices)

        # Get coefficients corresponding to layer features
        if len(coef_magnitude) >= n_layers:
            layer_coefs = coef_magnitude[:n_layers]
        else:
            # Pad with zeros if needed
            layer_coefs = np.zeros(n_layers)
            layer_coefs[:len(coef_magnitude)] = coef_magnitude

        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(layer_indices, layer_coefs, color='steelblue', alpha=0.7)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Coefficient Magnitude', fontsize=12)
        plt.title('Layer Contributions to Awareness Detection', fontsize=14)
        plt.grid(alpha=0.3, axis='y')

        # Highlight top 3 layers
        top_3_idx = np.argsort(layer_coefs)[-3:]
        for idx in top_3_idx:
            if idx < len(layer_indices):
                plt.bar(layer_indices[idx], layer_coefs[idx], color='darkred', alpha=0.8)

        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Layer contributions plot saved to {save_path}")

        # Print top layers
        print("\nTop contributing layers:")
        for idx in top_3_idx[::-1]:
            if idx < len(layer_indices):
                layer = layer_indices[idx]
                print(f"  Layer {layer}: {layer_coefs[idx]:.4f}")

    def analyze_errors(
        self,
        sentences_data: List,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        save_name: str = "error_analysis.txt"
    ):
        """
        Analyze misclassified sentences to understand failure modes.

        Args:
            sentences_data: List of SentenceData objects
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Optional predicted probabilities
            save_name: Filename to save analysis
        """
        # Find errors
        errors = y_true != y_pred
        n_errors = np.sum(errors)

        print(f"\nError Analysis:")
        print(f"  Total errors: {n_errors} / {len(y_true)} ({100 * n_errors / len(y_true):.2f}%)")

        if n_errors == 0:
            print("  No errors to analyze!")
            return

        # Categorize errors
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)

        print(f"  False positives: {np.sum(false_positives)}")
        print(f"  False negatives: {np.sum(false_negatives)}")

        # Analyze by awareness intensity
        if hasattr(sentences_data[0], 'awareness_intensity'):
            error_by_intensity = defaultdict(int)
            total_by_intensity = defaultdict(int)

            for i, sent_data in enumerate(sentences_data):
                intensity = sent_data.awareness_intensity
                total_by_intensity[intensity] += 1
                if errors[i]:
                    error_by_intensity[intensity] += 1

            print("\n  Error rate by awareness intensity:")
            for intensity in sorted(total_by_intensity.keys()):
                error_rate = error_by_intensity[intensity] / total_by_intensity[intensity]
                print(f"    Intensity {intensity}: {error_rate:.2%} ({error_by_intensity[intensity]}/{total_by_intensity[intensity]})")

        # Write detailed error examples
        save_path = os.path.join(self.output_dir, save_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=== ERROR ANALYSIS ===\n\n")
            f.write(f"Total errors: {n_errors} / {len(y_true)} ({100 * n_errors / len(y_true):.2f}%)\n")
            f.write(f"False positives: {np.sum(false_positives)}\n")
            f.write(f"False negatives: {np.sum(false_negatives)}\n\n")

            # Sample false positives
            f.write("=== FALSE POSITIVES (predicted aware, actually not aware) ===\n\n")
            fp_indices = np.where(false_positives)[0]
            for i in fp_indices[:20]:  # Show first 20
                sent_data = sentences_data[i]
                prob = y_proba[i] if y_proba is not None else None

                f.write(f"Example {i}:\n")
                f.write(f"  Text: {sent_data.text}\n")
                f.write(f"  True intensity: {sent_data.awareness_intensity}\n")
                if prob is not None:
                    if prob.ndim == 0:
                        f.write(f"  Predicted prob: {prob:.3f}\n")
                    else:
                        f.write(f"  Predicted prob: {prob[1]:.3f}\n")
                f.write(f"  Source: {sent_data.source_file}\n\n")

            # Sample false negatives
            f.write("\n=== FALSE NEGATIVES (predicted not aware, actually aware) ===\n\n")
            fn_indices = np.where(false_negatives)[0]
            for i in fn_indices[:20]:  # Show first 20
                sent_data = sentences_data[i]
                prob = y_proba[i] if y_proba is not None else None

                f.write(f"Example {i}:\n")
                f.write(f"  Text: {sent_data.text}\n")
                f.write(f"  True intensity: {sent_data.awareness_intensity}\n")
                if prob is not None:
                    if prob.ndim == 0:
                        f.write(f"  Predicted prob: {prob:.3f}\n")
                    else:
                        f.write(f"  Predicted prob: {prob[1]:.3f}\n")
                f.write(f"  Source: {sent_data.source_file}\n\n")

        print(f"\nDetailed error analysis saved to {save_path}")

    def analyze_agreement_with_llm_judge(
        self,
        sentences_data: List,
        y_pred: np.ndarray,
        threshold: int = 5
    ):
        """
        Compare probe predictions with LLM judge scores.

        Args:
            sentences_data: List of SentenceData objects with LLM judge scores
            y_pred: Probe predictions (binary)
            threshold: Intensity threshold for LLM judge (default 5)
        """
        # Extract LLM judge labels
        llm_labels = np.array([
            1 if sent.awareness_intensity >= threshold else 0
            for sent in sentences_data
        ])

        # Compute agreement
        agreement = (y_pred == llm_labels).mean()
        print(f"\nAgreement with LLM judge: {agreement:.2%}")

        # Cohen's kappa
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(llm_labels, y_pred)
        print(f"Cohen's kappa: {kappa:.3f}")

        # Where do they disagree?
        disagree = y_pred != llm_labels
        probe_yes_llm_no = (y_pred == 1) & (llm_labels == 0)
        probe_no_llm_yes = (y_pred == 0) & (llm_labels == 1)

        print(f"  Probe says aware, LLM says not: {np.sum(probe_yes_llm_no)}")
        print(f"  Probe says not aware, LLM says aware: {np.sum(probe_no_llm_yes)}")

        return {
            'agreement': agreement,
            'kappa': kappa,
            'probe_yes_llm_no': np.sum(probe_yes_llm_no),
            'probe_no_llm_yes': np.sum(probe_no_llm_yes)
        }

    def plot_decision_boundary_2d(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: Tuple[int, int] = (0, 1),
        feature_names: Optional[Tuple[str, str]] = None,
        save_name: str = "decision_boundary_2d.png"
    ):
        """
        Visualize decision boundary in 2D feature space.

        Args:
            X: Feature matrix
            y: Labels
            feature_indices: Indices of two features to plot
            feature_names: Optional names for the features
            save_name: Filename to save plot
        """
        # Extract two features
        X_2d = X[:, list(feature_indices)]

        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        # Predict on mesh (need to reconstruct full feature vector)
        # This is approximate - we only vary 2 features
        mesh_features = np.zeros((xx.ravel().shape[0], X.shape[1]))
        mesh_features[:, feature_indices[0]] = xx.ravel()
        mesh_features[:, feature_indices[1]] = yy.ravel()

        # Use mean values for other features
        for i in range(X.shape[1]):
            if i not in feature_indices:
                mesh_features[:, i] = X[:, i].mean()

        # Predict
        Z = self.probe.predict(mesh_features)
        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.colorbar(label='Predicted Class')

        # Scatter plot of actual data
        scatter = plt.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=y,
            cmap='RdYlBu',
            edgecolors='black',
            linewidths=0.5,
            alpha=0.7
        )

        if feature_names:
            plt.xlabel(feature_names[0], fontsize=12)
            plt.ylabel(feature_names[1], fontsize=12)
        else:
            plt.xlabel(f'Feature {feature_indices[0]}', fontsize=12)
            plt.ylabel(f'Feature {feature_indices[1]}', fontsize=12)

        plt.title('Decision Boundary (2D Projection)', fontsize=14)
        plt.colorbar(scatter, label='True Label')

        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Decision boundary plot saved to {save_path}")

    def compare_with_keyword_baseline(
        self,
        sentences_data: List,
        y_true: np.ndarray,
        keywords: List[str] = None
    ):
        """
        Compare probe with simple keyword-based baseline.

        Args:
            sentences_data: List of SentenceData objects
            y_true: True labels
            keywords: Optional list of keywords to use
        """
        if keywords is None:
            # Default awareness keywords
            keywords = [
                'test', 'evaluation', 'scenario', 'designed',
                'assess', 'example', 'hypothetical'
            ]

        # Keyword-based predictions
        keyword_preds = []
        for sent_data in sentences_data:
            text_lower = sent_data.text.lower()
            has_keyword = any(kw in text_lower for kw in keywords)
            keyword_preds.append(1 if has_keyword else 0)

        keyword_preds = np.array(keyword_preds)

        # Metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        acc = accuracy_score(y_true, keyword_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, keyword_preds, average='binary'
        )

        print(f"\nKeyword Baseline Performance:")
        print(f"  Keywords: {keywords}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def generate_analysis_report(
        self,
        sentences_data: List,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        steering_vectors: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Generate comprehensive analysis report.

        Args:
            sentences_data: List of SentenceData objects
            X: Feature matrix
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Optional predicted probabilities
            steering_vectors: Optional steering vectors
            feature_names: Optional feature names
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE PROBE ANALYSIS")
        print("="*60)

        # Feature importance
        print("\n### Feature Importance ###")
        top_features = self.analyze_feature_importance(feature_names, top_k=15)

        # Layer contributions (if steering vectors provided)
        if steering_vectors:
            print("\n### Layer Contributions ###")
            self.analyze_layer_contributions(steering_vectors)

        # Error analysis
        print("\n### Error Analysis ###")
        self.analyze_errors(sentences_data, y_true, y_pred, y_proba)

        # Agreement with LLM judge
        print("\n### Agreement with LLM Judge ###")
        llm_agreement = self.analyze_agreement_with_llm_judge(sentences_data, y_pred)

        # Keyword baseline
        print("\n### Keyword Baseline Comparison ###")
        keyword_results = self.compare_with_keyword_baseline(sentences_data, y_true)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    print("=== Evaluation Awareness Probe - Analysis ===\n")

    print("Note: This is a demonstration. Use with real trained probe and data.\n")

    # This would normally use real probe and data
    print("To use:")
    print("1. Train a probe with train_probe.py")
    print("2. Load the probe: probe = EvalAwarenessProbe.load('path/to/probe.pkl')")
    print("3. Create analyzer: analyzer = ProbeAnalyzer(probe)")
    print("4. Run analyses as shown above")

    print("\n✓ Analysis module ready!")
