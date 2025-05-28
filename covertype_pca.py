import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings
import os
import concurrent.futures
warnings.filterwarnings('ignore')

# Configuration parameters
FILE_PATH = './input/covertype.csv'

ROW_SIZES = [1_000, 5_000, 25_000, 100_000, 500_000, 1_000_000]
FEATURE_COUNTS = [5, 10, 20, 30, 40, 54]  # 54 is the total number of features in the Covertype dataset
PCA_COMPONENTS = [3, 5, 10, 15, 20, 25]  # Number of PCA components to test
NEIGHBORS_COUNTS = [5]  # Common choices for K in KNN

FEATURE_IMPORTANCE_METHODS = ['random_forest', 'statistical']
FEATURE_IMPORTANCE_METHOD = FEATURE_IMPORTANCE_METHODS[1]


def load_and_prepare_data(data_url='https://archive.ics.uci.edu/static/public/31/data.csv'):
    """
    Load the Covertype dataset and prepare it for analysis
    """
    X, y = None, None
    df = None
    if not os.path.exists(FILE_PATH):
        # fetch dataset
        covertype = fetch_ucirepo(id=31)

        # data (as pandas dataframes)
        X = covertype.data.features
        y = covertype.data.targets
        df = pd.concat([X, y], axis=1)

        # metadata
        print(covertype.metadata)

        # variable information
        print(covertype.variables)
        df.to_csv(FILE_PATH, index=False)
    else:
        # Load from local CSV if available
        df = pd.read_csv(FILE_PATH)
        X = df.drop('Cover_Type', axis=1)
        y = df['Cover_Type']

    return X, y


def get_feature_importance(X, y, method='random_forest'):
    """
    Get feature importance using Random Forest or statistical methods
    """
    print(f"Calculating feature importance using {method}...")

    if method == 'random_forest':
        # Use Random Forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance_scores = rf.feature_importances_

    elif method == 'statistical':
        # Use statistical test (f_classif)
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        importance_scores = selector.scores_

    # Get feature names and their importance scores
    feature_names = X.columns.tolist()
    feature_importance = list(zip(feature_names, importance_scores))

    # Sort by importance (descending)
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"Top 10 most important features:")
    for i, (feature, score) in enumerate(feature_importance[:10]):
        print(f"{i+1}. {feature}: {score:.4f}")

    return feature_importance


def select_top_features(X, feature_importance, n_features):
    """
    Select top n_features based on importance scores
    """
    top_features = [feature for feature, _ in feature_importance[:n_features]]
    return X[top_features]


def apply_pca_reduction(X_train, X_test, n_components, explained_variance_threshold=0.95):
    """
    Apply PCA dimensionality reduction to the data
    """
    # Initialize PCA
    if n_components == 'auto':
        # Use enough components to explain specified variance
        pca = PCA(n_components=explained_variance_threshold, random_state=42)
    else:
        pca = PCA(n_components=n_components, random_state=42)
    
    # Fit PCA on training data and transform both sets
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Convert back to DataFrame with meaningful column names
    n_comp = X_train_pca.shape[1]
    columns = [f'PC{i+1}' for i in range(n_comp)]
    X_train_pca = pd.DataFrame(X_train_pca, columns=columns, index=X_train.index)
    X_test_pca = pd.DataFrame(X_test_pca, columns=columns, index=X_test.index)
    
    explained_variance_ratio = pca.explained_variance_ratio_.sum()
    
    return X_train_pca, X_test_pca, explained_variance_ratio, pca


def run_knn_brute(X_train, X_test, y_train, y_test, n_neighbors):
    """Wrapper function for KNN brute force evaluation"""
    try:
        return evaluate_knn_performance(
            X_train, X_test, y_train, y_test,
            n_neighbors, algorithm='brute'
        )
    except Exception as e:
        print(f"    Error in KNN brute: {e}")
        return {key: np.nan for key in ['accuracy', 'precision', 'recall', 'f1_score', 'fit_time', 'predict_time', 'total_time']}


def run_knn_kdtree(X_train, X_test, y_train, y_test, n_neighbors):
    """Wrapper function for KNN KDTree evaluation"""
    try:
        return evaluate_knn_performance(
            X_train, X_test, y_train, y_test,
            n_neighbors, algorithm='kd_tree'
        )
    except Exception as e:
        print(f"    Error in KDTree: {e}")
        return {key: np.nan for key in ['accuracy', 'precision', 'recall', 'f1_score', 'fit_time', 'predict_time', 'total_time']}


def evaluate_knn_performance(X_train, X_test, y_train, y_test, n_neighbors, algorithm='auto'):
    """
    Evaluate KNN performance with specified parameters
    """
    # Initialize KNN classifier
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        n_jobs=-1
    )

    # Train and predict
    start_time = time.time()
    knn.fit(X_train, y_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    y_pred = knn.predict(X_test)
    predict_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fit_time': fit_time,
        'predict_time': predict_time,
        'total_time': fit_time + predict_time
    }


def run_pca_comparison(X, y, feature_importance,
                      row_sizes=[1000, 5000, 10000, 25000],
                      feature_counts=[10, 20, 30, 40, 54],
                      pca_components=[5, 10, 15, 20],
                      neighbor_counts=[3, 5, 7, 9, 11, 15],
                      output_path='./output/covertype/results/pca_knn_comparison_partial.csv'):
    """
    Run comprehensive comparison between original features and PCA-reduced features.
    Tests both KNN brute force and KDTree on original and PCA-transformed data.
    """

    columns = [
        'n_rows', 'n_features', 'pca_components', 'n_neighbors', 'explained_variance',
        # Original data results
        'orig_knn_accuracy', 'orig_knn_precision', 'orig_knn_recall', 'orig_knn_f1_score',
        'orig_knn_fit_time', 'orig_knn_predict_time', 'orig_knn_total_time',
        'orig_kdtree_accuracy', 'orig_kdtree_precision', 'orig_kdtree_recall', 'orig_kdtree_f1_score',
        'orig_kdtree_fit_time', 'orig_kdtree_predict_time', 'orig_kdtree_total_time',
        # PCA data results
        'pca_knn_accuracy', 'pca_knn_precision', 'pca_knn_recall', 'pca_knn_f1_score',
        'pca_knn_fit_time', 'pca_knn_predict_time', 'pca_knn_total_time',
        'pca_kdtree_accuracy', 'pca_kdtree_precision', 'pca_kdtree_recall', 'pca_kdtree_f1_score',
        'pca_kdtree_fit_time', 'pca_kdtree_predict_time', 'pca_kdtree_total_time',
        # Comparison metrics
        'pca_knn_accuracy_diff', 'pca_kdtree_accuracy_diff',
        'pca_knn_time_improvement', 'pca_kdtree_time_improvement'
    ]

    start_at = None
    if os.path.exists(output_path):
        # Load existing results if file exists
        results = pd.read_csv(output_path)
        print(f"Loaded existing results from {output_path}.")
        if not results.empty:
            last_entry = results.iloc[-1]
            start_at = (
                last_entry['n_rows'],
                last_entry['n_features'],
                last_entry['pca_components'],
                last_entry['n_neighbors']
            )
    else:
        # Initialize empty DataFrame for results
        print(f"Creating new results DataFrame at {output_path}.")
        results = pd.DataFrame(columns=columns)

    total_combinations = len(row_sizes) * len(feature_counts) * len(pca_components) * len(neighbor_counts)
    current_combination = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Starting PCA comparison with {total_combinations} combinations...")
    save_every = 1

    for n_rows in row_sizes:
        if start_at and n_rows < start_at[0]:
            continue
        print(f"\n--- Testing with {n_rows} rows ---")

        # Sample data if needed
        if n_rows < len(X):
            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=n_rows, random_state=42, stratify=y
            )
        else:
            print(f"Using full dataset: {len(X)} rows")
            X_sample, y_sample = X, y

        for n_features in feature_counts:
            if start_at and (n_rows, n_features) < start_at[:2]:
                continue
            if n_features > len(X_sample.columns):
                continue

            print(f"  Testing with {n_features} features...")

            # Select top features
            X_selected = select_top_features(X_sample, feature_importance, n_features)

            # Scale features for better KNN performance
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_selected),
                columns=X_selected.columns
            )

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_sample, test_size=0.2, random_state=42, stratify=y_sample
            )

            for pca_comp in pca_components:
                if start_at and (n_rows, n_features, pca_comp) < start_at[:3]:
                    continue
                if pca_comp >= n_features:  # Skip if PCA components >= original features
                    continue

                print(f"    Testing PCA with {pca_comp} components...")

                # Apply PCA
                try:
                    X_train_pca, X_test_pca, explained_var, pca_model = apply_pca_reduction(
                        X_train, X_test, pca_comp
                    )
                except Exception as e:
                    print(f"    Error in PCA: {e}")
                    continue

                for n_neighbors in neighbor_counts:
                    if (start_at and (n_rows, n_features, pca_comp, n_neighbors) <= start_at or 
                        len(results[(results['n_rows'] == n_rows) & 
                                  (results['n_features'] == n_features) & 
                                  (results['pca_components'] == pca_comp) & 
                                  (results['n_neighbors'] == n_neighbors)]) > 0):
                        continue

                    current_combination += 1
                    print(f"      Progress: {current_combination}/{total_combinations} - Testing with {n_neighbors} neighbors")

                    # Test original data with parallel execution
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        # Submit original data tasks
                        future_orig_brute = executor.submit(run_knn_brute, X_train, X_test, y_train, y_test, n_neighbors)
                        future_orig_kdtree = executor.submit(run_knn_kdtree, X_train, X_test, y_train, y_test, n_neighbors)
                        
                        # Get original data results
                        orig_knn_results = future_orig_brute.result()
                        orig_kdtree_results = future_orig_kdtree.result()

                    # Test PCA data with parallel execution
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        # Submit PCA data tasks
                        future_pca_brute = executor.submit(run_knn_brute, X_train_pca, X_test_pca, y_train, y_test, n_neighbors)
                        future_pca_kdtree = executor.submit(run_knn_kdtree, X_train_pca, X_test_pca, y_train, y_test, n_neighbors)
                        
                        # Get PCA data results
                        pca_knn_results = future_pca_brute.result()
                        pca_kdtree_results = future_pca_kdtree.result()

                    # Calculate comparison metrics
                    pca_knn_acc_diff = pca_knn_results['accuracy'] - orig_knn_results['accuracy']
                    pca_kdtree_acc_diff = pca_kdtree_results['accuracy'] - orig_kdtree_results['accuracy']
                    pca_knn_time_improvement = (orig_knn_results['total_time'] - pca_knn_results['total_time']) / orig_knn_results['total_time'] if orig_knn_results['total_time'] > 0 else 0
                    pca_kdtree_time_improvement = (orig_kdtree_results['total_time'] - pca_kdtree_results['total_time']) / orig_kdtree_results['total_time'] if orig_kdtree_results['total_time'] > 0 else 0

                    # Store results
                    result = {
                        'n_rows': n_rows,
                        'n_features': n_features,
                        'pca_components': pca_comp,
                        'n_neighbors': n_neighbors,
                        'explained_variance': explained_var,
                        # Original data results
                        'orig_knn_accuracy': orig_knn_results['accuracy'],
                        'orig_knn_precision': orig_knn_results['precision'],
                        'orig_knn_recall': orig_knn_results['recall'],
                        'orig_knn_f1_score': orig_knn_results['f1_score'],
                        'orig_knn_fit_time': orig_knn_results['fit_time'],
                        'orig_knn_predict_time': orig_knn_results['predict_time'],
                        'orig_knn_total_time': orig_knn_results['total_time'],
                        'orig_kdtree_accuracy': orig_kdtree_results['accuracy'],
                        'orig_kdtree_precision': orig_kdtree_results['precision'],
                        'orig_kdtree_recall': orig_kdtree_results['recall'],
                        'orig_kdtree_f1_score': orig_kdtree_results['f1_score'],
                        'orig_kdtree_fit_time': orig_kdtree_results['fit_time'],
                        'orig_kdtree_predict_time': orig_kdtree_results['predict_time'],
                        'orig_kdtree_total_time': orig_kdtree_results['total_time'],
                        # PCA data results
                        'pca_knn_accuracy': pca_knn_results['accuracy'],
                        'pca_knn_precision': pca_knn_results['precision'],
                        'pca_knn_recall': pca_knn_results['recall'],
                        'pca_knn_f1_score': pca_knn_results['f1_score'],
                        'pca_knn_fit_time': pca_knn_results['fit_time'],
                        'pca_knn_predict_time': pca_knn_results['predict_time'],
                        'pca_knn_total_time': pca_knn_results['total_time'],
                        'pca_kdtree_accuracy': pca_kdtree_results['accuracy'],
                        'pca_kdtree_precision': pca_kdtree_results['precision'],
                        'pca_kdtree_recall': pca_kdtree_results['recall'],
                        'pca_kdtree_f1_score': pca_kdtree_results['f1_score'],
                        'pca_kdtree_fit_time': pca_kdtree_results['fit_time'],
                        'pca_kdtree_predict_time': pca_kdtree_results['predict_time'],
                        'pca_kdtree_total_time': pca_kdtree_results['total_time'],
                        # Comparison metrics
                        'pca_knn_accuracy_diff': pca_knn_acc_diff,
                        'pca_kdtree_accuracy_diff': pca_kdtree_acc_diff,
                        'pca_knn_time_improvement': pca_knn_time_improvement,
                        'pca_kdtree_time_improvement': pca_kdtree_time_improvement
                    }

                    # Append to DataFrame
                    results.loc[len(results)] = result

                    # Periodically save to CSV
                    if current_combination % save_every == 0 or current_combination == total_combinations:
                        results.to_csv(output_path, index=False)
                        print(f"      Partial results saved to {output_path}")

    # Final save
    results.to_csv(output_path, index=False)
    print(f"\nAll results saved to {output_path}")
    return results


def analyze_pca_results(results_df):
    """
    Analyze and summarize the PCA comparison results
    """
    print("\n" + "="*80)
    print("PCA DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*80)

    # Overall PCA impact
    print("\n1. OVERALL PCA IMPACT:")
    print(f"   Average Original KNN Accuracy: {results_df['orig_knn_accuracy'].mean():.4f}")
    print(f"   Average PCA KNN Accuracy: {results_df['pca_knn_accuracy'].mean():.4f}")
    print(f"   Average KNN Accuracy Change: {results_df['pca_knn_accuracy_diff'].mean():.4f}")
    print(f"   Average Original KDTree Accuracy: {results_df['orig_kdtree_accuracy'].mean():.4f}")
    print(f"   Average PCA KDTree Accuracy: {results_df['pca_kdtree_accuracy'].mean():.4f}")
    print(f"   Average KDTree Accuracy Change: {results_df['pca_kdtree_accuracy_diff'].mean():.4f}")

    # Time improvements
    print("\n2. TIME PERFORMANCE:")
    print(f"   Average KNN Time Improvement: {results_df['pca_knn_time_improvement'].mean()*100:.1f}%")
    print(f"   Average KDTree Time Improvement: {results_df['pca_kdtree_time_improvement'].mean()*100:.1f}%")
    
    positive_knn_improvements = results_df[results_df['pca_knn_time_improvement'] > 0]
    positive_kdtree_improvements = results_df[results_df['pca_kdtree_time_improvement'] > 0]
    
    print(f"   Cases where PCA improved KNN speed: {len(positive_knn_improvements)}/{len(results_df)} ({len(positive_knn_improvements)/len(results_df)*100:.1f}%)")
    print(f"   Cases where PCA improved KDTree speed: {len(positive_kdtree_improvements)}/{len(results_df)} ({len(positive_kdtree_improvements)/len(results_df)*100:.1f}%)")

    # Variance explained analysis
    print("\n3. VARIANCE EXPLAINED:")
    variance_analysis = results_df.groupby('pca_components')['explained_variance'].mean()
    print("   Average explained variance by PCA components:")
    for comp, var in variance_analysis.items():
        print(f"   {comp} components: {var:.3f} ({var*100:.1f}%)")

    # Best configurations
    print("\n4. BEST CONFIGURATIONS:")
    
    # Best PCA configuration for accuracy
    best_pca_knn = results_df.loc[results_df['pca_knn_accuracy'].idxmax()]
    best_pca_kdtree = results_df.loc[results_df['pca_kdtree_accuracy'].idxmax()]
    
    print(f"   Best PCA KNN: {best_pca_knn['n_rows']} rows, {best_pca_knn['n_features']} features, {best_pca_knn['pca_components']} PCA, {best_pca_knn['n_neighbors']} neighbors")
    print(f"   Accuracy: {best_pca_knn['pca_knn_accuracy']:.4f} (vs {best_pca_knn['orig_knn_accuracy']:.4f} original)")
    
    print(f"   Best PCA KDTree: {best_pca_kdtree['n_rows']} rows, {best_pca_kdtree['n_features']} features, {best_pca_kdtree['pca_components']} PCA, {best_pca_kdtree['n_neighbors']} neighbors")
    print(f"   Accuracy: {best_pca_kdtree['pca_kdtree_accuracy']:.4f} (vs {best_pca_kdtree['orig_kdtree_accuracy']:.4f} original)")

    # Performance by PCA components
    print("\n5. PERFORMANCE BY PCA COMPONENTS:")
    pca_analysis = results_df.groupby('pca_components').agg({
        'pca_knn_accuracy': 'mean',
        'pca_kdtree_accuracy': 'mean',
        'pca_knn_time_improvement': 'mean',
        'pca_kdtree_time_improvement': 'mean',
        'explained_variance': 'mean'
    }).round(4)
    print(pca_analysis)

    return results_df


def create_pca_visualizations(results_df, output_dir='./output/covertype/plots/'):
    """
    Create comprehensive visualizations for PCA analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(16, 12))

    # 1. Accuracy comparison: Original vs PCA
    plt.subplot(3, 3, 1)
    plt.scatter(results_df['orig_knn_accuracy'], results_df['pca_knn_accuracy'], alpha=0.6, label='KNN')
    plt.scatter(results_df['orig_kdtree_accuracy'], results_df['pca_kdtree_accuracy'], alpha=0.6, label='KDTree')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    plt.xlabel('Original Accuracy')
    plt.ylabel('PCA Accuracy')
    plt.title('Original vs PCA Accuracy')
    plt.legend()

    # 2. Time improvement distribution
    plt.subplot(3, 3, 2)
    plt.hist(results_df['pca_knn_time_improvement'] * 100, bins=20, alpha=0.6, label='KNN')
    plt.hist(results_df['pca_kdtree_time_improvement'] * 100, bins=20, alpha=0.6, label='KDTree')
    plt.xlabel('Time Improvement (%)')
    plt.ylabel('Frequency')
    plt.title('PCA Time Improvement Distribution')
    plt.legend()

    # 3. Accuracy vs PCA components
    plt.subplot(3, 3, 3)
    acc_by_pca = results_df.groupby('pca_components')[['pca_knn_accuracy', 'pca_kdtree_accuracy']].mean()
    acc_by_pca.plot(kind='line', marker='o', ax=plt.gca())
    plt.xlabel('PCA Components')
    plt.ylabel('Mean Accuracy')
    plt.title('Accuracy vs PCA Components')
    plt.legend()

    # 4. Explained variance vs accuracy
    plt.subplot(3, 3, 4)
    plt.scatter(results_df['explained_variance'], results_df['pca_knn_accuracy'], alpha=0.6, label='KNN')
    plt.scatter(results_df['explained_variance'], results_df['pca_kdtree_accuracy'], alpha=0.6, label='KDTree')
    plt.xlabel('Explained Variance Ratio')
    plt.ylabel('PCA Accuracy')
    plt.title('Explained Variance vs Accuracy')
    plt.legend()

    # 5. Time improvement vs PCA components
    plt.subplot(3, 3, 5)
    time_by_pca = results_df.groupby('pca_components')[['pca_knn_time_improvement', 'pca_kdtree_time_improvement']].mean() * 100
    time_by_pca.plot(kind='bar', ax=plt.gca())
    plt.xlabel('PCA Components')
    plt.ylabel('Mean Time Improvement (%)')
    plt.title('Time Improvement vs PCA Components')
    plt.legend()
    plt.xticks(rotation=0)

    # 6. Accuracy difference heatmap by features and PCA components
    plt.subplot(3, 3, 6)
    pivot_knn = results_df.groupby(['n_features', 'pca_components'])['pca_knn_accuracy_diff'].mean().unstack()
    plt.imshow(pivot_knn.values, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='KNN Accuracy Difference')
    plt.xlabel('PCA Components')
    plt.ylabel('Original Features')
    plt.title('KNN Accuracy Difference Heatmap')
    plt.xticks(range(len(pivot_knn.columns)), pivot_knn.columns)
    plt.yticks(range(len(pivot_knn.index)), pivot_knn.index)

    # 7. Time vs Accuracy trade-off
    plt.subplot(3, 3, 7)
    plt.scatter(results_df['pca_knn_time_improvement'] * 100, results_df['pca_knn_accuracy_diff'] * 100, 
                alpha=0.6, label='KNN')
    plt.scatter(results_df['pca_kdtree_time_improvement'] * 100, results_df['pca_kdtree_accuracy_diff'] * 100, 
                alpha=0.6, label='KDTree')
    plt.xlabel('Time Improvement (%)')
    plt.ylabel('Accuracy Change (%)')
    plt.title('Time vs Accuracy Trade-off')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.legend()

    # 8. Performance by dataset size
    plt.subplot(3, 3, 8)
    size_analysis = results_df.groupby('n_rows')[['pca_knn_accuracy', 'pca_kdtree_accuracy', 
                                                  'orig_knn_accuracy', 'orig_kdtree_accuracy']].mean()
    size_analysis.plot(kind='line', marker='o', ax=plt.gca())
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Mean Accuracy')
    plt.title('Accuracy vs Dataset Size')
    plt.legend()
    plt.xscale('log')

    # 9. Dimensionality reduction effectiveness
    plt.subplot(3, 3, 9)
    results_df['reduction_ratio'] = results_df['pca_components'] / results_df['n_features']
    plt.scatter(results_df['reduction_ratio'], results_df['pca_knn_accuracy_diff'], alpha=0.6, label='KNN')
    plt.scatter(results_df['reduction_ratio'], results_df['pca_kdtree_accuracy_diff'], alpha=0.6, label='KDTree')
    plt.xlabel('Dimensionality Reduction Ratio')
    plt.ylabel('Accuracy Change')
    plt.title('Reduction Ratio vs Accuracy Change')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}pca_knn_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create additional focused plots
    plt.figure(figsize=(12, 8))

    # Curse of dimensionality analysis
    plt.subplot(2, 2, 1)
    curse_analysis = results_df.groupby('n_features').agg({
        'orig_knn_accuracy': 'mean',
        'pca_knn_accuracy': 'mean',
        'orig_kdtree_accuracy': 'mean', 
        'pca_kdtree_accuracy': 'mean'
    })
    curse_analysis.plot(kind='line', marker='o', ax=plt.gca())
    plt.xlabel('Number of Original Features')
    plt.ylabel('Mean Accuracy')
    plt.title('Curse of Dimensionality: Original vs PCA')
    plt.legend()

    # PCA effectiveness by explained variance
    plt.subplot(2, 2, 2)
    var_bins = pd.cut(results_df['explained_variance'], bins=5)
    var_analysis = results_df.groupby(var_bins)[['pca_knn_accuracy_diff', 'pca_kdtree_accuracy_diff']].mean()
    var_analysis.plot(kind='bar', ax=plt.gca())
    plt.xlabel('Explained Variance Range')
    plt.ylabel('Mean Accuracy Difference')
    plt.title('PCA Effectiveness by Variance Explained')
    plt.xticks(rotation=45)
    plt.legend()

    # Speed improvement analysis
    plt.subplot(2, 2, 3)
    speed_analysis = results_df.groupby('pca_components')[['pca_knn_time_improvement', 'pca_kdtree_time_improvement']].mean() * 100
    speed_analysis.plot(kind='line', marker='o', ax=plt.gca())
    plt.xlabel('PCA Components')
    plt.ylabel('Mean Time Improvement (%)')
    plt.title('Speed Improvement by PCA Components')
    plt.legend()

    # Best trade-off identification
    plt.subplot(2, 2, 4)
    # Calculate a combined score: accuracy preservation + time improvement
    results_df['combined_score'] = (1 + results_df['pca_knn_accuracy_diff']) * (1 + results_df['pca_knn_time_improvement'])
    best_configs = results_df.nlargest(20, 'combined_score')
    plt.scatter(best_configs['pca_knn_time_improvement'] * 100, best_configs['pca_knn_accuracy_diff'] * 100, 
                c=best_configs['pca_components'], cmap='viridis', s=60)
    plt.colorbar(label='PCA Components')
    plt.xlabel('Time Improvement (%)')
    plt.ylabel('Accuracy Change (%)')
    plt.title('Top 20 PCA Configurations')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}pca_curse_dimensionality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nVisualizations saved to {output_dir}")


def main():
    """
    Main execution function for PCA analysis
    """
    print("PCA + KNN Performance Analysis for Curse of Dimensionality")
    print("="*70)

    # Load and prepare data
    X, y = load_and_prepare_data()

    # Get feature importance
    feature_importance = get_feature_importance(X, y, method=FEATURE_IMPORTANCE_METHOD)

    # Run PCA comparison
    results_df = run_pca_comparison(
        X, y, feature_importance,
        row_sizes=ROW_SIZES,
        feature_counts=FEATURE_COUNTS,
        pca_components=PCA_COMPONENTS,
        neighbor_counts=NEIGHBORS_COUNTS
    )

    # Analyze results
    final_results = analyze_pca_results(results_df)

    # Create visualizations
    create_pca_visualizations(final_results)

    # Save results
    os.makedirs('./output/covertype/results', exist_ok=True)
    output_filename = './output/covertype/results/pca_knn_comparison_results.csv'
    final_results.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")

    # Summary insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Find best PCA configurations
    positive_improvements = final_results[
        (final_results['pca_knn_accuracy_diff'] >= -0.01) &  # Allow small accuracy loss
        (final_results['pca_knn_time_improvement'] > 0.1)     # Require significant time improvement
    ]
    
    if len(positive_improvements) > 0:
        print(f"\n✓ Found {len(positive_improvements)} configurations where PCA provides good trade-offs")
        best_trade_off = positive_improvements.loc[positive_improvements['pca_knn_time_improvement'].idxmax()]
        print(f"  Best trade-off: {best_trade_off['n_features']}→{best_trade_off['pca_components']} features")
        print(f"  Time improvement: {best_trade_off['pca_knn_time_improvement']*100:.1f}%")
        print(f"  Accuracy change: {best_trade_off['pca_knn_accuracy_diff']*100:.2f}%")
    
    # Curse of dimensionality analysis
    high_dim = final_results[final_results['n_features'] >= 30]
    low_dim = final_results[final_results['n_features'] <= 15]
    
    if len(high_dim) > 0 and len(low_dim) > 0:
        high_dim_improvement = high_dim['pca_knn_time_improvement'].mean()
        low_dim_improvement = low_dim['pca_knn_time_improvement'].mean()
        
        print(f"\n📈 Curse of Dimensionality Impact:")
        print(f"  High-dim datasets (≥30 features): {high_dim_improvement*100:.1f}% avg time improvement")
        print(f"  Low-dim datasets (≤15 features): {low_dim_improvement*100:.1f}% avg time improvement")
        
        if high_dim_improvement > low_dim_improvement * 1.5:
            print(f"  → PCA is {high_dim_improvement/low_dim_improvement:.1f}x more effective for high-dimensional data")

    return final_results


if __name__ == "__main__":
    # Execute the PCA comparison
    results = main()

    # Display sample results
    print("\nSample results (first 5 rows):")
    display_cols = ['n_rows', 'n_features', 'pca_components', 'n_neighbors', 
                   'orig_knn_accuracy', 'pca_knn_accuracy', 'pca_knn_accuracy_diff',
                   'pca_knn_time_improvement', 'explained_variance']
    print(results[display_cols].head().round(4).to_string(index=False))

    # Show best PCA configurations
    print("\nTop 5 PCA configurations by combined performance:")
    results['combined_score'] = (1 + results['pca_knn_accuracy_diff']) * (1 + results['pca_knn_time_improvement'])
    top_configs = results.nlargest(5, 'combined_score')[display_cols]
    print(top_configs.round(4).to_string(index=False))

    print("\n🎯 Analysis complete! Check the generated plots and CSV files for detailed results.")