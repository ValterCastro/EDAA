import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings
import os
warnings.filterwarnings('ignore')

FILE_PATH = './input/covertype.csv'

ROW_SIZES = [1_000, 5_000, 25_000, 100_000, 500_000, 1_000_000]
FEATURE_COUNTS = [5, 10, 20, 30, 40, 54]  # 54 is the total number of features in the Covertype dataset
NEIGHBORS_COUNTS = [3, 5, 7, 9, 11, 15]  # Common choices for K in KNN

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

    #print(f"Dataset shape: {X.shape}")
    #print(f"Target classes: {sorted(y.unique())}")

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

def run_comprehensive_comparison(X, y, feature_importance,
                               row_sizes=[1000, 5000, 10000, 25000],
                               feature_counts=[5, 10, 20, 30, 40, 54],
                               neighbor_counts=[3, 5, 7, 9, 11, 15],
                               output_path='./output/covertype/results/knn_kdtree_comparison_partial.csv'):
    """
    Run comprehensive comparison between KNN and KDTree algorithms.
    Results are stored in a DataFrame and periodically saved to CSV.
    """

    columns = [
        'n_rows', 'n_features', 'n_neighbors',
        'knn_accuracy', 'knn_precision', 'knn_recall', 'knn_f1_score',
        'knn_fit_time', 'knn_predict_time', 'knn_total_time',
        'kdtree_accuracy', 'kdtree_precision', 'kdtree_recall', 'kdtree_f1_score',
        'kdtree_fit_time', 'kdtree_predict_time', 'kdtree_total_time',
        'accuracy_difference', 'time_difference', 'kdtree_faster'
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
            last_entry['n_neighbors']
            )
    else:
        # Initialize empty DataFrame for results
        print(f"Creating new results DataFrame at {output_path}.")
        results = pd.DataFrame(columns=columns)
    total_combinations = len(row_sizes) * len(feature_counts) * len(neighbor_counts)
    current_combination = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Starting comprehensive comparison with {total_combinations} combinations...")
    save_every = total_combinations // 10 if total_combinations > 10 else 1
    for n_rows in row_sizes:
        save_every = (total_combinations - current_combination) // 10 if (total_combinations - current_combination) > 10 else 1
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

            for n_neighbors in neighbor_counts:
                if (start_at and (n_rows, n_features, n_neighbors) < start_at or 
                    results[(results['n_rows'] == n_rows) & (results['n_features'] == n_features) & (results['n_neighbors'] == n_neighbors)].shape[0] > 0):
                    continue
                current_combination += 1
                print(f"    Progress: {current_combination}/{total_combinations} - Testing with {n_neighbors} neighbors")
                # Test KNN with brute force (ball_tree for comparison)
                try:
                    knn_results = evaluate_knn_performance(
                        X_train, X_test, y_train, y_test,
                        n_neighbors, algorithm='brute'
                    )
                except Exception as e:
                    print(f"    Error in KNN brute: {e}")
                    knn_results = {key: np.nan for key in ['accuracy', 'precision', 'recall', 'f1_score', 'fit_time', 'predict_time', 'total_time']}
                # Test KNN with KDTree
                try:
                    kdtree_results = evaluate_knn_performance(
                        X_train, X_test, y_train, y_test,
                        n_neighbors, algorithm='kd_tree'
                    )
                except Exception as e:
                    print(f"    Error in KDTree: {e}")
                    kdtree_results = {key: np.nan for key in ['accuracy', 'precision', 'recall', 'f1_score', 'fit_time', 'predict_time', 'total_time']}

                # Store results as a DataFrame row
                result = {
                    'n_rows': n_rows,
                    'n_features': n_features,
                    'n_neighbors': n_neighbors,
                    'knn_accuracy': knn_results['accuracy'],
                    'knn_precision': knn_results['precision'],
                    'knn_recall': knn_results['recall'],
                    'knn_f1_score': knn_results['f1_score'],
                    'knn_fit_time': knn_results['fit_time'],
                    'knn_predict_time': knn_results['predict_time'],
                    'knn_total_time': knn_results['total_time'],
                    'kdtree_accuracy': kdtree_results['accuracy'],
                    'kdtree_precision': kdtree_results['precision'],
                    'kdtree_recall': kdtree_results['recall'],
                    'kdtree_f1_score': kdtree_results['f1_score'],
                    'kdtree_fit_time': kdtree_results['fit_time'],
                    'kdtree_predict_time': kdtree_results['predict_time'],
                    'kdtree_total_time': kdtree_results['total_time'],
                    'accuracy_difference': kdtree_results['accuracy'] - knn_results['accuracy'],
                    'time_difference': knn_results['total_time'] - kdtree_results['total_time'],
                    'kdtree_faster': kdtree_results['total_time'] < knn_results['total_time']
                }

                # Append to DataFrame
                results.loc[len(results)] = result

                # Periodically save to CSV
                if current_combination % save_every == 0 or current_combination == total_combinations:
                    results.to_csv(output_path, index=False)
                    print(f"    Partial results saved to {output_path}")

    # Final save
    results.to_csv(output_path, index=False)
    print(f"\nAll results saved to {output_path}")
    return results

def analyze_results(results_df):
    """
    Analyze and summarize the comparison results
    """
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    # Overall performance
    print("\n1. OVERALL PERFORMANCE:")
    print(f"   Average KNN Accuracy: {results_df['knn_accuracy'].mean():.4f}")
    print(f"   Average KDTree Accuracy: {results_df['kdtree_accuracy'].mean():.4f}")
    print(f"   Average Accuracy Difference (KDTree - KNN): {results_df['accuracy_difference'].mean():.4f}")

    # Timing analysis
    print("\n2. TIMING ANALYSIS:")
    print(f"   Average KNN Total Time: {results_df['knn_total_time'].mean():.4f}s")
    print(f"   Average KDTree Total Time: {results_df['kdtree_total_time'].mean():.4f}s")
    print(f"   KDTree faster in {results_df['kdtree_faster'].sum()}/{len(results_df)} cases ({results_df['kdtree_faster'].mean()*100:.1f}%)")

    # Best configurations
    print("\n3. BEST CONFIGURATIONS:")
    best_knn = results_df.loc[results_df['knn_accuracy'].idxmax()]
    best_kdtree = results_df.loc[results_df['kdtree_accuracy'].idxmax()]

    print(f"   Best KNN: {best_knn['n_rows']} rows, {best_knn['n_features']} features, {best_knn['n_neighbors']} neighbors")
    print(f"   Accuracy: {best_knn['knn_accuracy']:.4f}, Time: {best_knn['knn_total_time']:.4f}s")

    print(f"   Best KDTree: {best_kdtree['n_rows']} rows, {best_kdtree['n_features']} features, {best_kdtree['n_neighbors']} neighbors")
    print(f"   Accuracy: {best_kdtree['kdtree_accuracy']:.4f}, Time: {best_kdtree['kdtree_total_time']:.4f}s")

    # Performance by dimensions
    print("\n4. PERFORMANCE BY FEATURE COUNT:")
    feature_analysis = results_df.groupby('n_features').agg({
        'knn_accuracy': 'mean',
        'kdtree_accuracy': 'mean',
        'knn_total_time': 'mean',
        'kdtree_total_time': 'mean'
    }).round(4)
    print(feature_analysis)

    print("\n5. PERFORMANCE BY DATASET SIZE:")
    size_analysis = results_df.groupby('n_rows').agg({
        'knn_accuracy': 'mean',
        'kdtree_accuracy': 'mean',
        'knn_total_time': 'mean',
        'kdtree_total_time': 'mean'
    }).round(4)
    print(size_analysis)

    return results_df

def main():
    """
    Main execution function
    """
    print("KNN vs KDTree Performance Comparison")
    print("="*50)

    # Load and prepare data
    X, y = load_and_prepare_data()

    # Get feature importance
    feature_importance = get_feature_importance(X, y, method=FEATURE_IMPORTANCE_METHOD)

    # Run comprehensive comparison
    # Note: Using smaller ranges for demonstration - adjust as needed
    results_df = run_comprehensive_comparison(
        X, y, feature_importance,
        row_sizes=ROW_SIZES, # Adjust sizes based on computational resources
        feature_counts=FEATURE_COUNTS,
        neighbor_counts=NEIGHBORS_COUNTS
    )

    # Analyze results
    final_results = analyze_results(results_df)

    # Save results
    os.makedirs('./output', exist_ok=True)
    os.makedirs('./output/covertype', exist_ok=True)
    os.makedirs('./output/covertype/results', exist_ok=True)
    output_filename = './output/covertype/results/knn_kdtree_comparison_results.csv'
    final_results.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")

    return final_results

if __name__ == "__main__":
    # Execute the comparison
    results = main()

    # Display first few results
    print("\nFirst 10 results:")
    print(results.head(10).to_string(index=False))

    # Accuracy comparison
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.scatter(results['knn_accuracy'], results['kdtree_accuracy'], alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('KNN Accuracy')
    plt.ylabel('KDTree Accuracy')
    plt.title('Accuracy Comparison')

    plt.subplot(2, 2, 2)
    plt.scatter(results['knn_total_time'], results['kdtree_total_time'], alpha=0.6)
    plt.plot([0, results['knn_total_time'].max()], [0, results['knn_total_time'].max()], 'r--')
    plt.xlabel('KNN Total Time (s)')
    plt.ylabel('KDTree Total Time (s)')
    plt.title('Time Comparison')

    plt.subplot(2, 2, 3)
    accuracy_by_features = results.groupby('n_features')[['knn_accuracy', 'kdtree_accuracy']].mean()
    accuracy_by_features.plot(kind='line', marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Accuracy')
    plt.title('Accuracy vs Number of Features')
    plt.legend()

    plt.subplot(2, 2, 4)
    time_by_features = results.groupby('n_features')[['knn_total_time', 'kdtree_total_time']].mean()
    time_by_features.plot(kind='line', marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Total Time (s)')
    plt.title('Time vs Number of Features')
    plt.legend()

    plt.tight_layout()
    os.makedirs('./output/', exist_ok=True)
    os.makedirs('./output/covertype/', exist_ok=True)
    os.makedirs('./output/covertype/plots/', exist_ok=True)
    plt.savefig('./output/covertype/plots/knn_kdtree_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nPlots saved as 'knn_kdtree_comparison_plots.png'")