# train_model.py

# Data handling and math
import pandas as pd
import numpy as np

# ---- replace these lines ----
# from pycaret.classification import *
# from pycaret.clustering import *

# ---- with this: ----
from pycaret import classification as pcc
from pycaret import clustering as pcl


# OS utilities
import os
import shutil

# (Imported but only used indirectly by PyCaret plotting)
import matplotlib.pyplot as plt


def generate_synthetic_data(num_samples=600, profile_weights=(0.34, 0.33, 0.33), random_state=42):
    """Generate a synthetic URL dataset with logical patterns for three phishing actor profiles
    (State-Sponsored, Organized Cybercrime, Hacktivist) + benign traffic."""
    rng = np.random.default_rng(random_state)
    print("Generating synthetic dataset with three threat actor profiles...")

    features = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service',
        'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
        'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags',
        'SFH', 'Abnormal_URL',
        'has_political_keyword'  # NEW
    ]

    num_phishing = num_samples // 2
    num_benign = num_samples - num_phishing

    w_state, w_crime, w_hackt = profile_weights
    n_state = int(round(num_phishing * w_state))
    n_crime = int(round(num_phishing * w_crime))
    n_hackt = num_phishing - n_state - n_crime

    def sample_state_sponsored(n):
        return pd.DataFrame({
            'having_IP_Address': np.random.choice([1, -1], n, p=[0.10, 0.90]),
            'URL_Length':        np.random.choice([1, 0, -1], n, p=[0.25, 0.60, 0.15]),
            'Shortining_Service':np.random.choice([1, -1], n, p=[0.10, 0.90]),
            'having_At_Symbol':  np.random.choice([1, -1], n, p=[0.10, 0.90]),
            'double_slash_redirecting': np.random.choice([1, -1], n, p=[0.10, 0.90]),
            'Prefix_Suffix':     np.random.choice([1, -1], n, p=[0.70, 0.30]),
            'having_Sub_Domain': np.random.choice([1, 0, -1], n, p=[0.50, 0.40, 0.10]),
            'SSLfinal_State':    np.random.choice([1, 0, -1], n, p=[0.80, 0.15, 0.05]),
            'URL_of_Anchor':     np.random.choice([1, 0, -1], n, p=[0.40, 0.45, 0.15]),
            'Links_in_tags':     np.random.choice([1, 0, -1], n, p=[0.35, 0.50, 0.15]),
            'SFH':               np.random.choice([1, 0, -1], n, p=[0.30, 0.50, 0.20]),
            'Abnormal_URL':      np.random.choice([1, -1], n, p=[0.40, 0.60]),
            'has_political_keyword': np.random.choice([1, -1], n, p=[0.10, 0.90]),
        })

    def sample_organized_crime(n):
        return pd.DataFrame({
            'having_IP_Address': np.random.choice([1, -1], n, p=[0.75, 0.25]),
            'URL_Length':        np.random.choice([1, 0, -1], n, p=[0.70, 0.20, 0.10]),
            'Shortining_Service':np.random.choice([1, -1], n, p=[0.80, 0.20]),
            'having_At_Symbol':  np.random.choice([1, -1], n, p=[0.60, 0.40]),
            'double_slash_redirecting': np.random.choice([1, -1], n, p=[0.65, 0.35]),
            'Prefix_Suffix':     np.random.choice([1, -1], n, p=[0.75, 0.25]),
            'having_Sub_Domain': np.random.choice([1, 0, -1], n, p=[0.70, 0.20, 0.10]),
            'SSLfinal_State':    np.random.choice([1, 0, -1], n, p=[0.10, 0.20, 0.70]),
            'URL_of_Anchor':     np.random.choice([1, 0, -1], n, p=[0.70, 0.20, 0.10]),
            'Links_in_tags':     np.random.choice([1, 0, -1], n, p=[0.65, 0.25, 0.10]),
            'SFH':               np.random.choice([1, 0, -1], n, p=[0.70, 0.20, 0.10]),
            'Abnormal_URL':      np.random.choice([1, -1], n, p=[0.85, 0.15]),
            'has_political_keyword': np.random.choice([1, -1], n, p=[0.05, 0.95]),
        })

    def sample_hacktivist(n):
        return pd.DataFrame({
            'having_IP_Address': np.random.choice([1, -1], n, p=[0.35, 0.65]),
            'URL_Length':        np.random.choice([1, 0, -1], n, p=[0.45, 0.35, 0.20]),
            'Shortining_Service':np.random.choice([1, -1], n, p=[0.45, 0.55]),
            'having_At_Symbol':  np.random.choice([1, -1], n, p=[0.35, 0.65]),
            'double_slash_redirecting': np.random.choice([1, -1], n, p=[0.30, 0.70]),
            'Prefix_Suffix':     np.random.choice([1, -1], n, p=[0.50, 0.50]),
            'having_Sub_Domain': np.random.choice([1, 0, -1], n, p=[0.45, 0.35, 0.20]),
            'SSLfinal_State':    np.random.choice([1, 0, -1], n, p=[0.30, 0.40, 0.30]),
            'URL_of_Anchor':     np.random.choice([1, 0, -1], n, p=[0.50, 0.35, 0.15]),
            'Links_in_tags':     np.random.choice([1, 0, -1], n, p=[0.45, 0.40, 0.15]),
            'SFH':               np.random.choice([1, 0, -1], n, p=[0.50, 0.30, 0.20]),
            'Abnormal_URL':      np.random.choice([1, -1], n, p=[0.55, 0.45]),
            'has_political_keyword': np.random.choice([1, -1], n, p=[0.70, 0.30]),
        })

    df_state = sample_state_sponsored(n_state); df_state['actor_profile'] = 'state'
    df_crime = sample_organized_crime(n_crime); df_crime['actor_profile'] = 'crime'
    df_hackt = sample_hacktivist(n_hackt);      df_hackt['actor_profile'] = 'hacktivist'
    df_phishing = pd.concat([df_state, df_crime, df_hackt], ignore_index=True)
    df_phishing['label'] = 1

    benign = pd.DataFrame({
        'having_IP_Address': np.random.choice([1, -1], num_benign, p=[0.03, 0.97]),
        'URL_Length':        np.random.choice([1, 0, -1], num_benign, p=[0.10, 0.60, 0.30]),
        'Shortining_Service':np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'having_At_Symbol':  np.random.choice([1, -1], num_benign, p=[0.03, 0.97]),
        'double_slash_redirecting': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'Prefix_Suffix':     np.random.choice([1, -1], num_benign, p=[0.08, 0.92]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_benign, p=[0.10, 0.45, 0.45]),
        'SSLfinal_State':    np.random.choice([1, 0, -1], num_benign, p=[0.85, 0.10, 0.05]),
        'URL_of_Anchor':     np.random.choice([1, 0, -1], num_benign, p=[0.15, 0.25, 0.60]),
        'Links_in_tags':     np.random.choice([1, 0, -1], num_benign, p=[0.15, 0.25, 0.60]),
        'SFH':               np.random.choice([1, 0, -1], num_benign, p=[0.10, 0.10, 0.80]),
        'Abnormal_URL':      np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'has_political_keyword': np.random.choice([1, -1], num_benign, p=[0.02, 0.98]),
    })
    benign['label'] = 0
    benign['actor_profile'] = 'benign'

    final_df = pd.concat([df_phishing, benign], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return final_df


def train_supervised(data):
    model_path = 'models/phishing_url_detector'
    plot_path = 'models/feature_importance.png'
    os.makedirs('models', exist_ok=True)

    print("Initializing PyCaret Classification Setup...")
    s = pcc.setup(                             # <--- use pcc
        data,
        target='label',
        session_id=42,
        verbose=False,
        ignore_features=['actor_profile'],
        fold=3,
        n_jobs=-1
    )

    print("Comparing models...")
    best_model = pcc.compare_models(n_select=1, include=['rf', 'et', 'lightgbm'])

    print("Finalizing model...")
    final_model = pcc.finalize_model(best_model)

    print("Saving feature importance plot...")
    pcc.plot_model(final_model, plot='feature', save=True)
    if os.path.exists('Feature Importance.png'):
        try:
            os.replace('Feature Importance.png', plot_path)
        except Exception:
            shutil.copyfile('Feature Importance.png', plot_path)

    print("Saving classifier...")
    pcc.save_model(final_model, model_path)
    print("Supervised model saved.")



def train_unsupervised(data):
    cluster_model_path = 'models/threat_actor_profiler'
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    features_df = data.drop(columns=[c for c in ['label', 'actor_profile'] if c in data.columns])

    print("Initializing PyCaret Clustering Setup (features only)...")
    cs = pcl.setup(                            # <--- use pcl
        features_df,
        session_id=42,
        normalize=True,
        transformation=True,
        verbose=False
    )

    print("Creating K-Means (k=3)...")
    kmeans3 = pcl.create_model('kmeans', num_clusters=3)

    print("Assigning cluster labels...")
    clustered = pcl.assign_model(kmeans3)

    out = data.copy().reset_index(drop=True)
    out['cluster_id'] = clustered['Cluster'].values
    out.to_csv('data/phishing_with_clusters.csv', index=False)

    try:
        pcl.plot_model(kmeans3, plot='elbow', save=True)
        pcl.plot_model(kmeans3, plot='silhouette', save=True)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("Saving clustering model...")
    pcl.save_model(kmeans3, cluster_model_path)
    print("Unsupervised model saved.")



def train():
    """End-to-end: generate data, train & save BOTH models every run."""
    # Fresh dirs
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 1) Data
    data = generate_synthetic_data()
    data.to_csv('data/phishing_synthetic.csv', index=False)

    # 2) Supervised
    train_supervised(data)

    # 3) Unsupervised
    train_unsupervised(data)

    print("All artifacts saved to ./models and ./data.")


if __name__ == "__main__":
    train()
