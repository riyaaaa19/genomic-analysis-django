from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import io
import base64
from collections import Counter
from imblearn.over_sampling import SMOTE

logging.basicConfig(filename='analysis_log.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

UPLOAD_DIR = 'uploads/'
RESULT_DIR = 'analysis/static/results/'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def index(request):
    return render(request, 'analysis/index.html')

def analyze(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        file = request.FILES['csv_file']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(file.name, file)
        filepath = os.path.join(UPLOAD_DIR, filename)

        try:
            df = pd.read_csv(filepath)

            # Identify target column heuristically
            target_column = None
            for col in df.columns:
                if col.lower() not in ['id', 'patient id', 'name']:
                    if df[col].dtype == 'object' or df[col].nunique() <= 10:
                        if df[col].nunique() > 1 and df[col].isnull().mean() < 0.3:
                            target_column = col
                            break

            logging.debug(f"Target column: {target_column}")

            if target_column is None:
                return perform_kmeans(request, df, "No suitable target column found — ran clustering instead.")

            df[target_column] = df[target_column].astype(str).replace(['NA', 'N/A', '?'], np.nan)
            df = df.dropna(subset=[target_column])

            X = df.drop(columns=[target_column])
            y = df[target_column]

            nan_mask = X.isnull().any(axis=1) | y.isnull()
            X = X[~nan_mask].reset_index(drop=True)
            y = y[~nan_mask].reset_index(drop=True).astype(str)

            # --- Always define context variables for the template ---
            context = {
                'cluster_img': None, 'tree_img': None, 'report': None,
                'insights': None, 'target_column': target_column, 'dist_img': None,
                'conf_matrix_img': None, 'report_file': None,
                'feature_img': None, 'model_name': None
            }

            if len(X) < 20:
                context.update({
                    'insights': ['Dataset too small for meaningful prediction. Please upload at least 20 rows.'],
                })
                return render(request, 'analysis/result.html', context)

            if y.nunique() > 50:
                return perform_kmeans(request, X, f"Too many unique labels in target ({y.nunique()}) — classification skipped.")

            X = X.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
            X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

            selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
            selector.fit(X, y)
            selected_feat = X.columns[selector.get_support()]

            if len(selected_feat) == 0:
                context.update({
                    'insights': ["No features were selected. Please check your data."],
                })
                return render(request, 'analysis/result.html', context)

            X = X[selected_feat]
            X_scaled = StandardScaler().fit_transform(X)

            # Class distribution plot
            label_counts = Counter(y)
            fig = plt.figure()
            plt.bar(label_counts.keys(), label_counts.values())
            plt.title(f"Distribution of Target Labels: {target_column}")
            plt.xlabel("Class")
            plt.ylabel("Count")
            tmpfile = io.BytesIO()
            fig.savefig(tmpfile, format='png')
            dist_img = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            plt.close(fig)
            context['dist_img'] = dist_img

            # Clustering plot with PCA
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X_scaled)
            cluster_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)
            fig = plt.figure()
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
            plt.title("K-Means Clustering (PCA-reduced)")
            tmpfile = io.BytesIO()
            fig.savefig(tmpfile, format='png')
            encoded_cluster = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            plt.close(fig)
            context['cluster_img'] = encoded_cluster

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
            )

            # --- Always use Decision Tree ---
            clf = DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=3)
            model_display = "Decision Tree"
            context['model_name'] = model_display

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_txt = classification_report(y_test, y_pred, zero_division=0)

            # Save report to file
            report_path = os.path.join(RESULT_DIR, 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(report_txt)
            context['report_file'] = os.path.basename(report_path)

            # Cross-validation
            if len(X) >= 10:
                min_samples_per_class = min([(y_encoded == i).sum() for i in range(len(set(y_encoded)))])
                n_splits = min(5, min_samples_per_class)
                if n_splits > 1:
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(clf, X_scaled, y_encoded, cv=cv)
                    cv_mean = np.mean(cv_scores)
                else:
                    cv_scores = []
                    cv_mean = 0
            else:
                cv_scores = []
                cv_mean = 0

            if 'macro avg' in report_dict:
                report_dict['macro_avg'] = report_dict.pop('macro avg')
            if 'weighted avg' in report_dict:
                report_dict['weighted_avg'] = report_dict.pop('weighted avg')

            insights = []
            acc = report_dict.get('accuracy', 0)
            if acc < 0.6:
                insights.append(
                    f"The model accuracy is {acc:.2f}, which suggests it struggled to identify reliable patterns. "
                    "This could be due to limited or noisy input data. Consider uploading a cleaner or larger dataset."
                )
            if len(cv_scores) > 0:
                insights.append(f"Cross-validated accuracy: {cv_mean:.2f}, showing general performance stability.")
            context['insights'] = insights
            context['report'] = report_dict

            # Decision Tree Plot
            fig = plt.figure(figsize=(30, 15))
            plot_tree(clf, feature_names=X.columns, class_names=list(label_encoder.classes_),
                      filled=True, fontsize=12, proportion=True, precision=2)
            tmpfile = io.BytesIO()
            fig.savefig(tmpfile, format='png', bbox_inches='tight')
            encoded_tree = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            plt.close(fig)
            context['tree_img'] = encoded_tree

            # Confusion Matrix Plot
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax)
            tmpfile = io.BytesIO()
            fig.savefig(tmpfile, format='png')
            encoded_conf_matrix = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            plt.close(fig)
            context['conf_matrix_img'] = encoded_conf_matrix

            # --- Feature Importance Chart (Matplotlib) ---
            feature_img = None
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(X.columns, importances, color='skyblue')
                ax.set_title('Feature Importances')
                ax.set_xlabel('Importance')
                ax.invert_yaxis()
                plt.tight_layout()
                tmpfile = io.BytesIO()
                fig.savefig(tmpfile, format='png')
                feature_img = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                plt.close(fig)
            context['feature_img'] = feature_img

            return render(request, 'analysis/result.html', context)

        except Exception as e:
            logging.exception("Error during analysis")
            return render(request, 'analysis/index.html', {'error': f"Error during analysis: {str(e)}"})

    return render(request, 'analysis/index.html')


def perform_kmeans(request, df, message):
    df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    context = {
        'cluster_img': None, 'tree_img': None, 'report': None,
        'insights': None, 'target_column': None, 'dist_img': None,
        'conf_matrix_img': None, 'report_file': None,
        'feature_img': None, 'model_name': None
    }
    if df_numeric.empty:
        context['insights'] = ["No valid numeric data found."]
        return render(request, 'analysis/result.html', context)
    X = SimpleImputer(strategy='mean').fit_transform(df_numeric)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)
    cluster_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)

    fig = plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.title("K-Means Clustering (PCA-reduced)")
    tmpfile = io.BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    plt.close(fig)
    context['cluster_img'] = encoded
    context['insights'] = [message]
    return render(request, 'analysis/result.html', context)