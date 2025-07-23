import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.feature_selection import VarianceThreshold
# import matplotlib.pyplot as plt
import logging 
from tqdm import tqdm
# global progress_status
from sklearn.decomposition import PCA
from progress_tracker import progress_status
import re
import openai
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
import ast
import difflib
from  difflib import get_close_matches
from cluster import cluster_data,cluster_stats,cluster_top_underwriters,clusters_top_10_policies


logging.basicConfig(level=logging.INFO)
def fuzzy_match_filter_keys(filters, df_columns, threshold=0.6):
    matched_filters = {}

    for f_key, condition in filters.items():
        close_matches = difflib.get_close_matches(f_key, df_columns, n=1, cutoff=threshold)
        
        if close_matches:
            matched_col = close_matches[0]
            matched_filters[matched_col] = condition
        else:
            print(f"❌ No close match found for: {f_key}")

    return matched_filters

# Suppose raw_filters is a string (from GPT or response)
 # or however you access it

# Convert string to dictionary



def parse_filters(statement):
    filters = {}

    # Match country
    country_match = re.search(r'country\s+([\w\s]+?)(?:,|and|in|$)', statement, re.I)
    if country_match:
        filters['Insured Country'] = {'op': '==', 'value': country_match.group(1).strip()}

    # Match underwriter year
    year_match = re.search(r'underwriter year\s+(\d{4})', statement, re.I)
    if year_match:
        filters['Underwriting Year'] = {'op': '==', 'value': int(year_match.group(1))}

    # Match multiple classes
    class_match = re.search(r'class\s+([a-z\s,]+)', statement, re.I)
    if class_match:
        classes_raw = class_match.group(1)
        class_list = re.split(r'\s+and\s+|,|\s+', classes_raw)
        class_list = [cls.strip().title() for cls in class_list if cls.strip()]
        if len(class_list) == 1:
            filters['Class'] = {'op': '==', 'value': class_list[0]}
        elif len(class_list) > 1:
            filters['Class'] = {'op': 'in', 'value': class_list}

    return filters


def apply_filters(df, filters):
   

    fuzzy_columns = ['Insured Country', 'Class']  # Only these will use fuzzy value matching

    for col, condition in filters.items():
        matched_col = None

        # Step 1: Find the right column
        if col in df.columns:
            matched_col = col
        else:
            # fallback fuzzy matching for column name if required
            matched_col = col if col in df.columns else None
            if not matched_col:
                print(f"⚠️ Column '{col}' not found.")
                continue

        op = condition.get('op')
        val = condition.get('value')

        # Step 2: Fuzzy match values if this column is allowed
        if matched_col in fuzzy_columns and isinstance(val, list):
            # Fuzzy match each value in list
            all_values = df[matched_col].dropna().astype(str).unique().tolist()
            fuzzy_vals = []

            for v in val:
                matches = get_close_matches(str(v), all_values, n=1, cutoff=0.6)
                if matches:
                    fuzzy_vals.append(matches[0])
            if not fuzzy_vals:
                print(f"⚠️ No fuzzy match found for values in {matched_col}")
                continue
            val = fuzzy_vals

        elif matched_col in fuzzy_columns and isinstance(val, str):
            # Fuzzy match a single string
            all_values = df[matched_col].dropna().astype(str).unique().tolist()
            matches = get_close_matches(str(val), all_values, n=1, cutoff=0.6)
            if matches:
                val = matches[0]
            else:
                print(f"⚠️ No fuzzy match found for value '{val}' in {matched_col}")
                continue

        # Step 3: Apply the condition
        try:
            if op == '==':
                df = df[df[matched_col] == val]
            elif op == '!=':
                df = df[df[matched_col] != val]
            elif op == '>':
                df = df[df[matched_col] > val]
            elif op == '>=':
                df = df[df[matched_col] >= val]
            elif op == '<':
                df = df[df[matched_col] < val]
            elif op == '<=':
                df = df[df[matched_col] <= val]
            elif op == 'in':
                df = df[df[matched_col].isin(val)]
            elif op == 'not in':
                df = df[~df[matched_col].isin(val)]
        except Exception as e:
            print(f"❌ Error applying filter on '{matched_col}': {e}")

    return df
    
# Load your dataset
def run_kmeans_and_return_json(prompt):
    df = pd.read_csv("./data2.csv")  # replace with your actual file
    api_key=os.getenv("OPEN_API_KEY")
   




    print("api_key---->",api_key)
    if api_key:
    # ---------------------------------------------chatgpt-code---------------------------------------------
        print("-----------------------------open api called------------------------------------------") 
        user_prompt = f"""
        Convert the following SENTENCE into a structured Python dictionary of filters like  
        'Class': {{'op': '==', 'value': 'Energy'}},
                'Insured Country': {{'op': 'in', 'value': ['India', 'US']}},
                'Underwriting Year': {{'op': '>', 'value': 2000}}
            if value is single "op" should be "==" , if value is multiple or array it should be "in" and no explainataion.
            SENTENCE:=> '{prompt}'
            """


        client = OpenAI(api_key=api_key)  # replace with your actual key
        try:
            response = client.models.list()
            print("✅ Connection successful. You can access the API.")
        except Exception as e:
            print("❌ API call failed:", e)
            

        # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def ask_gpt():
            return client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0
            )

        response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0
            )
        print("filters from openai---->",response.choices[0].message.content)
        filters_raw=response.choices[0].message.content
        filters=fuzzy_match_filter_keys(ast.literal_eval(filters_raw),df.columns.tolist())
        print("filters from fuzzy matching------>",filters)
        df=apply_filters(df,filters)

#-----------------------------------------------------------------------------------------------------------
    
    # filters = {
    #     # 'Class': {'op': '==', 'value': 'Energy'},
    #     # 'Insured Country': {'op': 'in', 'value': ['India', 'USA']},
    #     'Underwriting Year': {'op': '>', 'value': 2000}
    # }
    # print(prompt)

    # filters=parse_filters(prompt)
    # print(filters)
    # df=apply_filters(df,filters)

    # -----------------------------------------------demo------------------------------------------
    # filters_raw="{'New Renewal': {'op': '==', 'value': 'Renewal'} }"
    else:
        print("-----------------------------default called------------------------------------------") 
        filters_raw="{'Insured Country':{'op': 'in', 'value': ['India','USA']}}"
        filters=fuzzy_match_filter_keys(ast.literal_eval(filters_raw),df.columns.tolist())
        
        df=apply_filters(df,filters)
    #------------------------------------------------------------------------------------------


    global_summary = {
        "country_stats": df['Insured Country'].value_counts().to_dict(),
        "unique_countries": df['Insured Country'].nunique(),
        "total_policy_volume": len(df),
        "total_premium": float((df['Gross Premium'] ).sum()),
        "total_loss": float((df['Paid']+df['Outstanding']).sum())
    }

    

    numeric_data=df.select_dtypes(include=[np.number])

    # print(numeric_data)
    if len(numeric_data)>1:
        selector=VarianceThreshold(threshold=0.2)
        data=selector.fit_transform(numeric_data);
        # print(data)

        feature_cols = numeric_data.columns[selector.get_support()]  # replace with actual feature names
        data = pd.DataFrame(data,columns=feature_cols,index=numeric_data.index)

    else:
        data=numeric_data.copy()



    data_clean = data.fillna(data.min(numeric_only=True))

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_clean)
    pca=PCA(n_components=2)

    X_pca=pca.fit_transform(X_scaled)





    # Apply KMeans and compute silhouette scores for different k
    silhouette_scores = []
    K = list(range(2, 11)) # trying from 2 to 10 clusters

    desired_k = 5  # 👈 set the k you want to inspect
    percentiles = [0.95]

    for i,k in enumerate(tqdm(K, desc="Clustering")):
        progress_status["progress"] = int((i + 1) / len(K) * 100)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels, sample_size=10000, random_state=42)
        silhouette_scores.append(score)

        # Store labels back to original data
        data_clean['Cluster'] = labels

    k_optimal=K[np.argmax(silhouette_scores)]
    kmeans=KMeans(n_clusters=k_optimal,random_state=42)
    labels = kmeans.fit_predict(X_pca)
    data_clean['Cluster'] = labels
    df_filtered = df.loc[data_clean.index].copy()
    df_filtered['Cluster'] = labels
    df_filtered = df_filtered.fillna(data.min(numeric_only=True))

        # If current k is the one we care about, calculate and show percentiles
        
    cluster_profiles = data_clean.groupby('Cluster').quantile(percentiles)
    cluster_profiles = cluster_profiles.unstack(level=-1)
    cluster_profiles.columns = [f"{col[0]}_p{int(col[1]*100)}" for col in cluster_profiles.columns]
    cluster_profiles = cluster_profiles.reset_index()

            # Show the output
    # from IPython.display import display
    # display(df_filtered.columns)
   

    cluster_summary = {}

    # logging.info(df_filtered.columns)
 
    clusters_top_10_policies.clear()
    for cluster_id,group in df_filtered.groupby('Cluster'):
        # logging.info("Columns:", group.columns.tolist())
        group['Inception Date'] = pd.to_datetime(group['Inception Date'], format='mixed', dayfirst=True, errors='coerce')

        
        group['Inception_Year']=group['Inception Date'].dt.year
        print(group['Inception_Year'])
        top_10=group.sort_values(by="Gross Premium",ascending=False).head(10)
        clusters_top_10_policies[cluster_id]=top_10
        yearly_summary=(group.groupby('Inception_Year').agg(yearly_premium=('Gross Premium','sum'),yearly_count=('PolicyNo','count')).reset_index())
        yearly_breakdown={str(row['Inception_Year']):{
            'premium':row['yearly_premium'],
            'count':row['yearly_count']
        }
        for _,row in yearly_summary.iterrows()
        }

        gross_premium=group['Gross Premium'].tolist()
        cluster_summary[int(cluster_id)] = {
            "policy_volume": len(group),
            "total_premium": float((group['Gross Premium'] * group['OurShare']).sum()),
            "total_loss": float(group['Paid'].sum()),
            "yearly_breakdown":yearly_breakdown,
        #     "mean_premium": float(group['Gross Premium'].mean()),
        # "median_premium": float(group['Gross Premium'].median()),
        # "min_premium": float(min(gross_premium)),
        # "max_premium": float(max(gross_premium)),
        # "percentile_90": float(np.percentile(gross_premium, 90)),
            "loss_ratio": float((group['Paid']+group['Outstanding']).sum()) / max((group['Gross Premium']).sum(), 1)*100
        }
    # print(cluster_summary)

    # cluster_data = {}
    for cluster_id, group in df_filtered.groupby('Cluster'):
        top_underwriters = (
            group.groupby('Underwriter')['Gross Premium']
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
            .to_dict(orient='records')
        )
        
        cluster_data[int(cluster_id)] = group[[
            'PolicyNo', 'Gross Premium', 'Class', 'Inception Date', 'Expiry Date','Division','New Renewal'
        ]].to_dict(orient='records')
        gross_premiums = group['Gross Premium'].tolist()

        cluster_stats[int(cluster_id)] = {
            "mean": float(group['Gross Premium'].mean()),
            "median": float(group['Gross Premium'].median()),
            "min": float(min(gross_premiums)),
            "max": float(max(gross_premiums)),
            "percentile_90": float(np.percentile(gross_premiums, 90))
        }
        cluster_top_underwriters[int(cluster_id)] = top_underwriters


    import json

    final_summary = {
        "global": global_summary,
        "clusters": cluster_summary,
        "cluster_stats":cluster_stats,
        "cluster_top_underwriters":cluster_top_underwriters,
         "filters":filters
    }

    return final_summary

   







    # for col, condition in filters.items():
        # if col not in df.columns:
        #     continue

        # op = condition.get('op')
        # val = condition.get('value')

        # if op == '==':
        #     df = df[df[col] == val]
        # elif op == '!=':
        #     df = df[df[col] != val]
        # elif op == '>':
        #     df = df[df[col] > val]
        # elif op == '>=':
        #     df = df[df[col] >= val]
        # elif op == '<':
        #     df = df[df[col] < val]
        # elif op == '<=':
        #     df = df[df[col] <= val]
        # elif op == 'in':
        #     df = df[df[col].isin(val)]
        # elif op == 'not in':
        #     df = df[~df[col].isin(val)]