from flask import Flask, request, jsonify
from progress_tracker import progress_status
from kmeans_script import run_kmeans_and_return_json
from flask_cors import CORS
from cluster import cluster_data,clusters_top_10_policies
import pandas as pd
from linesizeoptimization import calculate_loss_ratio,optimize_portfolio
app = Flask(__name__)
CORS(app)
@app.route("/reset-progress", methods=["POST"])
def reset_progress():
    progress_status["progress"] = 0

    return jsonify({"message": "Progress reset to 0"})

@app.route("/progress", methods=["GET"])
def get_progress():
    return jsonify(progress_status)


@app.route('/cluster/<int:cluster_id>', methods=['GET'])
def get_cluster_data(cluster_id):
    if cluster_id not in cluster_data:
        return jsonify({'error': f'Cluster {cluster_id} not found'}), 404

    return jsonify({
        'cluster_id': cluster_id,
        'data': cluster_data[cluster_id]
    })

@app.route('/run-kmeans', methods=['POST'])
def run_kmeans():
    progress_status["progress"]=0
    cluster_data={}
    data = request.get_json()
    prompt = data.get('prompt',None)
    print(prompt)
    result = run_kmeans_and_return_json(prompt)
    return jsonify(result)

@app.route('/linesize', methods=['POST'])
def linesize():
    progress_status["progress"] = 0
    data = request.get_json()
    result = {}

    print(clusters_top_10_policies)
    # for cluster_id, policies in clusters_top_10_policies.items():
    #     transformed = []
    #     for _, p in policies.iterrows():
    #         transformed.append({
    #             "Policy_ID": p["PolicyNo"],
    #             "Gross_Premium": p["Gross Premium"],
    #             "Loss": p["Paid"] + p["Outstanding"],
    #             "Original_LineSize": p["OurShare"],
    #             "Renewal": p["Renewed Flag"]
    #         })
    #     result[cluster_id] = transformed
    for cluster_id, policies in clusters_top_10_policies.items():
        df_sample = pd.DataFrame({
            "Policy_ID": [p["PolicyNo"] for _, p in policies.iterrows()],
            "Gross_Premium": [p["Gross Premium"] for _, p in policies.iterrows()],
            "Loss": [p["Paid"] + p["Outstanding"] for _, p in policies.iterrows()],
            "Original_LineSize": [p["OurShare"] for _, p in policies.iterrows()],
            "Renewal": [p["Renewed Flag"] for _, p in policies.iterrows()],
           
        })

        # Constraints (these can be dynamic if needed)
        constraints = {
        "target_gross_premium": float(data.get("target_gross_premium", 150000)),
        "min_gross_premium": float(data.get("min_gross_premium", 140000)),
        "max_line_change_pct": float(data.get("max_line_change_pct", 0.2)) / 100,
        "max_non_renewal_pct": float(data.get("max_non_renewal_pct", 0.3)) / 100,
        "max_pml_pct": float(data.get("max_pml_pct", 0.8)) / 100
        }

        original_loss_ratio = calculate_loss_ratio(df_sample, 'Original_LineSize','Renewal')
        optimized_df = optimize_portfolio(
            df_sample.copy(),
            constraints["target_gross_premium"],
            constraints["max_pml_pct"],
            constraints["max_line_change_pct"],
            constraints["max_non_renewal_pct"]
        )
        new_loss_ratio = calculate_loss_ratio(optimized_df, 'Optimized_LineSize','Optimized_Renewal')

        # Build the result
        policies_result = optimized_df[[
            "Policy_ID", "Gross_Premium", "Loss", "Original_LineSize", "Optimized_LineSize", "Renewal",
            'Optimized_Renewal'
        ]].to_dict(orient="records")

        result[cluster_id] = {
            "policies": policies_result,
            "Original_Loss_Ratio": round(original_loss_ratio * 100, 2),
            "New_Loss_Ratio": round(new_loss_ratio * 100, 2)
        }

    return jsonify(result)
    # return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)