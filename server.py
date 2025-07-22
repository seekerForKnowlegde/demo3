from flask import Flask, request, jsonify
from progress_tracker import progress_status
from kmeans_script import run_kmeans_and_return_json
from flask_cors import CORS
from cluster import cluster_data
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)