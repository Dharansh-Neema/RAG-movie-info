from flask import Flask, render_template, request, jsonify
from src.retrieval import Retrieval
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")  # Serves the frontend

@app.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.json
    query = data.get("query", "")
    retrievalInstance = Retrieval()
    response = retrievalInstance.llm_response(query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
