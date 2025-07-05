from flask import Flask, request, jsonify
from model import EmotionActivityEnsemble, Config, predict_emotion_activity_manual
import emotion_state
from flask_cors import CORS

config = Config()
model = EmotionActivityEnsemble(config)
model.load_models()

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    predictions, confidence = predict_emotion_activity_manual(model, data)
    emotion_state.current_emotion = predictions[0]  # update global emotion
    return jsonify({
        "emotion": predictions[0],
        "confidence": float(confidence[0])
    })

@app.route("/current_emotion", methods=["GET"])
def get_current_emotion():
    return jsonify({
        "emotion": emotion_state.current_emotion
    })

if __name__ == "__main__":
    app.run(debug=True)
