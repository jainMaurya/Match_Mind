from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ------------------ Helper to load models safely ------------------ #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(filename):
    """Load a pickle model if it exists, else return None."""
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
            return None
    else:
        print(f"⚠️ {filename} not found in this environment.")
        return None


# ------------------ Load models (local vs Render) ------------------ #

score_model = load_model("ipl1_model.pkl")        # big → only local
batsman_model = load_model("batsmanmodel.pkl")    # big → only local
win_model = load_model("winmodel.pkl")            # smaller → also on Render


# ------------------ Routes ------------------ #

@app.route("/")
def home():
    return render_template("index.html")


# 1️⃣ IPL Score Predictor
@app.route("/predict", methods=["POST"])
def predict():
    # On Render, score_model will be None because ipl1_model.pkl isn't deployed
    if score_model is None:
        return jsonify({
            "error": "Score prediction model is not available in the deployed version."
        }), 200

    try:
        data = request.form

        batting_team = data.get("battingTeam")
        bowling_team = data.get("bowlingTeam")
        over = float(data.get("over", 0))
        runs = int(data.get("runs", 0))
        wickets = int(data.get("wickets", 0))
        runs5overs = int(data.get("runs5overs", 0))
        wickets5overs = int(data.get("wickets5overs", 0))
        city = "Mumbai"  # or take from user later

        row = [[
            batting_team,
            bowling_team,
            city,
            runs,
            wickets,
            over,
            runs5overs,
            wickets5overs
        ]]

        columns = [
            "BattingTeam",
            "BowlingTeam",
            "City",
            "runs",
            "wickets",
            "overs",
            "runs_last_5",
            "wickets_last_5",
        ]

        df = pd.DataFrame(row, columns=columns)

        prediction = score_model.predict(df)[0]
        return jsonify({"prediction": int(round(float(prediction)))})

    except Exception as e:
        print("Error in /predict route:", e)
        return jsonify({"error": "Unable to compute score prediction."}), 200


# 2️⃣ Player Performance Predictor
@app.route("/predict_batsman", methods=["POST"])
def predict_batsman():
    # On Render, batsman_model will be None
    if batsman_model is None:
        return jsonify({
            "error": "Player performance model is not available in the deployed version."
        }), 200

    try:
        data = request.form

        batter = data.get("batter")
        bowling_team = data.get("BowlingTeam")
        city = data.get("City")
        toss_decision = data.get("TossDecision")

        cols = ["batter", "BowlingTeam", "City", "TossDecision"]
        row = [[batter, bowling_team, city, toss_decision]]
        df = pd.DataFrame(row, columns=cols)

        preds = batsman_model.predict(df)[0]

        result = {
            "TotalRuns":   round(float(preds[0]), 1),
            "StrikeRate":  round(float(preds[1]), 1),
            "Avg4s":       round(float(preds[2]), 1),
            "Avg6s":       round(float(preds[3]), 1),
            "ImpactScore": round(float(preds[4]), 1),
        }

        return jsonify(result)

    except Exception as e:
        print("Error in /predict_batsman route:", e)
        return jsonify({"error": "Unable to compute player performance."}), 200


# 3️⃣ Win Probability Predictor
@app.route("/predict_win_probability", methods=["POST"])
def predict_win_probability():
    if win_model is None:
        return jsonify({
            "error": "Win probability model is not available."
        }), 200

    try:
        data = request.form
        batting_team = data.get("battingTeam")
        bowling_team = data.get("bowlingTeam")
        city = data.get("city")
        runs_left = int(data.get("runsleft", 0))
        balls_left = int(data.get("ballsleft", 0))
        wickets_left = int(data.get("wicketsleft", 0))
        currrr = float(data.get("currrr", 0))
        reqrr = float(data.get("reqrr", 0))
        target = int(data.get("target", 0))

        l = [[batting_team, bowling_team, city,
              runs_left, balls_left, wickets_left,
              currrr, reqrr, target]]

        columns = [
            "BattingTeam", "BowlingTeam", "City", "runs_left", "balls_left",
            "wickets_left", "current_run_rate", "required_run_rate", "target"
        ]

        team2023 = pd.DataFrame(l, columns=columns)

        prob = win_model.predict_proba(team2023)[0][1]
        return jsonify({"prediction": float(prob)})

    except Exception as e:
        print("Error in /predict_win_probability route:", e)
        return jsonify({"error": "Unable to compute win probability."}), 200


# ------------------ Entry point ------------------ #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
