"""
api_flask.py
OCP El Jadida — API REST Flask pour SmartPredict
"""
import os, json, pickle, time, random, smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

MODEL_PATH   = "models/smartpredict_model.pkl"
METRICS_PATH = "models/metrics.json"
FI_PATH      = "models/feature_importance.csv"

FEATURE_NAMES = ["TI171","TI170","TI169","VI167X","VI167Y","AZI165",
                 "T_moy","V_max","AZI_abs","DeltaT"]
CLASS_NAMES   = ["FAIBLE", "MOYEN", "CRITIQUE"]

# ── Chargement modèle ─────────────────────────────────────────
model = None
metrics_cache = {}

def load_model():
    global model, metrics_cache
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics_cache = json.load(f)

def engineer_features(d: dict) -> np.ndarray:
    ti171, ti170, ti169 = d["TI171"], d["TI170"], d["TI169"]
    vi167x, vi167y      = d["VI167X"], d["VI167Y"]
    azi165              = d["AZI165"]
    return np.array([[
        ti171, ti170, ti169, vi167x, vi167y, azi165,
        (ti171 + ti170 + ti169) / 3,
        max(vi167x, vi167y),
        abs(azi165),
        ti170 - ti169
    ]])

# ── Simulation capteurs temps réel ───────────────────────────
SENSOR_STATE = {"mode": 0, "step": 0}

def simulate_sensor():
    s = SENSOR_STATE
    s["step"] += 1
    if s["step"] % 40 == 0:
        s["mode"] = (s["mode"] + 1) % 3
    if s["mode"] == 0:
        return {
            "TI171":  round(random.gauss(70, 0.8), 2),
            "TI170":  round(random.gauss(75, 0.8), 2),
            "TI169":  round(random.gauss(76, 0.8), 2),
            "VI167X": round(random.gauss(13, 0.5), 3),
            "VI167Y": round(random.gauss(15, 0.5), 3),
            "AZI165": round(random.gauss(-0.31, 0.01), 3),
        }
    elif s["mode"] == 1:
        return {
            "TI171":  round(random.gauss(72, 1.5), 2),
            "TI170":  round(random.gauss(78, 1.5), 2),
            "TI169":  round(random.gauss(79, 1.5), 2),
            "VI167X": round(random.gauss(20, 2), 3),
            "VI167Y": round(random.gauss(22, 2), 3),
            "AZI165": round(random.gauss(-0.14, 0.02), 3),
        }
    else:
        return {
            "TI171":  round(random.gauss(38, 3), 2),
            "TI170":  round(random.gauss(37, 3), 2),
            "TI169":  round(random.gauss(37, 3), 2),
            "VI167X": round(random.gauss(1.5, 0.3), 3),
            "VI167Y": round(random.gauss(1.5, 0.3), 3),
            "AZI165": round(random.gauss(0.04, 0.01), 3),
        }

# ── Génération email HTML ─────────────────────────────────────
def build_email_html(sensors, proba, is_test=False):
    heure = datetime.now().strftime("%H:%M:%S")
    date  = datetime.now().strftime("%d/%m/%Y")
    p_crit = round(proba[2] * 100, 1)

    def row_status(val, lo, hi):
        ok = lo <= val <= hi
        color = "#006600" if ok else "#cc0000"
        status = "✓ Normal" if ok else "⚠ ANORMAL"
        return f'<td style="padding:6px 10px;border:1px solid #e0e0e0;color:{color};font-weight:bold">{status}</td>'

    subject_prefix = "[TEST]" if is_test else "🚨 CRITIQUE"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f0f0f0;font-family:Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f0f0f0;padding:24px 0">
  <tr><td align="center">
    <table width="580" cellpadding="0" cellspacing="0"
           style="background:#fff;border-radius:8px;overflow:hidden;
                  box-shadow:0 4px 16px rgba(0,0,0,0.12)">

      <!-- Header -->
      <tr><td style="background:{'#888' if is_test else '#cc0000'};padding:24px 32px;text-align:center">
        <div style="font-size:32px">{'🧪' if is_test else '🚨'}</div>
        <h1 style="color:#fff;margin:8px 0 4px;font-size:22px">
          {'EMAIL TEST — SmartPredict' if is_test else 'ALERTE CRITIQUE — Panne Imminente'}
        </h1>
        <p style="color:rgba(255,255,255,0.85);margin:0;font-size:13px">
          Pompe H₂SO₄ — OCP El Jadida — Ligne 401
        </p>
      </td></tr>

      <!-- KPIs -->
      <tr><td style="padding:20px 32px">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td style="background:#fff2f2;border:1px solid #ffcccc;border-radius:6px;
                       padding:14px;text-align:center" width="30%">
              <div style="font-size:11px;color:#888">Risque critique</div>
              <div style="font-size:28px;font-weight:bold;color:#cc0000">{p_crit}%</div>
            </td>
            <td width="5%"></td>
            <td style="background:#f8f8f8;border:1px solid #e0e0e0;border-radius:6px;
                       padding:14px;text-align:center" width="30%">
              <div style="font-size:11px;color:#888">Détecté à</div>
              <div style="font-size:18px;font-weight:bold;color:#333">{heure}</div>
              <div style="font-size:10px;color:#999">{date}</div>
            </td>
            <td width="5%"></td>
            <td style="background:#fff8e1;border:1px solid #ffe082;border-radius:6px;
                       padding:14px;text-align:center" width="30%">
              <div style="font-size:11px;color:#888">Action requise</div>
              <div style="font-size:13px;font-weight:bold;color:#cc6600">
                {'— Test —' if is_test else 'ARRÊT IMMÉDIAT'}
              </div>
            </td>
          </tr>
        </table>

        <!-- Capteurs -->
        <h3 style="color:#333;font-size:13px;margin:20px 0 10px;
                   border-bottom:2px solid #cc0000;padding-bottom:6px">
          État des capteurs SCADA
        </h3>
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border:1px solid #e0e0e0;border-radius:6px;overflow:hidden;
                      border-collapse:collapse;font-size:12px">
          <tr style="background:#f5f5f5">
            <th style="padding:8px 10px;text-align:left;border:1px solid #e0e0e0">Capteur</th>
            <th style="padding:8px 10px;text-align:left;border:1px solid #e0e0e0">Valeur</th>
            <th style="padding:8px 10px;text-align:left;border:1px solid #e0e0e0">Plage normale</th>
            <th style="padding:8px 10px;text-align:left;border:1px solid #e0e0e0">État</th>
          </tr>
          <tr>
            <td style="padding:6px 10px;border:1px solid #e0e0e0">TI-171 (Temp. palier 1)</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;font-weight:bold">{sensors.get('TI171',0):.1f} °C</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;color:#888">68 – 72 °C</td>
            {row_status(sensors.get('TI171',0), 68, 72)}
          </tr>
          <tr style="background:#fafafa">
            <td style="padding:6px 10px;border:1px solid #e0e0e0">TI-170 (Temp. palier 2)</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;font-weight:bold">{sensors.get('TI170',0):.1f} °C</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;color:#888">72 – 77 °C</td>
            {row_status(sensors.get('TI170',0), 72, 77)}
          </tr>
          <tr>
            <td style="padding:6px 10px;border:1px solid #e0e0e0">TI-169 (Temp. palier 3)</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;font-weight:bold">{sensors.get('TI169',0):.1f} °C</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;color:#888">72 – 78 °C</td>
            {row_status(sensors.get('TI169',0), 72, 78)}
          </tr>
          <tr style="background:#fafafa">
            <td style="padding:6px 10px;border:1px solid #e0e0e0">VI-167X (Vibration X)</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;font-weight:bold">{sensors.get('VI167X',0):.2f} mm/s</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;color:#888">10 – 16 mm/s</td>
            {row_status(sensors.get('VI167X',0), 10, 16)}
          </tr>
          <tr>
            <td style="padding:6px 10px;border:1px solid #e0e0e0">VI-167Y (Vibration Y)</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;font-weight:bold">{sensors.get('VI167Y',0):.2f} mm/s</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;color:#888">12 – 18 mm/s</td>
            {row_status(sensors.get('VI167Y',0), 12, 18)}
          </tr>
          <tr style="background:#fafafa">
            <td style="padding:6px 10px;border:1px solid #e0e0e0">AZI-165 (Position angulaire)</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;font-weight:bold">{sensors.get('AZI165',0):.3f} rad</td>
            <td style="padding:6px 10px;border:1px solid #e0e0e0;color:#888">-0.36 à -0.29</td>
            {row_status(sensors.get('AZI165',0), -0.36, -0.29)}
          </tr>
        </table>

        <!-- Probas -->
        <div style="margin-top:16px;padding:12px;background:#f8f8f8;
                    border:1px solid #e0e0e0;border-radius:6px;font-size:11px">
          <b>Probabilités du modèle ML :</b><br>
          🟢 FAIBLE : {round(proba[0]*100,1)}% &nbsp;|&nbsp;
          🟡 MOYEN  : {round(proba[1]*100,1)}% &nbsp;|&nbsp;
          🔴 CRITIQUE: {round(proba[2]*100,1)}%
        </div>
      </td></tr>

      <!-- Footer -->
      <tr><td style="background:#f5f5f5;padding:14px 32px;border-top:1px solid #e0e0e0;
                     text-align:center;font-size:11px;color:#999">
        Généré automatiquement par <b>SmartPredict v2.0</b> — OCP El Jadida<br>
        Département Informatique · IATE · {date} à {heure}
      </td></tr>

    </table>
  </td></tr>
</table>
</body></html>"""
    return html

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 503
    data = request.get_json()
    try:
        X     = engineer_features(data)
        proba = model.predict_proba(X)[0].tolist()
        label = int(np.argmax(proba))
        return jsonify({
            "risk":       label,
            "risk_name":  CLASS_NAMES[label],
            "proba":      [round(p, 4) for p in proba],
            "timestamp":  datetime.now().isoformat(),
            "sensors":    data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/live")
def live():
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 503
    sensors = simulate_sensor()
    X       = engineer_features(sensors)
    proba   = model.predict_proba(X)[0].tolist()
    label   = int(np.argmax(proba))
    return jsonify({
        "risk":      label,
        "risk_name": CLASS_NAMES[label],
        "proba":     [round(p, 4) for p in proba],
        "timestamp": datetime.now().isoformat(),
        "sensors":   sensors
    })

@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics_cache)

@app.route("/api/feature_importance")
def feature_importance():
    if os.path.exists(FI_PATH):
        df = pd.read_csv(FI_PATH)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])

@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded": model is not None,
        "best_model":   metrics_cache.get("best_model", "—"),
        "uptime":       time.time(),
        "version":      "2.0.0"
    })

# ── Envoi alerte email ────────────────────────────────────────
@app.route("/api/send_alert", methods=["POST"])
def send_alert():
    """Envoie une alerte critique par email (ou email de test)."""
    data = request.get_json()
    email_from = data.get("email_from", "").strip()
    email_pass = data.get("email_pass", "").strip()
    email_to   = data.get("email_to", "").strip()
    sensors    = data.get("sensors", {})
    proba      = data.get("proba", [0.9, 0.05, 0.05])
    is_test    = data.get("test", False)

    if not email_from or not email_pass or not email_to:
        return jsonify({"success": False, "error": "Champs email manquants"}), 400

    try:
        heure = datetime.now().strftime("%H:%M:%S")
        prefix = "[TEST SmartPredict]" if is_test else "🚨 [CRITIQUE]"
        subject = f"{prefix} Alerte Pompe H₂SO₄ — OCP El Jadida — {heure}"

        html_body = build_email_html(sensors, proba, is_test=is_test)

        # Texte fallback
        plain = f"""{'[TEST] ' if is_test else ''}ALERTE CRITIQUE — SmartPredict OCP
Machine : Pompe H2SO4 — Ligne 401 — OCP El Jadida
Heure   : {heure}
Risque critique : {round(proba[2]*100,1)}%

Capteurs :
  TI-171 = {sensors.get('TI171',0)} °C
  TI-170 = {sensors.get('TI170',0)} °C
  TI-169 = {sensors.get('TI169',0)} °C
  VI-167X = {sensors.get('VI167X',0)} mm/s
  VI-167Y = {sensors.get('VI167Y',0)} mm/s
  AZI-165 = {sensors.get('AZI165',0)} rad

Action requise : {'TEST — aucune action' if is_test else 'ARRÊT IMMÉDIAT de la pompe'}
"""
        msg = MIMEMultipart("alternative")
        msg["From"]    = email_from
        msg["To"]      = email_to
        msg["Subject"] = subject
        msg.attach(MIMEText(plain, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html",  "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(email_from, email_pass)
            server.sendmail(email_from, [email_to], msg.as_string())

        print(f"[EMAIL] ✅ {'Test' if is_test else 'Alerte critique'} envoyé → {email_to} à {heure}")
        return jsonify({"success": True, "to": email_to, "timestamp": heure})

    except smtplib.SMTPAuthenticationError:
        return jsonify({"success": False, "error": "Authentification Gmail échouée — vérifie le mot de passe d'application"}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ── Lancement ─────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)