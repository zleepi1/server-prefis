import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Inicializar la aplicación Flask
app = Flask(__name__)
CORS(app)

# Cargar el modelo
try:
    model = tf.keras.models.load_model("modelo_grieta_cuadrada_1.keras")
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None


# Definir el endpoint de predicción
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "El modelo no está disponible."}), 500

    try:
        data = request.get_json(force=True)
        features = data.get("features")

        # Validar que la entrada sea una lista no vacía
        if features is None or not isinstance(features, list) or len(features) == 0:
            return (
                jsonify({"error": 'La clave "features" debe ser una lista no vacía.'}),
                400,
            )

        # --- LÓGICA MEJORADA PARA MANEJAR LOTES (BATCHES) ---
        # Convertimos la lista de entrada a un array de NumPy y la reformateamos.
        # El truco .reshape(-1, 1) funciona para uno o para varios elementos.
        #   - Si features = [5.08], se convierte en [[5.08]].
        #   - Si features = [5.08, 6.10, 7.20], se convierte en [[5.08], [6.10], [7.20]].
        # Esto alinea la entrada perfectamente con lo que el modelo espera: (N_muestras, N_features).

        input_data = np.array(features).reshape(-1, 1)
        predictions = model.predict(input_data)
        output = predictions.tolist()

        # Devolver las predicciones.
        return jsonify({"predictions": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Iniciar el servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
