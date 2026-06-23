from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Memuat (load) model dan scaler dari file .pkl
# Pastikan file model.pkl dan scaler.pkl ada di folder yang sama dengan app.py
with open('model.pkl', 'rb') as f:
    model_list = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Nama model yang kita simpan sebelumnya di Colab
model_names = ["Decision Tree", "SVC"]

@app.route('/')
def index():
    # Menampilkan halaman web awal
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil pilihan model dari dropdown
    selected_model_name = request.form['model']
    # Memilih model dari list berdasarkan urutan index
    selected_model = model_list[model_names.index(selected_model_name)]

    # Menangkap semua data inputan angka dari form HTML
    data = {
        'Pregnancies': int(request.form['Pregnancies']),
        'Glucose': float(request.form['Glucose']),
        'BloodPressure': float(request.form['BloodPressure']),
        'SkinThickness': float(request.form['SkinThickness']),
        'Insulin': float(request.form['Insulin']),
        'BMI': float(request.form['BMI']),
        'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
        'Age': int(request.form['Age'])
    }

    # Mengubah data menjadi DataFrame Pandas (seperti saat training)
    df_input = pd.DataFrame(data, index=[0])

    # Normalisasi data input menggunakan scaler
    scaled_input = scaler.transform(df_input)

    # Melakukan prediksi
    prediction = selected_model.predict(scaled_input)

    # Menentukan hasil teks
    if prediction[0] == 1:
        hasil_teks = "Pasien diprediksi POSITIF Diabetes."
    else:
        hasil_teks = "Pasien diprediksi NEGATIF Diabetes."

    # Mengembalikan hasil ke halaman web
    return render_template('index.html', model_names=model_names, prediction_text=hasil_teks)

if __name__ == '__main__':
    app.run(debug=True)