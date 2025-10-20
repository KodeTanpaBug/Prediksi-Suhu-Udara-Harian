from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
import logging
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Setup logging to file for easier debugging when running the web app
log_file = os.path.join(os.path.dirname(__file__), 'app_errors.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
app.logger = logging.getLogger('prediksi_suhu_app')
app.logger.setLevel(logging.INFO)

try:
    # Load model dan scaler
    # Workaround for incompatible LSTM config saved with different TF/Keras versions
    # Some saved models contain kwargs like 'time_major' that newer Keras LSTM __init__ doesn't accept.
    # Define a thin wrapper that strips unsupported kwargs during deserialization.
    from tensorflow.keras.layers import LSTM as _KerasLSTM

    class _LSTMWrapper(_KerasLSTM):
        def __init__(self, *args, **kwargs):
            # remove keys that may be present in saved model config but are not accepted by current Keras
            kwargs.pop('time_major', None)
            kwargs.pop('unroll', None)  # safe to pop if older configs used it
            super().__init__(*args, **kwargs)

    try:
        model = load_model('lstm_temperature_model.h5', custom_objects={'LSTM': _LSTMWrapper}, compile=False)
        app.logger.info("Model loaded with LSTM wrapper (compile=False)")
    except Exception as _mod_err:
        # Fallback: try normal load (will likely raise same error) but capture exception
        app.logger.warning("Model load with wrapper failed, retrying default load: %s", _mod_err)
        model = load_model('lstm_temperature_model.h5', compile=False)
    scaler = None
    try:
        scaler = joblib.load('temperature_scaler.save')
    except Exception as _load_err:
        # We'll attempt to fit a new scaler below if needed
        print(f"⚠️ Gagal memuat 'temperature_scaler.save': {_load_err}")
    
    # Load data untuk plotting
    df = pd.read_csv('DailyDelhiClimate.csv')
    
    # Preprocessing data
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Pembersihan data - hapus nilai yang tidak masuk akal
    df = df[(df['meantemp'] >= -10) & (df['meantemp'] <= 50)].copy()
    df = df[(df['humidity'] >= 0) & (df['humidity'] <= 100)].copy()
    df = df[(df['wind_speed'] >= 0) & (df['wind_speed'] <= 50)].copy()
    df = df[(df['meanpressure'] >= 800) & (df['meanpressure'] <= 1100)].copy()
    
    # Features yang digunakan
    features_to_scale = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
    target = 'meantemp'
    
    print(f"✅ Model dan data berhasil dimuat. Dataset: {len(df)} baris")
    print(f"✅ Features: {features_to_scale}")
    # Validasi bahwa scaler sesuai dengan jumlah fitur yang digunakan
    try:
        if scaler is None:
            raise ValueError("scaler is None")

        n_in = getattr(scaler, 'n_features_in_', None)
        # older sklearn versions may not have n_features_in_
        if n_in is None:
            try:
                n_in = scaler.scale_.shape[0]
            except Exception:
                n_in = None

        if n_in != len(features_to_scale):
            print(f"⚠️ Scaler expects {n_in} features but code uses {len(features_to_scale)} features. Re-fitting scaler on current dataset.")
            scaler = MinMaxScaler()
            scaler.fit(df[features_to_scale].values)
            try:
                joblib.dump(scaler, 'temperature_scaler.save')
                print("✅ Re-fitted scaler and saved to 'temperature_scaler.save'.")
            except Exception as _dump_err:
                print(f"⚠️ Gagal menyimpan scaler baru: {_dump_err}")
        else:
            print(f"✅ Scaler valid: expects {n_in} features.")
    except Exception as e_check:
        # As a last resort, fit a new scaler on available data
        print(f"⚠️ Masalah dengan scaler yang dimuat ({e_check}), akan membuat scaler baru dari data saat ini.")
        scaler = MinMaxScaler()
        scaler.fit(df[features_to_scale].values)
        try:
            joblib.dump(scaler, 'temperature_scaler.save')
            print("✅ Dibuat dan disimpan scaler baru ke 'temperature_scaler.save'.")
        except Exception as _dump_err:
            print(f"⚠️ Gagal menyimpan scaler baru: {_dump_err}")
    
except Exception as e:
    app.logger.exception("Error loading model/data: %s", e)
    model = None
    scaler = None
    df = None

def create_sequences_multi_feature(data_array, timesteps=30, target_col_idx=0):
    """Membuat sequence untuk multiple features"""
    X, y = [], []
    for i in range(timesteps, len(data_array)):
        X.append(data_array[i-timesteps:i])
        y.append(data_array[i, target_col_idx])
    return np.array(X), np.array(y)

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/historical_data')
def historical_data():
    """Endpoint untuk menampilkan data historis"""
    try:
        if df is None:
            return jsonify({'error': 'Data tidak tersedia'}), 500
            
        # Plot tren suhu historis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Tren suhu harian
        ax1.plot(df.index, df['meantemp'], label='Data Historis', 
                 alpha=0.8, linewidth=1, color='#2E86AB')
        ax1.set_title('Tren Suhu Udara Harian di Delhi', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Tanggal', fontsize=12)
        ax1.set_ylabel('Suhu Rata-rata (°C)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Analisis musiman
        df_copy = df.copy()
        df_copy['month'] = df_copy.index.month
        monthly_avg = df_copy.groupby('month')['meantemp'].mean()
        monthly_std = df_copy.groupby('month')['meantemp'].std()
        
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        x_pos = range(1, 13)
        
        ax2.plot(x_pos, monthly_avg.values, marker='o', linewidth=3, markersize=8, 
                 color='#F18F01', label='Rata-rata Suhu')
        ax2.fill_between(x_pos, monthly_avg.values - monthly_std.values, 
                         monthly_avg.values + monthly_std.values, alpha=0.3, 
                         color='#F18F01', label='±1 Std Dev')
        ax2.set_title('Analisis Musiman - Pola Suhu Tahunan di Delhi', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Bulan', fontsize=12)
        ax2.set_ylabel('Suhu Rata-rata (°C)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(months)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plot_url = plot_to_base64(fig)
        
        # Split data untuk train/test (80/20)
        split_index = int(len(df) * 0.8)
        train_data = df.iloc[:split_index]
        test_data = df.iloc[split_index:]
        
        return jsonify({
            'plot': plot_url,
            'stats': {
                'period': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
                'avg_temp': f"{df['meantemp'].mean():.2f}°C",
                'max_temp': f"{df['meantemp'].max():.2f}°C",
                'min_temp': f"{df['meantemp'].min():.2f}°C",
                'total_days': len(df),
                'train_period': f"{train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}",
                'test_period': f"{test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}",
                'avg_temp_train': f"{train_data['meantemp'].mean():.2f}°C",
                'avg_temp_test': f"{test_data['meantemp'].mean():.2f}°C"
            }
        })
        
    except Exception as e:
        app.logger.exception("Error in historical_data endpoint: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi suhu"""
    try:
        if model is None or scaler is None or df is None:
            return jsonify({'error': 'Model atau data tidak tersedia'}), 500
            
        # Ambil data terakhir untuk prediksi
        last_data = df[features_to_scale].tail(30).values
        
        # Normalisasi data dengan scaler yang benar (untuk 4 fitur)
        scaled_data = scaler.transform(last_data)
        
        # Reshape untuk prediksi
        X_pred = scaled_data.reshape(1, 30, len(features_to_scale))
        
        # Prediksi
        prediction_scaled = model.predict(X_pred, verbose=0)
        
        # Inverse transform - buat dummy array dengan semua fitur
        dummy_pred = np.zeros((1, len(features_to_scale)))
        dummy_pred[0, 0] = prediction_scaled[0, 0]  # Kolom pertama adalah target (meantemp)
        prediction = scaler.inverse_transform(dummy_pred)[0, 0]
        
        # Plot prediksi
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data historis terakhir
        last_dates = df.index[-30:]
        ax.plot(last_dates, df['meantemp'].tail(30), 
                label='Data Historis', linewidth=2, color='#2E86AB', alpha=0.8)
        
        # Plot prediksi
        next_date = df.index[-1] + pd.Timedelta(days=1)
        ax.plot([df.index[-1], next_date], 
                [df['meantemp'].iloc[-1], prediction], 
                'o-', label='Prediksi', linewidth=3, markersize=8, 
                color='#A23B72')
        
        ax.set_title('Prediksi Suhu Hari Berikutnya', fontsize=16, fontweight='bold')
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Suhu (°C)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_url = plot_to_base64(fig)
        
        return jsonify({
            'prediction': f"{prediction:.2f}°C",
            'plot': plot_url,
            'confidence': 'Model menggunakan 30 hari terakhir untuk prediksi',
            'model_info': {
                'architecture': 'LSTM (50 units) + Dense layers',
                'timesteps': 30,
                'features': features_to_scale,
                'rmse': '1.96°C',
                'mae': '1.56°C'
            }
        })
        
    except Exception as e:
        app.logger.exception("Error in predict endpoint: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/model_performance')
def model_performance():
    """Endpoint untuk menampilkan performa model"""
    try:
        if model is None or scaler is None or df is None:
            return jsonify({'error': 'Model atau data tidak tersedia'}), 500
            
        # Untuk demo, kita akan membuat prediksi pada subset data
        # Ambil 20% terakhir dari data sebagai "test set"
        split_index = int(len(df) * 0.8)
        test_data = df[features_to_scale].iloc[split_index:].values
        test_scaled = scaler.transform(test_data)
        
        # Gabungkan data terakhir dengan test data untuk membuat sequence
        full_scaled = np.concatenate((scaler.transform(df[features_to_scale].iloc[split_index-30:split_index].values), 
                                     test_scaled), axis=0)
        
        X_test_final = []
        for i in range(30, len(full_scaled)):
            X_test_final.append(full_scaled[i-30:i])
        X_test_final = np.array(X_test_final)

        # Prediksi
        pred_test_final = model.predict(X_test_final, verbose=0)
                
        # Inverse transform - buat dummy array dengan semua fitur
        dummy_test_pred = np.zeros((len(pred_test_final), len(features_to_scale)))
        dummy_test_pred[:, 0] = pred_test_final.flatten()  # Kolom pertama adalah target (meantemp)
        pred_test_inv_final = scaler.inverse_transform(dummy_test_pred)[:, 0]
        
        # Data aktual test
        actual_test_final = df[target].iloc[split_index:].values

        # Hitung metrik
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse_test = np.sqrt(mean_squared_error(actual_test_final, pred_test_inv_final))
        mae_test = mean_absolute_error(actual_test_final, pred_test_inv_final)
        r2_test = r2_score(actual_test_final, pred_test_inv_final)
        
        # Plot performa
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Time series comparison
        test_dates = df.index[split_index:]
        ax1.plot(test_dates, actual_test_final, label='Suhu Aktual', 
                 linewidth=2, color='#2E86AB', alpha=0.8)
        ax1.plot(test_dates, pred_test_inv_final, label='Suhu Prediksi', 
                 linewidth=2, color='#A23B72', alpha=0.8)
        ax1.fill_between(test_dates, actual_test_final, pred_test_inv_final, 
                         alpha=0.3, color='gray', label='Error Region')
        ax1.set_title('Perbandingan Prediksi vs Aktual (Test Set)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Tanggal', fontsize=12)
        ax1.set_ylabel('Suhu (°C)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
                
        # Plot 2: Scatter plot
        ax2.scatter(actual_test_final, pred_test_inv_final, alpha=0.6, s=30, 
                    color='#E74C3C', edgecolors='black', linewidth=0.5)
        ax2.plot([actual_test_final.min(), actual_test_final.max()], 
                 [actual_test_final.min(), actual_test_final.max()], 
                 'r--', lw=3, label='Perfect Prediction')
        ax2.set_xlabel('Suhu Aktual (°C)', fontsize=12)
        ax2.set_ylabel('Suhu Prediksi (°C)', fontsize=12)
        ax2.set_title('Scatter Plot: Prediksi vs Aktual', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
                
        # Tambahkan informasi R²
        ax2.text(0.05, 0.95, f'R² = {r2_test:.4f}', transform=ax2.transAxes, 
                 fontsize=12, fontweight='bold', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plot_url = plot_to_base64(fig)
        
        return jsonify({
            'plot': plot_url,
            'metrics': {
                'rmse': f"{rmse_test:.4f}°C",
                'mae': f"{mae_test:.4f}°C",
                'r2': f"{r2_test:.4f}",
                'mape': f"{np.mean(np.abs((actual_test_final - pred_test_inv_final) / actual_test_final)) * 100:.2f}%"
            },
            'model_info': {
                'architecture': 'LSTM (50 units) + Dense layers',
                'timesteps': 30,
                'features': features_to_scale,
                'training_samples': f'{split_index}',
                'test_samples': f'{len(df) - split_index}'
            }
        })
    
    except Exception as e:
        app.logger.exception("Error in model_performance endpoint: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/manual_predict', methods=['POST'])
def manual_predict():
    """Endpoint untuk prediksi dari dataset input manual (file CSV atau teks CSV)."""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model atau scaler tidak tersedia'}), 500
        # Baca data dari file upload atau teks csv
        df_input = None
        # 1) JSON rows (manual form input)
        if request.is_json and isinstance(request.json, dict) and 'rows' in request.json:
            rows = request.json.get('rows') or []
            if not isinstance(rows, list) or len(rows) == 0:
                return jsonify({'error': 'Rows JSON kosong. Kirim minimal 30 baris.'}), 400
            df_input = pd.DataFrame(rows)
        # 2) File upload CSV
        elif 'file' in request.files and request.files['file'] and request.files['file'].filename:
            file = request.files['file']
            df_input = pd.read_csv(file)
        # 3) CSV text pasted
        elif 'csv_text' in request.form and request.form['csv_text'].strip():
            csv_text = request.form['csv_text']
            df_input = pd.read_csv(io.StringIO(csv_text))
        else:
            return jsonify({'error': 'Harap unggah file CSV atau tempel teks CSV.'}), 400

        # Validasi kolom yang diperlukan
        missing = [col for col in features_to_scale if col not in df_input.columns]
        if missing:
            return jsonify({'error': f'Kolom wajib hilang: {", ".join(missing)}. Kolom yang dibutuhkan: {", ".join(features_to_scale)}'}), 400

        # Ambil hanya kolom yang diperlukan dan handle tanggal jika ada
        df_use = df_input.copy()
        # Jika ada kolom date, coba parse untuk plotting
        date_index = None
        if 'date' in df_use.columns:
            try:
                df_use['date'] = pd.to_datetime(df_use['date'])
                date_index = df_use['date']
            except Exception:
                date_index = None

        # Jika kurang dari 30 baris, pad dengan menyalin baris TERAKHIR hingga mencapai 30
        df_use_features = df_use[features_to_scale].reset_index(drop=True)
        if len(df_use_features) < 30:
            pad_count = 30 - len(df_use_features)
            last_row_df = df_use_features.iloc[[-1]]  # DataFrame 1 baris
            pad_df = pd.concat([last_row_df] * pad_count, ignore_index=True)
            df_padded = pd.concat([pad_df, df_use_features], ignore_index=True)
            used_features = df_padded.tail(30)
            padding_info = f"Input kurang dari 30 baris; dipad {pad_count}x dengan baris terakhir."
        else:
            used_features = df_use_features.tail(30)
            padding_info = None

        # Gunakan 30 baris terakhir (setelah padding bila perlu)
        last_30 = used_features.values

        # Fit dan gunakan scaler yang dibuat dari dataset input saat ini
        # Ini memastikan setiap dataset manual akan menghasilkan scaling dan grafik yang konsisten
        local_scaler = MinMaxScaler()
        try:
            # Fit pada dataset original (sebelum padding) agar distribusi tetap merepresentasikan input
            local_scaler.fit(df_use_features.values)
        except Exception:
            # Fallback: fit pada 30 baris yang digunakan
            local_scaler.fit(used_features.values)

        # Normalisasi
        scaled_last_30 = local_scaler.transform(last_30)

        # Bentuk input untuk model
        X_pred = scaled_last_30.reshape(1, 30, len(features_to_scale))

        # Prediksi
        prediction_scaled = model.predict(X_pred, verbose=0)

        # Inverse transform menggunakan local_scaler
        dummy_pred = np.zeros((1, len(features_to_scale)))
        dummy_pred[0, 0] = prediction_scaled[0, 0]  # Kolom pertama adalah target (meantemp)
        prediction = local_scaler.inverse_transform(dummy_pred)[0, 0]

        # Plot prediksi berdasarkan data input manual
        fig, ax = plt.subplots(figsize=(12, 6))
        last_meantemp = used_features['meantemp'].values  # selalu panjang 30 setelah padding
        # Buat sumbu waktu yang sesuai dengan input
        if date_index is not None and isinstance(date_index.iloc[-1], pd.Timestamp):
            # Jika ada minimal 30 tanggal, gunakan 30 tanggal terakhir
            if len(date_index) >= 30:
                last_dates = pd.to_datetime(date_index.tail(30).values)
            else:
                # Jika kurang dari 30, buat rentang tanggal berakhir di tanggal terakhir input
                last_date = pd.to_datetime(date_index.iloc[-1])
                last_dates = pd.date_range(end=last_date, periods=30, freq='D').values

            # Samakan panjang jika terjadi mismatch
            min_len = min(len(last_dates), len(last_meantemp))
            ax.plot(last_dates[:min_len], last_meantemp[:min_len], label='Data Historis (Input)', linewidth=2, color='#2E86AB', alpha=0.8)
            next_date = pd.to_datetime(last_dates[min_len-1]) + pd.Timedelta(days=1)
            ax.plot([last_dates[min_len-1], next_date], [last_meantemp[min_len-1], prediction], 'o-', label='Prediksi', linewidth=3, markersize=8, color='#A23B72')
            ax.set_xlabel('Tanggal', fontsize=12)
        else:
            x_idx = list(range(len(last_meantemp)))
            ax.plot(x_idx, last_meantemp, label='Data Historis (Input)', linewidth=2, color='#2E86AB', alpha=0.8)
            ax.plot([len(last_meantemp)-1, len(last_meantemp)], [last_meantemp[-1], prediction], 'o-', label='Prediksi', linewidth=3, markersize=8, color='#A23B72')
            ax.set_xlabel('Indeks Waktu', fontsize=12)

        ax.set_title('Prediksi Suhu Berdasarkan Dataset Manual (30 hari terakhir)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Suhu (°C)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_url = plot_to_base64(fig)

        result = {
            'prediction': f"{prediction:.2f}°C",
            'plot': plot_url,
            'info': 'Prediksi dihitung dari 30 baris terakhir dataset yang Anda berikan.' ,
            'note': 'Scaling dilakukan berdasarkan dataset yang Anda unggah (scaler lokal).' 
        }
        if padding_info:
            result['padding'] = padding_info
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in manual_predict endpoint: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)