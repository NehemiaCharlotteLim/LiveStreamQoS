"""LiveStreamQoS: uji model XGBoost asli TANPA permintaan HTTP monitoring."""
from pathlib import Path

import pandas as pd
import streamlit as st

from inference import (
    SOURCE_COLUMNS, WINDOW_SIZE, FORECAST_SECONDS,
    ModelSetupError, load_model_bundle, predict_window,
)
from scenarios import create_scenario, to_dataframe

st.set_page_config(page_title='LiveStreamQoS | Uji XGBoost', page_icon='📶', layout='centered')
HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / 'models' / 'model_bundle_qos_4parameter_CV_TERBAIK.joblib'

@st.cache_resource(show_spinner=False)
def get_model(model_mtime_ns: int):
    return load_model_bundle(MODEL_FILE)


def show_result(title: str, answer: dict):
    st.markdown('#### ' + title)
    label = str(answer['label'])
    if label == 'Normal':
        st.success('Hasil prediksi XGBoost: NORMAL (indikatif)')
    else:
        st.error('Hasil prediksi XGBoost: GANGGUAN (indikatif)')
    st.caption('Kelas penelitian: ' + answer['model_label'])
    p = float(answer['probability_qos_decrease'])
    a, b = st.columns(2)
    a.metric('Probabilitas penurunan QoS', f'{100 * p:.1f}%')
    b.metric('Kelas prediksi', str(answer['raw_class']))
    st.progress(min(1.0, max(0.0, p)))
    st.caption('Prediksi model untuk QoS 60 detik berikutnya menurut definisi data latih, '
               'bukan pengukuran WiFi aktual atau verifikasi gangguan streaming.')


def predict_samples(samples):
    if model is None:
        raise ModelSetupError(problem or 'Model belum tersedia.')
    return predict_window(model, bundle, samples)


try:
    model, bundle = get_model(MODEL_FILE.stat().st_mtime_ns)
    problem = ''
except Exception as exc:
    model, bundle, problem = None, None, f'{type(exc).__name__}: {exc}'

st.title('LiveStreamQoS')
st.caption('Uji prediksi Normal / Gangguan menggunakan model XGBoost terbaik penelitian')
st.info('Jenis jaringan: **WiFi**. Ini adalah label pengujian, bukan hasil deteksi adaptor atau pengukuran WiFi otomatis.')
if problem:
    st.error('Model belum dapat dimuat: ' + problem)
else:
    st.success('Model XGBoost asli siap: 12 sampel × 4 parameter menjadi 24 fitur. Tidak menggunakan HTTP untuk pengujian.')

skenario_tab, input_tab, info_tab = st.tabs(['Uji Normal / Gangguan', 'Input Manual / CSV', 'Panduan Penelitian'])

with skenario_tab:
    st.subheader('Pengujian dua kondisi')
    st.write('Dua skenario berisi **data sintetis** untuk 12 interval (masing-masing mewakili 5 detik). '
             'Klik tombol untuk mengetahui prediksi model terbaik. Hasilnya tidak dipaksakan melalui aturan ambang.')
    if st.button('Jalankan Uji Normal dan Gangguan', type='primary', disabled=model is None, key='run_both'):
        try:
            st.session_state['scenario_results'] = {
                'normal': predict_samples(create_scenario('normal')),
                'gangguan': predict_samples(create_scenario('gangguan')),
            }
        except (ModelSetupError, ValueError, TypeError) as exc:
            st.session_state.pop('scenario_results', None)
            st.error('Pengujian gagal: ' + str(exc))
    if 'scenario_results' in st.session_state:
        r = st.session_state['scenario_results']
        show_result('Skenario 1: QoS baik', r['normal'])
        st.divider()
        show_result('Skenario 2: QoS menurun', r['gangguan'])
    with st.expander('Lihat 12 data masukan untuk masing-masing skenario'):
        st.markdown('**Skenario QoS baik**')
        st.dataframe(to_dataframe(create_scenario('normal')), use_container_width=True, hide_index=True)
        st.markdown('**Skenario QoS menurun**')
        st.dataframe(to_dataframe(create_scenario('gangguan')), use_container_width=True, hide_index=True)

with input_tab:
    st.subheader('Prediksi dengan data yang Anda tentukan')
    st.write('Masukkan **12 sampel** berurutan dari paling lama ke paling baru. '
             'Dapat diedit langsung atau diimpor dari pengukuran jaringan eksternal melalui CSV. '
             'Aplikasi tidak mengukur jaringan secara otomatis.')
    data_mode = st.radio('Sumber data', ['Edit 12 sampel', 'Impor CSV'], horizontal=True)
    samples = None
    if data_mode == 'Edit 12 sampel':
        edited = st.data_editor(
            to_dataframe(create_scenario('normal')),
            key='qos_manual_editor', num_rows='fixed', use_container_width=True, hide_index=True,
            column_config={
                'latency_ms': st.column_config.NumberColumn('Latency (ms)', min_value=0.0, format='%.2f'),
                'jitter_ms': st.column_config.NumberColumn('Jitter (ms)', min_value=0.0, format='%.2f'),
                'packet_loss_percent': st.column_config.NumberColumn('Packet loss (%)', min_value=0.0, max_value=100.0, format='%.2f'),
                'throughput_mbps': st.column_config.NumberColumn('Throughput (Mbps)', min_value=0.001, format='%.3f'),
            },
        )
        samples = edited.to_dict(orient='records')
    else:
        uploaded = st.file_uploader('Unggah CSV berisi empat kolom QoS', type=['csv'])
        st.download_button('Unduh contoh CSV 12 interval',
                           data=to_dataframe(create_scenario('normal')).to_csv(index=False).encode('utf-8'),
                           file_name='contoh_data_qos_12_interval.csv', mime='text/csv')
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded, nrows=10001)
                if not all(c in raw.columns for c in SOURCE_COLUMNS):
                    raise ValueError('CSV harus memiliki kolom: ' + ', '.join(SOURCE_COLUMNS))
                if len(raw) < WINDOW_SIZE:
                    raise ValueError('CSV perlu sedikitnya 12 baris data.')
                if len(raw) > 10000:
                    raise ValueError('CSV maksimal 10.000 baris untuk pengujian ini.')
                df = raw.loc[:, list(SOURCE_COLUMNS)].tail(WINDOW_SIZE).copy()
                for column in SOURCE_COLUMNS:
                    df[column] = pd.to_numeric(df[column], errors='raise')
                samples = df.to_dict(orient='records')
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption('Model menggunakan 12 baris terakhir dalam CSV.')
            except (ValueError, KeyError, pd.errors.ParserError, UnicodeError) as exc:
                st.error('CSV tidak valid: ' + str(exc))
    if st.button('Prediksi Menggunakan XGBoost', type='primary', disabled=model is None or samples is None, key='run_manual'):
        try:
            st.session_state['manual_result'] = predict_samples(samples)
            st.session_state['manual_source'] = data_mode
        except (ModelSetupError, ValueError, TypeError) as exc:
            st.session_state.pop('manual_result', None)
            st.error('Data tidak valid atau prediksi gagal: ' + str(exc))
    if 'manual_result' in st.session_state:
        st.caption('Hasil terakhir (' + st.session_state.get('manual_source', 'input manual') + '). '
                   'Klik Prediksi lagi setelah mengubah data.')
        show_result('Hasil input pengguna', st.session_state['manual_result'])

with info_tab:
    st.subheader('Cara kerja model')
    st.write('**Model:** berkas `.joblib` asli, Pipeline `SimpleImputer` dan `XGBClassifier` yang dipilih sebagai model terbaik penelitian.')
    st.write('**Input:** 12 pengamatan dari empat parameter: latency, jitter, packet loss, throughput. '
             'Sistem menghitung `last`, `mean`, `min`, `max`, `std`, `delta` sehingga terbentuk 24 fitur, '
             'kemudian memanggil `model.predict()` dan `model.predict_proba()`.')
    st.write('**Output:** kelas 0 = Normal QoS indikatif; kelas 1 = Penurunan QoS indikatif, '
             'yang ditampilkan sebagai Gangguan. Model ditujukan untuk horizon 60 detik.')
    st.warning('**Batasan skripsi:** Skenario bawaan merupakan data sintetis untuk menguji jalur inferensi, '
               'bukan bukti gangguan jaringan atau tayangan streaming. Label training merupakan proxy QoS empiris. '
               'Rumus std (ddof=1) dan delta (terakhir - pertama) dalam integrasi perlu dibandingkan '
               'dengan notebook pelatihan sebelum mengklaim validasi akhir.')
    st.write('**Tanpa HTTP:** versi ini tidak memakai `fetch()`, komponen browser pengukur HTTP, '
             'ataupun file probe. Pengukuran jaringan nyata memerlukan perangkat atau aplikasi pengukur '
             'eksternal; hasilnya bisa dimasukkan melalui CSV.')
    st.caption('Label WiFi pada halaman adalah konfigurasi tampilan yang diminta, '
               'bukan deteksi otomatis koneksi yang sedang aktif.')
