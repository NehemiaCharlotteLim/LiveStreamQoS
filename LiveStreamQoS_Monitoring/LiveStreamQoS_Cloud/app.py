"""LiveStreamQoS: model penelitian XGBoost, 24 fitur dari 12 pembacaan HTTP."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from inference import ModelSetupError, load_model_bundle, predict_window

st.set_page_config(page_title='LiveStreamQoS | XGBoost Terbaik', page_icon='📶', layout='centered')
HERE = Path(__file__).resolve().parent
monitor = components.declare_component('livestreamqos_browser_best_xgb', path=str(HERE / 'monitor_component'))


@st.cache_resource(show_spinner=False)
def get_bundle(model_mtime: int):
    return load_model_bundle()


try:
    model_file = HERE / 'models' / 'model_bundle_qos_4parameter_CV_TERBAIK.joblib'
    model, bundle = get_bundle(model_file.stat().st_mtime_ns)
    issue = ''
except Exception as exc:
    model, bundle, issue = None, None, f'{type(exc).__name__}: {exc}'

st.title('LiveStreamQoS')
st.caption('Monitoring QoS HTTP di browser HP/laptop dan prediksi menggunakan model XGBoost terbaik penelitian')

home, pred, about = st.tabs(['Home', 'Prediksi QoS', 'About'])
with home:
    st.subheader('Monitoring kualitas jaringan')
    st.write('Pengukuran dilakukan dari browser HP atau laptop, sekitar setiap 5 detik. Sesudah 12 pengukuran valid, aplikasi membentuk 24 fitur statistik lalu menjalankan **Pipeline XGBoost terbaik** dari file model penelitian yang terpasang.')
    st.write('Target keluaran model adalah **indikasi Normal atau Penurunan QoS untuk 60 detik berikutnya**. Hasil bukan pengukuran gangguan pemutaran video yang telah diverifikasi.')
    st.info('Model yang digunakan: XGBoost hasil seleksi CV (4 fold), bukan aturan ambang manual dan bukan model hasil pelatihan ulang.')

with pred:
    st.subheader('Monitoring dan prediksi XGBoost')
    if issue:
        st.error('Model XGBoost belum siap. ' + issue)
    else:
        st.success('XGBoost terbaik berhasil dimuat: 24 fitur, 12 interval riwayat, prediksi QoS 60 detik ke depan.')
    st.session_state.setdefault('xgb_last_answer', None)
    st.session_state.setdefault('xgb_last_request_id', None)
    payload = monitor(
        key='monitor_xgb_24',
        model_ready=model is not None,
        model_issue=issue,
        result=st.session_state.xgb_last_answer,
        default=None,
    )
    if isinstance(payload, dict) and payload.get('type') == 'predict':
        rid = payload.get('request_id')
        if isinstance(rid, str) and 0 < len(rid) <= 128 and rid != st.session_state.xgb_last_request_id:
            st.session_state.xgb_last_request_id = rid
            try:
                if model is None:
                    raise ModelSetupError(issue or 'Model belum tersedia.')
                answer = predict_window(model, bundle, payload.get('samples'))
                st.session_state.xgb_last_answer = {'request_id': rid, 'ok': True, **answer}
            except (ModelSetupError, ValueError, TypeError) as exc:
                st.session_state.xgb_last_answer = {'request_id': rid, 'ok': False, 'error': str(exc)}
            st.rerun()

with about:
    st.subheader('Metode pengukuran dan prediksi')
    st.write('**Data:** latency dan jitter berasal dari durasi respons HTTP, packet loss merupakan persentase permintaan HTTP gagal, throughput merupakan estimasi unduh HTTP. Besaran ini tidak otomatis setara dengan ICMP/packet loss IP atau laju adaptor Wi-Fi.')
    st.write('**Model:** Pipeline `SimpleImputer` + `XGBClassifier` asli dari berkas `.joblib`. Empat parameter QoS ditransformasi menjadi 24 fitur (last, mean, min, max, std, delta), lalu diproses dengan `model.predict()`.')
    st.write('**Label penelitian:** kelas 0 = Normal QoS indikatif, kelas 1 = Penurunan QoS indikatif. Di tampilan, kelas 1 ditulis sebagai GANGGUAN (indikatif). Bundle menyatakan target 12 interval atau sekitar 60 detik ke depan.')
    st.warning('Label pelatihan model bersumber dari `label_biner_qos_indikatif` berbasis aturan QoS empiris, bukan verifikasi gangguan playback di HP. Label keluaran aplikasi tetap berasal dari XGBoost. Kesesuaian data HTTP dengan sumber data latih dan definisi fitur statistik perlu diuji terhadap notebook penelitian.')
    st.caption('Asumsi penyusunan fitur pada versi ini: std sampel (ddof=1), delta = pengukuran terakhir dikurangi pertama. Periksa implementasi ini terhadap kode pembentukan fitur saat training.')
