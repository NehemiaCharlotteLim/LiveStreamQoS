// Komponen v2: kode sesi hanya ada di browser, tidak pernah dikirim ke server Streamlit.
// Sensor Windows dibuka lewat peluncur dan meneruskan kode lewat URL fragment #lsq=<token>.
export default function(component) {
    const {parentElement, setStateValue, state} = component;
    const root = parentElement.querySelector('.lsq-bridge');
    const button = root.querySelector('.lsq-connect');
    const disconnect = root.querySelector('.lsq-disconnect');
    const status = root.querySelector('.lsq-status');
    const storageKey = 'lsq_sensor_token_v2';
    let token = '';
    try {
        const params = new URLSearchParams(window.location.hash.replace(/^#/, ''));
        const newToken = params.get('lsq');
        if (newToken && /^[a-zA-Z0-9_-]{12,128}$/.test(newToken)) {
            token = newToken;
            window.sessionStorage.setItem(storageKey, token);
            // Hilangkan token dari address bar/history setelah disimpan di browser tab.
            window.history.replaceState(window.history.state, '', window.location.pathname + window.location.search);
        } else {
            token = window.sessionStorage.getItem(storageKey) || '';
        }
    } catch (e) { token = ''; }

    let enabled = !!state?.enabled;
    let active = true;
    let timer = null;
    let inFlight = false;
    let lastSeq = Number(state?.snapshot?.sequence ?? -1);
    const autoKey = token ? ('lsq_auto_once_' + token) : '';
    let lastSession = String(state?.snapshot?.session_id ?? '');

    function tell(message, danger=false) {
        if (status) {
            status.textContent = message;
            status.style.color = danger ? '#D56B6B' : 'inherit';
        }
    }
    function paint() {
        if (button) button.disabled = enabled || !token;
        if (disconnect) disconnect.disabled = !enabled;
        if (!token) tell('Sensor belum terpasang/berjalan. Ekstrak ZIP lalu jalankan MULAI_MONITORING.bat. Browser akan membuka halaman ini otomatis.');
    }
    async function read() {
        if (!active || !enabled || !token || inFlight) return;
        inFlight = true;
        try {
            const response = await fetch('http://127.0.0.1:8765/v1/snapshot', {
                method: 'GET', mode: 'cors', cache: 'no-store',
                headers: {'X-LSQ-Token': token},
                targetAddressSpace: 'loopback'
            });
            if (!response.ok) {
                if (response.status === 401) throw new Error('Sesi sensor berubah. Tutup tab lalu jalankan ulang MULAI_MONITORING.bat.');
                if (response.status === 403) throw new Error('Akses browser tidak diizinkan oleh sensor.');
                throw new Error('Sensor merespons dengan status ' + response.status);
            }
            const snapshot = await response.json();
            if (snapshot.schema !== 'lsq_sensor_v1' || !Array.isArray(snapshot.rows)) {
                throw new Error('Format pembacaan sensor tidak sesuai.');
            }
            const n = Number(snapshot.sequence ?? -1);
            const sess = String(snapshot.session_id ?? '');
            tell('Sensor terhubung. Pengukuran terbaru: ' + snapshot.total_available + '/12 interval.');
            if (n !== lastSeq || sess !== lastSession) {
                lastSeq = n;
                lastSession = sess;
                snapshot.browser_received_at = new Date().toISOString();
                setStateValue('snapshot', snapshot);
            }
        } catch(e) {
            enabled = false;
            if (timer !== null) { clearInterval(timer); timer=null; }
            setStateValue('enabled', false);
            setStateValue('snapshot', null);
            paint();
            tell('Koneksi belum berhasil: ' + (e?.message || 'browser menolak akses lokal') + '. Klik Mulai Monitoring untuk mencoba kembali, lalu Izinkan akses lokal jika diminta.', true);
        } finally { inFlight = false; }
    }
    function start() {
        if (!token || !active) return;
        enabled = true;
        if (timer !== null) clearInterval(timer);
        paint();
        tell('Menghubungkan sensor. Izinkan akses jaringan lokal jika diminta browser.');
        read();
        timer = setInterval(read, 5000);
    }
    if (button) button.onclick = () => {
        if (!enabled && token) {
            setStateValue('enabled', true);
            start();
        }
    };
    if (disconnect) disconnect.onclick = () => {
        enabled = false;
        if (timer !== null) {clearInterval(timer); timer=null;}
        setStateValue('enabled', false);
        setStateValue('snapshot', null);
        tell('Tampilan monitoring dihentikan. Untuk menghentikan sensor, tutup jendela CMD.');
        paint();
    };
    paint();
    // Jika dibuka dari MULAI_MONITORING.bat dengan token baru, langsung mulai.
    // Tombol tetap tersedia jika browser meminta gesture/izin secara terpisah.
    if (token && !enabled && window.location.search.includes('monitor=1') && window.sessionStorage.getItem(autoKey) !== '1') {
        window.sessionStorage.setItem(autoKey, '1');
        setStateValue('enabled', true);
        start();
    } else if (enabled && token) {
        start();
    }
    return () => {
        active = false;
        if (timer !== null) clearInterval(timer);
    };
}
