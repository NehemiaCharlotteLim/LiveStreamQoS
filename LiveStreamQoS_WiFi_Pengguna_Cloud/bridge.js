// Streamlit custom component v2. Runs in the participant's browser, NOT on cloud.
// Data reaches Python only after an explicit click on "Hubungkan Sensor".
export default function(component) {
    const { parentElement, setStateValue, data, state } = component;
    const root = parentElement.querySelector('.lsq-bridge');
    const button = root.querySelector('.lsq-connect');
    const disconnect = root.querySelector('.lsq-disconnect');
    const status = root.querySelector('.lsq-status');
    const token = String(data?.token ?? '').trim();
    let enabled = !!state?.enabled;
    let active = true;
    let timer = null;
    let lastSeq = Number(state?.snapshot?.sequence ?? -1);
    let lastSession = String(state?.snapshot?.session_id ?? '');
    let inFlight = false;

    function statusText(msg, danger=false) {
        if (status) { status.textContent = msg; status.style.color = danger ? '#D56B6B' : 'inherit'; }
    }
    function paint() {
        if (button) button.disabled = enabled || token.length < 8;
        if (disconnect) disconnect.disabled = !enabled;
    }
    async function read() {
        if (!enabled || inFlight || !active) return;
        inFlight = true;
        try {
            // Chrome/Edge may require explicit loopback network permission.
            const response = await fetch('http://127.0.0.1:8765/v1/snapshot', {
                method: 'GET', mode: 'cors', cache: 'no-store',
                headers: {'X-LSQ-Token': token},
                targetAddressSpace: 'loopback'
            });
            if (!response.ok) {
                if (response.status === 401) throw new Error('Kode penghubung tidak sesuai.');
                if (response.status === 403) throw new Error('Origin website belum diizinkan oleh sensor.');
                throw new Error('Sensor merespons dengan status ' + response.status);
            }
            const snapshot = await response.json();
            if (snapshot.schema !== 'lsq_sensor_v1' || !Array.isArray(snapshot.rows)) {
                throw new Error('Format pembacaan sensor tidak sesuai.');
            }
            const n = Number(snapshot.sequence ?? -1);
            const sess = String(snapshot.session_id ?? '');
            statusText('Terhubung ke sensor Wi-Fi laptop. Sampel: ' + snapshot.total_available + '/12.');
            if (n !== lastSeq || sess !== lastSession) {
                lastSeq = n;
                lastSession = sess;
                snapshot.browser_received_at = new Date().toISOString();
                setStateValue('snapshot', snapshot);
            }
        } catch (e) {
            statusText('Belum tersambung: ' + (e?.message || 'browser menolak koneksi lokal') + '. Pastikan sensor aktif dan izin akses lokal diizinkan.', true);
            if (state?.snapshot) setStateValue('snapshot', null);
        } finally { inFlight = false; }
    }
    function start() {
        if (!token) return;
        enabled = true;
        paint();
        statusText('Menghubungkan sensor... Jika browser meminta izin akses lokal, klik Izinkan.');
        if (timer !== null) clearInterval(timer);
        read();
        timer = setInterval(read, 5000);
    }
    if (button) button.onclick = () => {
        if (!enabled && token.length >= 8) {
            setStateValue('enabled', true);
            start();
        }
    };
    if (disconnect) disconnect.onclick = () => {
        enabled = false;
        if (timer !== null) { clearInterval(timer); timer = null; }
        setStateValue('enabled', false);
        setStateValue('snapshot', null);
        statusText('Koneksi website ke sensor dihentikan. Anda bisa menutup CMD sensor.');
        paint();
    };
    paint();
    if (enabled && token) start();
    return () => {
        active = false;
        if (timer !== null) clearInterval(timer);
    };
}
