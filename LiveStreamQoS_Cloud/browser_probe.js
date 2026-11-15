// Pemantauan koneksi HP berbasis HTTP ke server website, dijalankan dalam browser.
// BERBEDA dari empat parameter ICMP/adapter Wi-Fi yang dipakai model XGBoost.
// Semua status indikatif Normal/Gangguan dihitung lokal di browser; tidak memakai model XGBoost.
export default function(component) {
  const root = component.parentElement.querySelector('.lsq-mobile');
  if (!root) return;
  const $ = (s) => root.querySelector(s);
  const startButton = $('.lsq-m-start');
  const stopButton = $('.lsq-m-stop');
  const csvButton = $('.lsq-m-csv');
  const status = $('.lsq-m-status');
  const labelNode = $('.lsq-m-label');
  const detailsNode = $('.lsq-m-reason');
  const latencyNode = $('.lsq-m-latency');
  const jitterNode = $('.lsq-m-jitter');
  const failuresNode = $('.lsq-m-failure');
  const downloadNode = $('.lsq-m-download');
  const counter = $('.lsq-m-count');
  const bar = $('.lsq-m-bar-fill');
  const activityNode = $('.lsq-m-activity');
  const probeInfo = $('.lsq-m-probe-info');
  let enabled = false;
  let alive = true;
  let timer = null;
  let busy = false;
  let startTime = 0;
  let lastMeasureStart = 0;
  let seq = 0;
  let samples = [];
  const ctrls = new Set();
  const fmt = (v, d = 1) => Number.isFinite(v) ? v.toFixed(d) : '—';
  const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  const timestamp = () => new Date().toISOString();
  const BASE = '/app/static/';
  // 256KiB per pengukuran; sekitar 3 MiB setiap satu menit jika koneksi normal.
  const PACKET_BYTES = 256 * 1024;

  async function transfer(file, timeoutMs) {
    const abort = new AbortController();
    ctrls.add(abort);
    const timerId = setTimeout(() => abort.abort(), timeoutMs);
    const url = BASE + file + '?trial=' + Date.now() + '_' + Math.random().toString(36).slice(2);
    const t0 = performance.now();
    try {
      const response = await fetch(url, {
        method:'GET', cache:'no-store', credentials:'same-origin', signal:abort.signal
      });
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const buffer = await response.arrayBuffer();
      if (!buffer.byteLength) throw new Error('Respons kosong');
      const elapsed = Math.max(1, performance.now() - t0);
      return {elapsed, bytes:buffer.byteLength};
    } finally {
      clearTimeout(timerId);
      ctrls.delete(abort);
    }
  }

  function classify(arr) {
    if (!arr.length) return {label:'Belum tersedia', reason:'Mulai monitoring untuk mengukur koneksi.'};
    const successful = arr.filter(s => s.http_ok > 0 && s.download_mbps !== null);
    if (!successful.length) {
      return {label:'Tidak dapat dinilai', reason:'Server uji tidak dapat diakses; ini bukan bukti Wi-Fi bermasalah.'};
    }
    const latest = arr[arr.length - 1];
    const goodLat = arr.map(s=>s.http_latency_ms).filter(Number.isFinite);
    const goodJit = arr.map(s=>s.http_jitter_ms).filter(Number.isFinite);
    const goodDown = arr.map(s=>s.download_mbps).filter(Number.isFinite);
    const mean = a => a.length ? a.reduce((x,y)=>x+y,0)/a.length : null;
    const avgLatency=mean(goodLat);
    const avgJitter=mean(goodJit);
    const avgDown=mean(goodDown);
    const failRatio=arr.reduce((s,r)=>s+r.http_failed,0)/(arr.length * 4);
    const reasons=[];
    // Batas eksploratif antarmuka, bukan standar QoS / model ML terlatih.
    if (failRatio >= 0.25) reasons.push('Banyak permintaan HTTP gagal');
    if (avgLatency !== null && avgLatency > 350) reasons.push('Respons server lambat');
    if (avgJitter !== null && avgJitter > 100) reasons.push('Waktu respons tidak stabil');
    if (avgDown !== null && avgDown < 1.5) reasons.push('Kecepatan unduh rendah');
    // Periksa kesalahan terbaru, agar status tidak hanya mengikuti rata-rata lama.
    if (latest.http_failed >= 2 && !reasons.length) reasons.push('Beberapa respons terbaru gagal');
    if (latest.download_mbps !== null && latest.download_mbps < 0.5 && !reasons.length) reasons.push('Unduhan terbaru sangat lambat');
    return {label: reasons.length ? 'Gangguan' : 'Normal',
      reason: reasons.length ? reasons.join(' · ') : 'Pengukuran HTTP tidak melewati ambang indikatif yang digunakan aplikasi.'};
  }

  function draw() {
    const recent = samples.slice(-12);
    const last = recent[recent.length - 1];
    if (counter) counter.textContent = `${recent.length}/12 pengukuran`;
    if (bar) bar.style.width = `${Math.min(100, recent.length/12*100)}%`;
    if (!last) return;
    if (latencyNode) latencyNode.textContent = fmt(last.http_latency_ms, 1) + ' ms';
    if (jitterNode) jitterNode.textContent = fmt(last.http_jitter_ms, 1) + ' ms';
    if (failuresNode) failuresNode.textContent = fmt(last.http_failed / 4 * 100, 0) + '%';
    if (downloadNode) downloadNode.textContent = fmt(last.download_mbps, 2) + ' Mbps';
    const result = classify(recent);
    labelNode.textContent = result.label;
    labelNode.className = 'lsq-m-label ' + (result.label==='Gangguan'?'lsq-m-bad':result.label==='Normal'?'lsq-m-good':'');
    detailsNode.textContent = result.reason;
    if (probeInfo) probeInfo.textContent = 'Pengukuran terakhir: ' + new Date(last.measured_at).toLocaleTimeString();
    if (csvButton) csvButton.disabled = samples.length===0;
  }

  async function measure() {
    if (!enabled || !alive || busy) return;
    busy = true;
    const tick = performance.now();
    startTime = tick;
    if (lastMeasureStart && tick - lastMeasureStart > 16000) {
      samples = [];
      if (status) status.textContent = 'Pengukuran dijeda terlalu lama. Riwayat 12 sampel dimulai ulang.';
    }
    lastMeasureStart = tick;
    seq += 1;
    if (activityNode) activityNode.textContent = 'Mengukur koneksi HP…';
    const times = [];
    let failed = 0;
    let errorMessage = '';
    try {
      for (let i=0;i<4;i++) {
        if (!enabled || !alive) return;
        try {
          const p = await transfer('probe.json', 2500);
          times.push(p.elapsed);
        } catch(e) {
          failed++;
          errorMessage = e?.message || 'Permintaan HTTP gagal';
        }
        if (i !== 3) await delay(90);
      }
      let speed = null;
      if (enabled && alive) {
        try {
          const result = await transfer('probe.bin', 5000);
          if (result.bytes < PACKET_BYTES) throw new Error('Berkas uji tidak lengkap');
          speed = result.bytes * 8 / (result.elapsed/1000) / 1e6;
        } catch(e) {
          errorMessage = e?.message || 'Gagal mengukur kecepatan unduh';
        }
      }
      if (!enabled || !alive) return;
      const avgLat = times.length ? times.reduce((a,b)=>a+b,0)/times.length : null;
      const changes = times.slice(1).map((v,i)=>Math.abs(v-times[i]));
      const jit = changes.length ? changes.reduce((a,b)=>a+b,0)/changes.length : null;
      const row = {
        measured_at: timestamp(), sequence:seq,
        http_latency_ms: avgLat===null?null:Number(avgLat.toFixed(3)),
        http_jitter_ms: jit===null?null:Number(jit.toFixed(3)),
        http_failed:failed, http_ok:times.length,
        download_mbps:speed===null?null:Number(speed.toFixed(4)),
        error: errorMessage
      };
      samples.push(row);
      if (samples.length>500) samples=samples.slice(-500);
      if (status) status.textContent = (times.length===0 && speed===null)
        ? 'Tidak bisa mengakses berkas uji di server. Periksa file static/, konfigurasi .streamlit, atau koneksi internet.'
        : 'Pengukuran HTTP aktif. Status Normal/Gangguan merupakan indikasi browser, bukan prediksi model XGBoost.';
      draw();
    } finally {
      busy=false;
      if (activityNode) activityNode.textContent = enabled ? 'Monitoring aktif' : 'Monitoring berhenti';
      if (enabled && alive) timer = setTimeout(measure, Math.max(350, 5000 - (performance.now()-tick)));
    }
  }

  function stop() {
    enabled=false;
    if (timer!==null) clearTimeout(timer);
    timer=null;
    for (const c of ctrls) c.abort();
    ctrls.clear();
    startButton.disabled=false;
    stopButton.disabled=true;
    if (activityNode) activityNode.textContent='Monitoring berhenti';
    if (status) status.textContent='Monitoring dihentikan. Hasil terakhir tetap tampil di browser.';
  }
  startButton.onclick=() => {
    if (enabled || !alive) return;
    samples=[];
    seq=0;
    lastMeasureStart=0;
    enabled=true;
    startButton.disabled=true;
    stopButton.disabled=false;
    counter.textContent='0/12 pengukuran';
    if(bar) bar.style.width='0%';
    labelNode.textContent='Mengukur…';
    detailsNode.textContent='Status awal muncul setelah pembacaan pertama; rangkuman stabil setelah 12 pembacaan.';
    status.textContent='Memulai pengukuran melalui koneksi browser HP Anda.';
    measure();
  };
  stopButton.onclick=stop;
  csvButton.onclick=() => {
    if(!samples.length)return;
    const cols=['measured_at','sequence','http_latency_ms','http_jitter_ms','http_failed','http_ok','download_mbps','error'];
    const cells = (v)=>'"'+String(v??'').replace(/"/g,'""')+'"';
    const csv='\ufeff'+cols.join(',')+'\n'+samples.map(s=>cols.map(c=>cells(s[c])).join(',')).join('\n');
    const blob = new Blob([csv],{type:'text/csv;charset=utf-8;'});
    const uri=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=uri;a.download='LiveStreamQoS_HTTP_HP.csv';
    document.body.appendChild(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(uri),1000);
  };
  if (startButton) startButton.disabled=false;
  if (stopButton) stopButton.disabled=true;
  if (csvButton) csvButton.disabled=true;
  return ()=>{alive=false;stop();};
}
