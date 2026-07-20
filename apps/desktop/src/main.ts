import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import "./styles.css";

type AppInfo = { version: string; backends: string[]; formats: string[]; fdkAvailable: boolean };
type JobProgress = {
  jobId: number; kind: string; status: string; message: string; current: number; total: number;
  fraction: number; elapsedSeconds: number; output?: string; error?: string;
};
type Comparison = {
  markdown: string; json: string; html: string; noisySnrDb: number; enhancedSnrDb: number; improvementDb: number;
};
type ModelRow = {
  name: string; backend: string; license: string; sampleRate: number; revision: string;
  installed: boolean; path: string;
};

const audioFilters = [{ name: "Audio", extensions: ["wav", "flac", "opus", "ogg", "mp3", "m4a", "aac"] }];
let appInfo: AppInfo;
let activeJob: number | null = null;
let comparison: Comparison | null = null;

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <div class="shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-mark"><span></span><span></span><span></span></div><div><strong>denoize</strong><small>studio</small></div></div>
      <nav>
        <button class="nav-item active" data-page="process"><span>◈</span>ノイズ除去</button>
        <button class="nav-item" data-page="batch"><span>▦</span>バッチ</button>
        <button class="nav-item" data-page="compare"><span>◒</span>品質比較</button>
        <button class="nav-item" data-page="models"><span>⬡</span>モデル</button>
      </nav>
      <div class="sidebar-foot"><span class="status-dot"></span><span id="engine-label">エンジンを確認中</span><small id="version"></small></div>
    </aside>
    <main>
      <header><div><p class="eyebrow">AUDIO RESTORATION</p><h1 id="page-title">ノイズ除去</h1></div><div class="header-badge">LOCAL · PRIVATE</div></header>

      <section class="page active" id="page-process">
        <div class="grid process-grid">
          <div class="stack">
            <article class="card file-card">
              <div class="card-heading"><div><span class="step">01</span><h2>ファイル</h2></div><span class="hint">WAV · FLAC · OPUS · MP3 · M4A · AAC</span></div>
              <div class="file-row"><div><label>入力</label><div id="input-display" class="path empty">音声ファイルを選択</div></div><button class="secondary" id="choose-input">選択</button></div>
              <div class="file-row"><div><label>出力</label><div id="output-display" class="path empty">保存先を選択</div></div><button class="secondary" id="choose-output">選択</button></div>
              <input type="hidden" id="input-path"><input type="hidden" id="output-path">
            </article>

            <article class="card">
              <div class="card-heading"><div><span class="step">02</span><h2>サウンド</h2></div><span class="hint">素材に合わせて調整</span></div>
              <div class="form-grid three">
                <label>モード<select id="mode"><option value="speech">音声</option><option value="music">音楽</option><option value="ambient">環境音</option></select></label>
                <label>プリセット<select id="preset"><option value="hifi">Hi-Fi</option><option value="speech">Speech</option><option value="music">Music</option><option value="gentle">Gentle</option><option value="aggressive">Aggressive</option><option value="restore">Restore</option></select></label>
                <label>バックエンド<select id="backend"><option value="auto">自動</option></select></label>
              </div>
              <div class="strength-row"><div><label>除去強度</label><span id="strength-value">40%</span></div><input id="strength" type="range" min="0" max="1" step="0.01" value="0.4"><div class="range-labels"><span>自然</span><span>強力</span></div></div>
              <div class="toggle-grid">
                <label class="toggle"><input id="adaptive" type="checkbox"><span></span><div><b>適応ノイズ追従</b><small>変化する環境ノイズを学習</small></div></label>
                <label class="toggle"><input id="vad" type="checkbox"><span></span><div><b>音声区間検出</b><small>無音区間の処理を最適化</small></div></label>
                <label class="toggle"><input id="metadata" type="checkbox" checked><span></span><div><b>メタデータ保持</b><small>タグとアートワークをコピー</small></div></label>
                <label class="toggle"><input id="force" type="checkbox"><span></span><div><b>上書きを許可</b><small>既存の出力を置換</small></div></label>
              </div>
            </article>
          </div>

          <div class="stack side-stack">
            <article class="card compact">
              <div class="card-heading"><div><span class="step">03</span><h2>出力</h2></div></div>
              <label>ステレオ処理<select id="channels"><option value="independent">独立</option><option value="linked" selected>ステレオリンク</option><option value="mid-side">Mid / Side</option></select></label>
              <div class="form-grid two"><label>MP3 kbps<input id="mp3-bitrate" type="number" value="192" min="32"></label><label>AAC kbps<input id="aac-bitrate" type="number" value="192" min="32"></label></div>
              <label>AACエンコーダー<select id="aac-encoder"><option value="oxide">OxideAV</option></select></label>
              <label class="toggle inline"><input id="loudness-enabled" type="checkbox"><span></span><div><b>ラウドネス正規化</b></div></label>
              <div class="form-grid two muted-fields" id="loudness-fields"><label>目標 LUFS<input id="loudness" type="number" value="-16" step="0.5"></label><label>True Peak<input id="true-peak" type="number" value="-1" step="0.1"></label></div>
            </article>
            <article class="card action-card">
              <div id="idle-state"><div class="ready-icon">◎</div><h3>準備ができたら開始</h3><p>処理はすべてこのコンピューター内で行われます。</p></div>
              <div id="job-state" class="hidden"><div class="progress-ring"><span id="progress-percent">0%</span></div><h3 id="progress-message">処理中</h3><p id="progress-meta"></p><div class="progress-track"><i id="progress-bar"></i></div></div>
              <button class="primary wide" id="start-process">ノイズ除去を開始 <span>→</span></button>
              <button class="danger wide hidden" id="cancel-process">キャンセル</button>
            </article>
          </div>
        </div>
      </section>

      <section class="page" id="page-batch">
        <div class="grid two-col">
          <article class="card tall"><div class="card-heading"><div><span class="step">01</span><h2>入力ファイル</h2></div><button class="secondary" id="choose-batch">ファイルを追加</button></div><div id="batch-files" class="empty-panel">複数の音声ファイルを選択してください</div></article>
          <div class="stack"><article class="card"><div class="card-heading"><div><span class="step">02</span><h2>出力先</h2></div></div><div class="file-row"><div><label>フォルダ</label><div id="batch-output-display" class="path empty">出力フォルダを選択</div></div><button class="secondary" id="choose-batch-output">選択</button></div><label>形式<select id="batch-format"><option>wav</option><option>flac</option><option>opus</option><option>mp3</option><option>m4a</option><option>aac</option></select></label></article><article class="card action-card"><h3>一括処理</h3><p id="batch-summary">ファイルが未選択です</p><button class="primary wide" id="start-batch">バッチを開始 <span>→</span></button><button class="danger wide hidden" id="cancel-batch">キャンセル</button></article></div>
        </div>
      </section>

      <section class="page" id="page-compare">
        <div class="compare-layout">
          <article class="card"><div class="card-heading"><div><span class="step">01</span><h2>参照ファイル</h2></div></div><div id="compare-inputs" class="compare-inputs"></div><button class="primary wide" id="run-compare">品質を比較</button></article>
          <article class="card result-card"><div class="card-heading"><div><span class="step">02</span><h2>結果</h2></div><button class="secondary hidden" id="export-report">HTMLを保存</button></div><div id="compare-empty" class="empty-panel">3つのファイルを選ぶと、改善量を可視化できます</div><div id="compare-result" class="hidden"><div class="metric-hero"><span>改善</span><strong id="improvement">+0.00 dB</strong></div><div class="metric-pair"><div><span>処理前 SNR</span><b id="noisy-snr">0</b></div><div><span>処理後 SNR</span><b id="enhanced-snr">0</b></div></div><pre id="report-markdown"></pre></div></article>
        </div>
      </section>

      <section class="page" id="page-models">
        <article class="card"><div class="card-heading"><div><span class="step">AI</span><h2>モデルライブラリ</h2></div><button class="secondary" id="refresh-models">更新</button></div><p class="section-copy">外部モデルはチェックサム検証後、ローカルキャッシュに保存されます。</p><div id="model-list" class="model-list"><div class="empty-panel">モデル情報を読み込んでいます</div></div></article>
      </section>
      <div id="toast" role="status"></div>
    </main>
  </div>`;

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const setPath = (input: string, display: string, path: string | null) => {
  const field = $<HTMLInputElement>(input); const view = $(display);
  field.value = path ?? ""; view.textContent = path ?? "選択されていません"; view.classList.toggle("empty", !path);
};
const showToast = (message: string, error = false) => {
  const toast = $("#toast"); toast.textContent = message; toast.className = error ? "show error" : "show";
  window.setTimeout(() => toast.className = "", 4200);
};
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

function options() {
  return {
    backend: $<HTMLSelectElement>("#backend").value,
    preset: $<HTMLSelectElement>("#preset").value,
    mode: $<HTMLSelectElement>("#mode").value,
    strength: Number($<HTMLInputElement>("#strength").value),
    adaptiveNoise: $<HTMLInputElement>("#adaptive").checked,
    vad: $<HTMLInputElement>("#vad").checked,
    channelMode: $<HTMLSelectElement>("#channels").value,
    loudnessLufs: $<HTMLInputElement>("#loudness-enabled").checked ? Number($<HTMLInputElement>("#loudness").value) : null,
    truePeakDbtp: Number($<HTMLInputElement>("#true-peak").value),
    preserveMetadata: $<HTMLInputElement>("#metadata").checked,
    force: $<HTMLInputElement>("#force").checked,
    mp3BitrateKbps: Number($<HTMLInputElement>("#mp3-bitrate").value),
    aacBitrateKbps: Number($<HTMLInputElement>("#aac-bitrate").value),
    aacEncoder: $<HTMLSelectElement>("#aac-encoder").value,
  };
}

async function init() {
  appInfo = await invoke<AppInfo>("app_info");
  $("#version").textContent = `v${appInfo.version}`;
  $("#engine-label").textContent = `${appInfo.backends.length} backend${appInfo.backends.length > 1 ? "s" : ""} ready`;
  const backend = $<HTMLSelectElement>("#backend");
  appInfo.backends.forEach((name) => backend.add(new Option(name === "classical" ? "Classical DSP" : name, name)));
  if (appInfo.fdkAvailable) $<HTMLSelectElement>("#aac-encoder").add(new Option("FDK-AAC", "fdk"));
  renderCompareInputs();
  await loadModels();
}

document.querySelectorAll<HTMLButtonElement>(".nav-item").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".nav-item,.page").forEach((node) => node.classList.remove("active"));
  button.classList.add("active"); $(`#page-${button.dataset.page}`).classList.add("active");
  $("#page-title").textContent = button.textContent?.trim() ?? "denoize";
}));

$("#choose-input").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: audioFilters }); if (typeof path !== "string") return;
  setPath("#input-path", "#input-display", path);
  const output = await defaultOutput(path); setPath("#output-path", "#output-display", output);
});
$("#choose-output").addEventListener("click", async () => {
  const path = await save({ filters: audioFilters, defaultPath: $<HTMLInputElement>("#output-path").value || undefined });
  if (path) setPath("#output-path", "#output-display", path);
});
async function defaultOutput(input: string) {
  const dot = input.lastIndexOf("."); const separator = Math.max(input.lastIndexOf("/"), input.lastIndexOf("\\"));
  const base = dot > separator ? input.slice(0, dot) : input;
  return `${base}.denoized.wav`;
}

$("#strength").addEventListener("input", (event) => $("#strength-value").textContent = `${Math.round(Number((event.target as HTMLInputElement).value) * 100)}%`);
$("#loudness-enabled").addEventListener("change", (event) => $("#loudness-fields").classList.toggle("enabled", (event.target as HTMLInputElement).checked));

$("#start-process").addEventListener("click", async () => {
  try {
    const input = $<HTMLInputElement>("#input-path").value, output = $<HTMLInputElement>("#output-path").value;
    if (!input || !output) throw new Error("入力と出力を選択してください");
    activeJob = await invoke<number>("start_process", { request: { input, output, options: options() } });
    setJobUi(true, "process");
  } catch (error) { showToast(errorText(error), true); }
});
$("#cancel-process").addEventListener("click", () => cancelActive());

let batchInputs: string[] = [];
let batchOutput = "";
$("#choose-batch").addEventListener("click", async () => {
  const paths = await open({ multiple: true, filters: audioFilters }); if (!Array.isArray(paths)) return;
  batchInputs = paths; renderBatch();
});
$("#choose-batch-output").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false }); if (typeof path !== "string") return;
  batchOutput = path; $("#batch-output-display").textContent = path; $("#batch-output-display").classList.remove("empty");
});
$("#start-batch").addEventListener("click", async () => {
  try {
    if (!batchInputs.length || !batchOutput) throw new Error("入力ファイルと出力フォルダを選択してください");
    activeJob = await invoke<number>("start_batch", { request: { inputs: batchInputs, outputDir: batchOutput, outputFormat: $<HTMLSelectElement>("#batch-format").value, options: options() } });
    setJobUi(true, "batch");
  } catch (error) { showToast(errorText(error), true); }
});
$("#cancel-batch").addEventListener("click", () => cancelActive());
function renderBatch() {
  $("#batch-summary").textContent = `${batchInputs.length}ファイルを処理します`;
  $("#batch-files").innerHTML = batchInputs.map((path, index) => `<div class="batch-item"><span>${String(index + 1).padStart(2, "0")}</span><div>${escapeHtml(path.split(/[\\/]/).pop() ?? path)}<small>${escapeHtml(path)}</small></div></div>`).join("");
}

const comparePaths: Record<string, string> = { clean: "", noisy: "", enhanced: "" };
function renderCompareInputs() {
  const labels: Record<string, string> = { clean: "クリーン参照", noisy: "処理前", enhanced: "処理後" };
  $("#compare-inputs").innerHTML = Object.entries(labels).map(([key, label]) => `<button class="compare-file" data-compare="${key}"><span>${label}</span><b>${comparePaths[key] ? escapeHtml(comparePaths[key].split(/[\\/]/).pop() ?? "") : "ファイルを選択"}</b><small>${comparePaths[key] ? escapeHtml(comparePaths[key]) : "クリックして選択"}</small></button>`).join("");
  document.querySelectorAll<HTMLButtonElement>("[data-compare]").forEach((button) => button.addEventListener("click", async () => {
    const path = await open({ multiple: false, filters: audioFilters }); if (typeof path !== "string") return;
    comparePaths[button.dataset.compare!] = path; renderCompareInputs();
  }));
}
$("#run-compare").addEventListener("click", async () => {
  try {
    if (Object.values(comparePaths).some((value) => !value)) throw new Error("3つの比較ファイルを選択してください");
    $("#run-compare").setAttribute("disabled", "true");
    comparison = await invoke<Comparison>("compare_audio", comparePaths);
    $("#compare-empty").classList.add("hidden"); $("#compare-result").classList.remove("hidden"); $("#export-report").classList.remove("hidden");
    $("#improvement").textContent = `${comparison.improvementDb >= 0 ? "+" : ""}${comparison.improvementDb.toFixed(2)} dB`;
    $("#noisy-snr").textContent = `${comparison.noisySnrDb.toFixed(2)} dB`; $("#enhanced-snr").textContent = `${comparison.enhancedSnrDb.toFixed(2)} dB`;
    $("#report-markdown").textContent = comparison.markdown;
  } catch (error) { showToast(errorText(error), true); } finally { $("#run-compare").removeAttribute("disabled"); }
});
$("#export-report").addEventListener("click", async () => {
  if (!comparison) return; const path = await save({ defaultPath: "denoize-comparison.html", filters: [{ name: "HTML", extensions: ["html"] }] });
  if (path) { await invoke("save_text_file", { path, contents: comparison.html }); showToast("レポートを保存しました"); }
});

async function loadModels() {
  try {
    const models = await invoke<ModelRow[]>("list_models");
    $("#model-list").innerHTML = models.map((model) => `<div class="model-row"><div class="model-icon">AI</div><div class="model-info"><div><b>${escapeHtml(model.name)}</b><span class="pill ${model.installed ? "installed" : ""}">${model.installed ? "インストール済み" : "未導入"}</span></div><p>${escapeHtml(model.backend)} · ${model.sampleRate.toLocaleString()} Hz · ${escapeHtml(model.license)}</p><small>${escapeHtml(model.path)}</small></div><div class="model-actions">${model.installed ? `<button data-model="${model.name}" data-action="verify">検証</button><button data-model="${model.name}" data-action="update">更新</button><button class="remove" data-model="${model.name}" data-action="remove">削除</button>` : `<button class="install" data-model="${model.name}" data-action="install">導入</button>`}</div></div>`).join("");
    document.querySelectorAll<HTMLButtonElement>("[data-model]").forEach((button) => button.addEventListener("click", async () => {
      try { button.disabled = true; const result = await invoke<string>("model_action", { name: button.dataset.model, action: button.dataset.action }); showToast(result); await loadModels(); }
      catch (error) { showToast(errorText(error), true); } finally { button.disabled = false; }
    }));
  } catch (error) { $("#model-list").textContent = errorText(error); }
}
$("#refresh-models").addEventListener("click", loadModels);

listen<JobProgress>("job-progress", ({ payload }) => {
  if (payload.jobId !== activeJob) return;
  updateProgress(payload);
  if (["completed", "failed", "cancelled"].includes(payload.status)) {
    activeJob = null; setJobUi(false, payload.kind); showToast(payload.error ?? payload.message, payload.status === "failed");
  }
});
function updateProgress(progress: JobProgress) {
  const percent = Math.round(progress.fraction * 100);
  $("#progress-percent").textContent = `${percent}%`; $("#progress-message").textContent = progress.message;
  $("#progress-meta").textContent = `${progress.current} / ${progress.total} · ${progress.elapsedSeconds.toFixed(1)}秒`;
  $<HTMLElement>("#progress-bar").style.width = `${percent}%`;
  if (progress.kind === "batch") $("#batch-summary").textContent = `${progress.message}  ${progress.current}/${progress.total}`;
}
function setJobUi(running: boolean, kind: string) {
  if (kind === "file" || kind === "process") {
    $("#idle-state").classList.toggle("hidden", running); $("#job-state").classList.toggle("hidden", !running);
    $("#start-process").classList.toggle("hidden", running); $("#cancel-process").classList.toggle("hidden", !running);
  } else { $("#start-batch").classList.toggle("hidden", running); $("#cancel-batch").classList.toggle("hidden", !running); }
}
async function cancelActive() { if (activeJob !== null) try { await invoke("cancel_job", { jobId: activeJob }); } catch (error) { showToast(errorText(error), true); } }
function escapeHtml(value: string) { return value.replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]!); }

init().catch((error) => showToast(errorText(error), true));
