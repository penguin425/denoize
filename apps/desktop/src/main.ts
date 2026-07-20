import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import "./styles.css";

type BackendInfo = { name: string; externalModel: boolean; managedModel: string | null; sampleRate: number | null };
type AppInfo = { version: string; backends: BackendInfo[]; formats: string[]; fdkAvailable: boolean };
type JobProgress = {
  jobId: number; kind: string; status: string; message: string; current: number; total: number;
  fraction: number; elapsedSeconds: number; output?: string; error?: string; etaSeconds?: number;
  item?: string; itemStatus?: "completed" | "failed" | "skipped";
};
type Comparison = {
  markdown: string; json: string; html: string; noisySnrDb: number; enhancedSnrDb: number; improvementDb: number;
};
type ModelRow = {
  name: string; backend: string; license: string; sampleRate: number; revision: string;
  installed: boolean; path: string;
};
type PreviewData = { source: string; playablePath: string; durationSeconds: number; rmsDb: number; waveform: number[] };

const audioFilters = [{ name: "Audio", extensions: ["wav", "flac", "opus", "ogg", "mp3", "m4a", "aac"] }];
let appInfo: AppInfo;
let activeJob: number | null = null;
let comparison: Comparison | null = null;
const previews: { input?: PreviewData; output?: PreviewData } = {};
let activePreview: "input" | "output" = "input";

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
      <header><div><p class="eyebrow">AUDIO RESTORATION</p><h1 id="page-title">ノイズ除去</h1></div><div class="header-actions"><button id="import-config">設定を読込</button><button id="export-config">設定を書出</button><button id="reset-config">初期化</button><div class="header-badge">LOCAL · PRIVATE</div></div></header>

      <section class="page active" id="page-process">
        <div class="grid process-grid">
          <div class="stack">
            <article class="card file-card">
              <div class="card-heading"><div><span class="step">01</span><h2>ファイル</h2></div><span class="hint">WAV · FLAC · OPUS · MP3 · M4A · AAC</span></div>
              <div class="file-row"><div><label>入力</label><div id="input-display" class="path empty">音声ファイルを選択</div></div><button class="secondary" id="choose-input">選択</button></div>
              <div class="file-row"><div><label>出力</label><div id="output-display" class="path empty">保存先を選択</div></div><button class="secondary" id="choose-output">選択</button></div>
              <input type="hidden" id="input-path"><input type="hidden" id="output-path">
              <div id="recent-files" class="recent-files"></div>
            </article>

            <article class="card preview-card">
              <div class="card-heading"><div><span class="step">A/B</span><h2>プレビュー</h2></div><div class="ab-buttons"><button id="preview-input" class="active">処理前</button><button id="preview-output" disabled>処理後</button></div></div>
              <div id="waveform" class="waveform empty"><span>入力ファイルを選ぶと波形を表示します</span></div>
              <audio id="preview-audio" controls preload="metadata"></audio>
              <div class="preview-loop"><label class="toggle inline"><input id="loop-enabled" type="checkbox"><span></span><div><b>区間ループ</b></div></label><label>開始 秒<input id="loop-start" type="number" value="0" min="0" step="0.1"></label><label>終了 秒<input id="loop-end" type="number" value="0" min="0" step="0.1"></label></div>
              <p id="preview-info" class="field-hint">同一位置のまま処理前／処理後を切り替え、RMS音量を揃えて試聴できます。</p>
            </article>

            <article class="card">
              <div class="card-heading"><div><span class="step">02</span><h2>サウンド</h2></div><span class="hint">素材に合わせて調整</span></div>
              <div class="form-grid three">
                <label>モード<select id="mode"><option value="speech">音声</option><option value="music">音楽</option><option value="ambient">環境音</option></select></label>
                <label>プリセット<select id="preset"><option value="hifi">Hi-Fi</option><option value="speech">Speech</option><option value="music">Music</option><option value="gentle">Gentle</option><option value="aggressive">Aggressive</option><option value="restore">Restore</option></select></label>
                <label>バックエンド<select id="backend"><option value="auto">自動</option></select></label>
              </div>
              <div id="backend-settings" class="backend-settings hidden">
                <div class="file-row"><div><label>ONNXモデル</label><div id="model-path-display" class="path empty">モデルファイルを選択</div></div><button class="secondary" id="choose-model">選択</button></div>
                <div class="form-grid two"><label>モデルレート Hz<input id="onnx-rate" type="number" value="16000" min="1"></label><label id="sgmse-profile-field" class="hidden">SGMSE品質<select id="sgmse-profile"><option value="fast">Fast</option><option value="balanced" selected>Balanced</option><option value="quality">Quality</option></select></label></div>
                <input type="hidden" id="model-path"><p id="backend-hint" class="field-hint"></p>
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
              <div class="preset-manager"><label>ユーザープリセット<select id="user-preset"><option value="">プリセットを選択</option></select></label><div><input id="preset-name" placeholder="プリセット名"><button id="save-preset">保存</button><button id="delete-preset">削除</button></div></div>
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
          <article class="card tall"><div class="card-heading"><div><span class="step">01</span><h2>入力</h2></div><div class="button-row"><button class="secondary" id="choose-batch-folder">フォルダ</button><button class="secondary" id="choose-batch">ファイル追加</button></div></div><div id="batch-files" class="empty-panel">フォルダまたは複数ファイルを選択してください</div><div id="batch-results" class="batch-results hidden"></div></article>
          <div class="stack"><article class="card"><div class="card-heading"><div><span class="step">02</span><h2>出力と実行</h2></div></div><div class="file-row"><div><label>出力フォルダ</label><div id="batch-output-display" class="path empty">出力フォルダを選択</div></div><button class="secondary" id="choose-batch-output">選択</button></div><div class="form-grid two"><label>形式<select id="batch-format"><option>wav</option><option>flac</option><option>opus</option><option>mp3</option><option>m4a</option><option>aac</option></select></label><label>並列数<input id="batch-jobs" type="number" value="2" min="1" max="32"></label></div><div class="toggle-grid"><label class="toggle"><input id="batch-recursive" type="checkbox" checked><span></span><div><b>サブフォルダ</b><small>相対構造を維持</small></div></label><label class="toggle"><input id="batch-resume" type="checkbox"><span></span><div><b>中断から再開</b><small>完了済みをスキップ</small></div></label><label class="toggle"><input id="batch-force" type="checkbox"><span></span><div><b>既存を上書き</b><small>出力先を置換</small></div></label></div></article><article class="card action-card"><h3>一括処理</h3><p id="batch-summary">入力が未選択です</p><button class="primary wide" id="start-batch">バッチを開始 <span>→</span></button><button class="danger wide hidden" id="cancel-batch">キャンセル</button></article></div>
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
const SETTINGS_KEY = "denoize.desktop.settings.v1";
const PRESETS_KEY = "denoize.desktop.presets.v1";
const RECENT_KEY = "denoize.desktop.recent.v1";
const settingIds = ["mode", "preset", "backend", "strength", "adaptive", "vad", "metadata", "force", "channels", "mp3-bitrate", "aac-bitrate", "aac-encoder", "loudness-enabled", "loudness", "true-peak", "model-path", "onnx-rate", "sgmse-profile", "batch-format", "batch-jobs", "batch-recursive", "batch-resume", "batch-force"];
type SavedValues = Record<string, string | number | boolean>;

function captureSettings(): SavedValues {
  return Object.fromEntries(settingIds.map((id) => {
    const element = document.getElementById(id) as HTMLInputElement | HTMLSelectElement;
    return [id, element instanceof HTMLInputElement && element.type === "checkbox" ? element.checked : element.value];
  }));
}

function applySettings(values: SavedValues) {
  for (const [id, value] of Object.entries(values)) {
    const element = document.getElementById(id) as HTMLInputElement | HTMLSelectElement | null; if (!element) continue;
    if (element instanceof HTMLInputElement && element.type === "checkbox") element.checked = Boolean(value);
    else element.value = String(value);
  }
  $("#strength-value").textContent = `${Math.round(Number($<HTMLInputElement>("#strength").value) * 100)}%`;
  $("#loudness-fields").classList.toggle("enabled", $<HTMLInputElement>("#loudness-enabled").checked);
  updateBackendSettings(); renderBatch();
}

function saveSettings() { localStorage.setItem(SETTINGS_KEY, JSON.stringify(captureSettings())); }
function restoreSettings() {
  try { const value = localStorage.getItem(SETTINGS_KEY); if (value) applySettings(JSON.parse(value)); } catch { localStorage.removeItem(SETTINGS_KEY); }
  renderPresets(); renderRecentFiles();
}

function presets(): Record<string, SavedValues> {
  try { return JSON.parse(localStorage.getItem(PRESETS_KEY) ?? "{}"); } catch { return {}; }
}
function renderPresets() {
  const selected = $<HTMLSelectElement>("#user-preset").value;
  $("#user-preset").innerHTML = `<option value="">プリセットを選択</option>${Object.keys(presets()).sort().map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("")}`;
  $<HTMLSelectElement>("#user-preset").value = selected;
}
function recentFiles(): string[] { try { return JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]"); } catch { return []; } }
function rememberFile(path: string) {
  localStorage.setItem(RECENT_KEY, JSON.stringify([path, ...recentFiles().filter((item) => item !== path)].slice(0, 6)));
  renderRecentFiles();
}
function renderRecentFiles() {
  const files = recentFiles();
  $("#recent-files").innerHTML = files.length ? `<span>最近:</span>${files.map((path) => `<button data-recent="${escapeHtml(path)}" title="${escapeHtml(path)}">${escapeHtml(path.split(/[\\/]/).pop() ?? path)}</button>`).join("")}` : "";
  document.querySelectorAll<HTMLButtonElement>("[data-recent]").forEach((button) => button.addEventListener("click", async () => {
    const path = button.dataset.recent!; setPath("#input-path", "#input-display", path); setPath("#output-path", "#output-display", await defaultOutput(path)); await preparePreview("input", path);
  }));
}

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
    onnxModel: $<HTMLInputElement>("#model-path").value || null,
    onnxSampleRate: Number($<HTMLInputElement>("#onnx-rate").value),
    sgmseProfile: $<HTMLSelectElement>("#sgmse-profile").value,
  };
}

async function init() {
  appInfo = await invoke<AppInfo>("app_info");
  $("#version").textContent = `v${appInfo.version}`;
  $("#engine-label").textContent = `${appInfo.backends.length} backend${appInfo.backends.length > 1 ? "s" : ""} ready`;
  const backend = $<HTMLSelectElement>("#backend");
  appInfo.backends.forEach(({ name }) => backend.add(new Option(name === "classical" ? "Classical DSP" : name, name)));
  if (appInfo.fdkAvailable) $<HTMLSelectElement>("#aac-encoder").add(new Option("FDK-AAC", "fdk"));
  restoreSettings();
  renderCompareInputs();
  await loadModels();
}

function updateBackendSettings() {
  const selected = $<HTMLSelectElement>("#backend").value;
  const descriptor = appInfo.backends.find(({ name }) => name === selected);
  const needsModel = descriptor?.externalModel ?? false;
  $("#backend-settings").classList.toggle("hidden", !needsModel);
  $("#sgmse-profile-field").classList.toggle("hidden", selected !== "sgmse");
  if (descriptor?.sampleRate) $<HTMLInputElement>("#onnx-rate").value = String(descriptor.sampleRate);
  $("#backend-hint").textContent = selected === "sgmse"
    ? "変換済みSGMSE+モデルと推論ステップ数を指定します。"
    : needsModel ? "このバックエンド用に変換したONNXモデルが必要です。" : "";
}

$("#backend").addEventListener("change", updateBackendSettings);
$("#choose-model").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "ONNX model", extensions: ["onnx"] }] });
  if (typeof path !== "string") return;
  setPath("#model-path", "#model-path-display", path);
});

document.addEventListener("change", (event) => {
  if (settingIds.includes((event.target as HTMLElement).id)) saveSettings();
});
$("#save-preset").addEventListener("click", () => {
  const name = $<HTMLInputElement>("#preset-name").value.trim(); if (!name) return showToast("プリセット名を入力してください", true);
  const values = presets(); values[name] = captureSettings(); localStorage.setItem(PRESETS_KEY, JSON.stringify(values)); renderPresets(); $<HTMLSelectElement>("#user-preset").value = name; showToast("プリセットを保存しました");
});
$("#user-preset").addEventListener("change", (event) => {
  const value = presets()[(event.target as HTMLSelectElement).value]; if (value) { applySettings(value); saveSettings(); }
});
$("#delete-preset").addEventListener("click", () => {
  const name = $<HTMLSelectElement>("#user-preset").value; if (!name) return;
  const values = presets(); delete values[name]; localStorage.setItem(PRESETS_KEY, JSON.stringify(values)); renderPresets();
});
$("#reset-config").addEventListener("click", () => { localStorage.removeItem(SETTINGS_KEY); location.reload(); });

function exportConfig() {
  const values = captureSettings();
  return {
    backend: values.backend, preset: values.preset, mode: values.mode, strength: Number(values.strength),
    adaptive_noise: values.adaptive, vad: values.vad, channels: values.channels,
    loudness_lufs: values["loudness-enabled"] ? Number(values.loudness) : null,
    true_peak_dbtp: Number(values["true-peak"]), preserve_metadata: values.metadata, force: values.force,
    mp3_bitrate_kbps: Number(values["mp3-bitrate"]), m4a_bitrate_kbps: Number(values["aac-bitrate"]),
    aac_encoder: values["aac-encoder"], onnx_model: values["model-path"] || null,
    onnx_rate: Number(values["onnx-rate"]), sgmse_profile: values["sgmse-profile"],
  };
}
$("#export-config").addEventListener("click", async () => {
  const path = await save({ defaultPath: "denoize.toml", filters: [{ name: "TOML", extensions: ["toml"] }] });
  if (path) { await invoke("save_gui_config", { path, config: exportConfig() }); showToast("設定を書き出しました"); }
});
$("#import-config").addEventListener("click", async () => {
  try {
    const path = await open({ multiple: false, filters: [{ name: "TOML", extensions: ["toml"] }] }); if (typeof path !== "string") return;
    const config = await invoke<Record<string, string | number | boolean>>("load_gui_config", { path });
    const map: Record<string, string> = { adaptive_noise: "adaptive", channels: "channels", loudness_lufs: "loudness", true_peak_dbtp: "true-peak", preserve_metadata: "metadata", mp3_bitrate_kbps: "mp3-bitrate", m4a_bitrate_kbps: "aac-bitrate", aac_encoder: "aac-encoder", onnx_model: "model-path", onnx_rate: "onnx-rate", sgmse_profile: "sgmse-profile" };
    const values: SavedValues = {}; for (const [key, value] of Object.entries(config)) values[map[key] ?? key] = value;
    if (config.loudness_lufs != null) values["loudness-enabled"] = true;
    applySettings(values); saveSettings(); showToast("設定を読み込みました");
  } catch (error) { showToast(errorText(error), true); }
});

document.querySelectorAll<HTMLButtonElement>(".nav-item").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".nav-item,.page").forEach((node) => node.classList.remove("active"));
  button.classList.add("active"); $(`#page-${button.dataset.page}`).classList.add("active");
  $("#page-title").textContent = button.textContent?.trim() ?? "denoize";
}));

async function preparePreview(kind: "input" | "output", path: string) {
  try {
    const preview = await invoke<PreviewData>("prepare_preview", { path, points: 180 });
    previews[kind] = preview;
    if (kind === "output") $<HTMLButtonElement>("#preview-output").disabled = false;
    await selectPreview(kind);
  } catch (error) { showToast(`プレビュー: ${errorText(error)}`, true); }
}

async function selectPreview(kind: "input" | "output") {
  const preview = previews[kind]; if (!preview) return;
  const audio = $<HTMLAudioElement>("#preview-audio");
  const position = audio.currentTime || 0; const playing = !audio.paused;
  activePreview = kind;
  document.querySelectorAll(".ab-buttons button").forEach((button) => button.classList.toggle("active", button.id === `preview-${kind}`));
  audio.src = convertFileSrc(preview.playablePath);
  const levels = [previews.input?.rmsDb, previews.output?.rmsDb].filter((value): value is number => value != null);
  const target = levels.length ? Math.min(...levels) : preview.rmsDb;
  audio.volume = Math.min(1, 10 ** ((target - preview.rmsDb) / 20));
  audio.currentTime = Math.min(position, preview.durationSeconds);
  renderWaveform(preview);
  $<HTMLInputElement>("#loop-end").max = String(preview.durationSeconds);
  if (Number($<HTMLInputElement>("#loop-end").value) <= 0) $<HTMLInputElement>("#loop-end").value = preview.durationSeconds.toFixed(1);
  $("#preview-info").textContent = `${kind === "input" ? "処理前" : "処理後"} · ${preview.durationSeconds.toFixed(1)}秒 · RMS ${preview.rmsDb.toFixed(1)} dB`;
  if (playing) await audio.play();
}

function renderWaveform(preview: PreviewData) {
  const waveform = $("#waveform"); waveform.classList.remove("empty");
  waveform.innerHTML = preview.waveform.map((peak) => `<i style="height:${Math.max(2, peak * 100).toFixed(1)}%"></i>`).join("");
}

$("#preview-input").addEventListener("click", () => void selectPreview("input"));
$("#preview-output").addEventListener("click", () => void selectPreview("output"));
$("#waveform").addEventListener("click", (event) => {
  const preview = previews[activePreview]; if (!preview) return;
  const rect = $("#waveform").getBoundingClientRect();
  $<HTMLAudioElement>("#preview-audio").currentTime = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * preview.durationSeconds;
});
$<HTMLAudioElement>("#preview-audio").addEventListener("timeupdate", (event) => {
  if (!$<HTMLInputElement>("#loop-enabled").checked) return;
  const audio = event.currentTarget as HTMLAudioElement;
  const start = Number($<HTMLInputElement>("#loop-start").value), end = Number($<HTMLInputElement>("#loop-end").value);
  if (end > start && audio.currentTime >= end) audio.currentTime = start;
});

$("#choose-input").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: audioFilters }); if (typeof path !== "string") return;
  setPath("#input-path", "#input-display", path);
  const output = await defaultOutput(path); setPath("#output-path", "#output-display", output);
  rememberFile(path);
  previews.output = undefined; $<HTMLButtonElement>("#preview-output").disabled = true;
  await preparePreview("input", path);
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
let batchInputDir = "";
let batchOutput = "";
const batchStatuses = new Map<string, { status: string; error?: string }>();
$("#choose-batch").addEventListener("click", async () => {
  const paths = await open({ multiple: true, filters: audioFilters }); if (!Array.isArray(paths)) return;
  batchInputs = paths; renderBatch();
});
$("#choose-batch-folder").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false }); if (typeof path !== "string") return;
  batchInputDir = path; batchInputs = []; renderBatch();
});
$("#choose-batch-output").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false }); if (typeof path !== "string") return;
  batchOutput = path; $("#batch-output-display").textContent = path; $("#batch-output-display").classList.remove("empty");
});
$("#start-batch").addEventListener("click", async () => {
  try {
    if ((!batchInputs.length && !batchInputDir) || !batchOutput) throw new Error("入力と出力フォルダを選択してください");
    batchStatuses.clear(); $("#batch-results").innerHTML = ""; $("#batch-results").classList.remove("hidden");
    activeJob = await invoke<number>("start_batch", { request: { inputs: batchInputs, inputDir: batchInputDir || null, outputDir: batchOutput, outputFormat: $<HTMLSelectElement>("#batch-format").value, recursive: $<HTMLInputElement>("#batch-recursive").checked, jobs: Number($<HTMLInputElement>("#batch-jobs").value), resume: $<HTMLInputElement>("#batch-resume").checked, options: { ...options(), force: $<HTMLInputElement>("#batch-force").checked } } });
    setJobUi(true, "batch");
  } catch (error) { showToast(errorText(error), true); }
});
$("#cancel-batch").addEventListener("click", () => cancelActive());
function renderBatch() {
  $("#batch-summary").textContent = batchInputDir ? `フォルダを${$<HTMLInputElement>("#batch-recursive").checked ? "再帰的に" : ""}処理します` : `${batchInputs.length}ファイルを処理します`;
  $("#batch-files").innerHTML = batchInputDir ? `<div class="batch-item"><span>DIR</span><div>${escapeHtml(batchInputDir.split(/[\\/]/).pop() ?? batchInputDir)}<small>${escapeHtml(batchInputDir)}</small></div></div>` : batchInputs.map((path, index) => `<div class="batch-item"><span>${String(index + 1).padStart(2, "0")}</span><div>${escapeHtml(path.split(/[\\/]/).pop() ?? path)}<small>${escapeHtml(path)}</small></div></div>`).join("");
  $("#batch-files").classList.toggle("empty-panel", !batchInputDir && !batchInputs.length);
}
$("#batch-recursive").addEventListener("change", renderBatch);

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
  if (payload.kind === "batch" && payload.item && payload.itemStatus) renderBatchResult(payload);
  updateProgress(payload);
  if (["completed", "failed", "cancelled"].includes(payload.status)) {
    if (payload.kind === "file" && payload.status === "completed" && payload.output) void preparePreview("output", payload.output);
    activeJob = null; setJobUi(false, payload.kind); showToast(payload.error ?? payload.message, payload.status === "failed");
  }
});
function updateProgress(progress: JobProgress) {
  const percent = Math.round(progress.fraction * 100);
  $("#progress-percent").textContent = `${percent}%`; $("#progress-message").textContent = progress.message;
  $("#progress-meta").textContent = `${progress.current} / ${progress.total} · ${progress.elapsedSeconds.toFixed(1)}秒${progress.etaSeconds != null ? ` · 残り約${progress.etaSeconds.toFixed(0)}秒` : ""}`;
  $<HTMLElement>("#progress-bar").style.width = `${percent}%`;
  if (progress.kind === "batch") $("#batch-summary").textContent = `${progress.message}  ${progress.current}/${progress.total}`;
}
function renderBatchResult(progress: JobProgress) {
  batchStatuses.set(progress.item!, { status: progress.itemStatus!, error: progress.error });
  const rows = [...batchStatuses.entries()].map(([path, result]) => `<div class="batch-result ${result.status}"><b>${result.status === "completed" ? "完了" : result.status === "skipped" ? "スキップ" : "失敗"}</b><span title="${escapeHtml(path)}">${escapeHtml(path)}${result.error ? ` — ${escapeHtml(result.error)}` : ""}</span></div>`).join("");
  $("#batch-results").innerHTML = rows;
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
