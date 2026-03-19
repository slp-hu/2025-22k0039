import numpy as np
import soundfile as sf
import sounddevice as sd
import pygame
import threading
import time
import math
import tkinter as tk
from tkinter import filedialog
import scipy.special
import scipy.signal

# ---------------- SETTINGS ----------------
WAV_PATH = "band_20s.wav"
#WAV_PATH = "260123_017.wav"

SR_TARGET = None
BLOCK_SIZE = 1024   # hop
N_FFT = 2048        # 2*hop recommended
OUTPUT_CHANNELS = 2

WIN_W, WIN_H = 1460, 760

MOVE_STEP = math.radians(3.0)
LONGPRESS_DELAY = 0.6

MAP_LEFT = 30
MAP_TOP = 78
MAP_WIDTH = 820
MAP_HEIGHT = 520

DB_MIN = -60.0
DB_MAX = 10.0

DEFAULT_CH_DB = 0
DEFAULT_MASTER_DB = -10

# Crossfade length (ms) for clickless toggles
XFADE_MS = 50.0

# Mix auto-compensation: user requested not to use => fixed OFF
AUTO_COMPENSATE = False

# Meter
PEAK_HOLD_SEC = 2.0
METER_FLOOR_DB = -60.0
METER_GREEN_DB = -18.0
METER_ORANGE_DB = -6.0

# Limiter
LIMIT_THRESH = 0.95
ATTACK = 0.15
RELEASE = 0.02
SOFTCLIP_AFTER = True

# --- 追加: MWF切替時、先頭だけMWF OFFを使う時間（秒） ---
MWF_START_BYPASS_SEC = 0.5

# === Added: 高域補償(HF)のデフォルト状態 ===
USE_HF_COMP = True
# ==========================================

# --- sources/directions (MWF.py reference) ---

#250803_cut.wav
#バンド録音

SOURCES = [
    {"name": "Vo", "theta": math.radians(294.894), "phi": math.radians(4.909)},
    {"name": "Bass", "theta": math.radians(180.383), "phi": math.radians(-0.409)},
    {"name": "Drums", "theta": math.radians(271.532), "phi": math.radians(21.273)},
    {"name": "Guitar", "theta": math.radians(54.000), "phi": math.radians(-6.545)},
]
"""
# ambi_mix.wav (simulation)

SOURCES = [
    {"name": "VOCAL",  "theta":  2.6947528307118316,  "phi": (math.pi/2) - 1.4670462948741323},
    {"name": "DRUMS",  "theta": -1.8118784234730931,  "phi": (math.pi/2) - 1.4398219185213423},
    {"name": "BASE",   "theta":  0.4815752872053996,  "phi": (math.pi/2) - 0.928283063920418},
    {"name": "PIANO",  "theta": -2.7222635743040753,  "phi": (math.pi/2) - 1.9377506306465673},
    {"name": "GUITAR", "theta": -1.0314138809847597,  "phi": (math.pi/2) - 2.018550372683015},
]

SOURCES = [
    {"name": "Guitar2", "theta": math.radians(0.766), "phi": math.radians(5.727)},
    {"name": "Guitar1", "theta": math.radians(114.894), "phi": math.radians(-15.136)},
    {"name": "Drums", "theta": math.radians(265.021), "phi": math.radians(-0.409)},
    {"name": "Vocals", "theta": math.radians(160.085), "phi": math.radians(0.409)},
    {"name": "Key", "theta": math.radians(77.362), "phi": math.radians(15.136)},
    {"name": "Bass", "theta": math.radians(203.362), "phi": math.radians(-4.500)},
]
"""
# ---------------- Utility: fader mapping ----------------
def db_to_fader_pos(db):
    db = float(db)
    pos = (db - DB_MIN) / (DB_MAX - DB_MIN)
    return max(0.0, min(1.0, pos))

def fader_pos_to_db(pos):
    pos = max(0.0, min(1.0, pos))
    if pos <= 0.01:
        return float("-inf")
    return DB_MIN + pos * (DB_MAX - DB_MIN)

def fader_pos_to_gain(pos):
    pos = max(0.0, min(1.0, pos))
    if pos <= 0.0:
        return 0.0
    db = DB_MIN + pos * (DB_MAX - DB_MIN)
    return 10.0 ** (db / 20.0)

DEFAULT_CH_POS = db_to_fader_pos(DEFAULT_CH_DB)
DEFAULT_MASTER_POS = db_to_fader_pos(DEFAULT_MASTER_DB)

# ---------------- LOAD WAV ----------------
data, sr = sf.read(WAV_PATH, always_2d=True)
if data.shape[1] != 4:
    raise RuntimeError("WAV must be 4 channels (AmbiX: W,Y,Z,X)")
data = data.astype(np.float32, copy=False)

if SR_TARGET is None:
    SR_TARGET = sr
if SR_TARGET != sr:
    print("[WARN] SR_TARGET != file sr. This script does NOT resample. Using file sr.")
    SR_TARGET = sr

n_frames = data.shape[0]
playback_done = False
stop_playback = False
paused = True
read_ptr = 0

# ---------------- Beam/mixer states ----------------
beam_states = []
for src in SOURCES:
    beam_states.append({
        "name": src["name"],
        "base_theta": src["theta"],
        "base_phi": src["phi"],
        "theta": src["theta"],
        "phi": src["phi"],
        "fader_pos": DEFAULT_CH_POS,
        "pan": 0.0,
        "mute": False,
        "solo": False,
    })

selected_index = 0
master_fader_pos = DEFAULT_MASTER_POS
state_lock = threading.Lock()

# ---------------- Meter shared ----------------
meter_lock = threading.Lock()

meter_peak_db = -120.0
meter_rms_db  = -120.0
meter_L_db    = -120.0
meter_R_db    = -120.0

meter_peak_hold_db = -120.0
meter_L_hold_db = -120.0
meter_R_hold_db = -120.0
_meter_peak_hold_t = 0.0
_meter_L_hold_t = 0.0
_meter_R_hold_t = 0.0

# Smooth state
meter_pk_smooth = -120.0
meter_rms_smooth = -120.0
meter_L_smooth = -120.0
meter_R_smooth = -120.0
_meter_last_t = None

def _safe_db(x, floor=-120.0):
    x = float(x)
    if x <= 0.0 or not np.isfinite(x):
        return floor
    return max(floor, min(0.0, 20.0 * math.log10(x)))

# ---------------- UI/Progress shared ----------------
precompute_lock = threading.Lock()
precompute_in_progress = False
precompute_progress = 0.0     # 0..1
precompute_phase = "idle"
precompute_status = "未計算"
precompute_error = None
cache_dirty = False
cache_version = 0

# 再生進捗（0..1）
playback_progress = 0.0

# ---------------- SH (AmbiX ACN+SN3D) ----------------
SH_NORM = "sn3d"

def eval_sh(max_order, dirs_sph, norm=SH_NORM):
    dirs_sph = np.array(dirs_sph, dtype=np.float64)
    if dirs_sph.ndim == 1:
        dirs_sph = dirs_sph.reshape((1, 2))

    num_sh_channels = (max_order + 1) ** 2
    num_dir = dirs_sph.shape[0]
    azi = dirs_sph[:, 0]
    ele = dirs_sph[:, 1]

    Y = np.zeros((num_dir, num_sh_channels), dtype=np.float64)
    colat = (np.pi / 2.0) - ele

    for n in range(0, max_order + 1):
        for m in range(-n, n + 1):
            acn = n * n + n + m

            if m < 0:
                Yazi = np.sqrt(2.0) * np.sin(azi * abs(m))
            elif m > 0:
                Yazi = np.sqrt(2.0) * np.cos(azi * m)
            else:
                Yazi = np.ones(num_dir, dtype=np.float64)

            Yzen = np.zeros(num_dir, dtype=np.float64)
            for iDir in range(num_dir):
                Yzen[iDir] = ((-1) ** m) * scipy.special.lpmv(abs(m), n, np.cos(colat[iDir]))

            base = math.factorial(n - abs(m)) / (math.factorial(n + abs(m)) + 1e-30)
            norm_sn3d = math.sqrt(base)
            if str(norm).lower() == "n3d":
                norm_factor = norm_sn3d * math.sqrt((2.0 * n + 1.0) / (4.0 * np.pi))
            else:
                norm_factor = norm_sn3d

            Y[:, acn] = Yazi * Yzen * norm_factor

    return Y

def expand_weights(weights_per_order):
    ambi_order = weights_per_order.shape[0] - 1
    num_sh_channels = (ambi_order + 1) ** 2
    weights = np.zeros(num_sh_channels, dtype=np.float64)
    i = 0
    for n in range(0, ambi_order + 1):
        num_ch = 2*n + 1
        weights[i:i+num_ch] = weights_per_order[n]
        i += num_ch
    return weights

def getMaxReWeights(order, maxOrder=None):
    table = [
        np.array([1.0]),
        np.array([0.3659487, 0.2113504]),
        np.array([0.18752139, 0.14536802, 0.07527491]),
        np.array([0.11367906, 0.09794028, 0.06973125, 0.03483484]),
        np.array([0.0761984, 0.0690694, 0.0558118, 0.03827066, 0.01884882])
    ]
    if maxOrder is None:
        maxOrder = order
    if order == maxOrder:
        return table[order] / 0.079
    elif order <= maxOrder:
        return np.hstack((table[order], np.zeros(maxOrder - order))) / 0.079
    else:
        return table[maxOrder] / 0.079

def _normalize_vec(v, eps=1e-12):
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < eps:
        return v
    return v / n

# Beam weights (W,Y,Z,X)
def weight_basic(theta, phi):
    c = 1.0 / math.sqrt(3.0)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    w = np.array([1.0, c*(sin_theta*cos_phi), c*(sin_phi), c*(cos_theta*cos_phi)], dtype=np.float64)
    return _normalize_vec(w)

def weight_maxdi(theta, phi):
    y = eval_sh(1, np.array([[theta, phi]], dtype=np.float64)).reshape(-1)
    return _normalize_vec(y)

def weight_maxre(theta, phi):
    ambi_order = 1
    weights = expand_weights(getMaxReWeights(ambi_order)).astype(np.float64)
    y = eval_sh(ambi_order, np.array([[theta, phi]], dtype=np.float64)).reshape(-1)
    return _normalize_vec(weights * y)

def get_weight_for_mode(theta, phi, beam_mode):
    if beam_mode == 0:
        return weight_basic(theta, phi)
    elif beam_mode == 1:
        return weight_maxre(theta, phi)
    else:
        return weight_maxdi(theta, phi)

# ---------------- Beam mode gains (kept) ----------------
BEAM_MODE = 2
MODE_GAINS = [1.0, 1.0, 1.0]

def calibrate_mode_gains():
    global MODE_GAINS
    N = min(int(SR_TARGET * 2.0), n_frames)
    if N <= 2048:
        return
    calib_block = data[:N, :].astype(np.float32, copy=False)

    dirs = [(0.0, 0.0), (math.radians(90), 0.0), (math.radians(180), 0.0), (math.radians(270), 0.0)]

    def beam_apply(block, th, ph, mode):
        w = get_weight_for_mode(th, ph, mode).astype(np.float64)
        return (block.astype(np.float64) @ w).astype(np.float64)

    monos_basic, monos_re, monos_di = [], [], []
    for th0, ph0 in dirs:
        monos_basic.append(beam_apply(calib_block, th0, ph0, 0))
        monos_re.append(beam_apply(calib_block, th0, ph0, 1))
        monos_di.append(beam_apply(calib_block, th0, ph0, 2))

    def rms(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.sqrt(np.mean(x*x) + 1e-12))

    rms_basic = float(np.mean([rms(x) for x in monos_basic]))
    rms_re    = float(np.mean([rms(x) for x in monos_re]))
    rms_di    = float(np.mean([rms(x) for x in monos_di]))

    target = rms_re if rms_re > 1e-9 else 1.0
    g_basic = target / rms_basic if rms_basic > 1e-9 else 1.0
    g_di    = target / rms_di    if rms_di > 1e-9 else 1.0

    def cap(g, lo=0.25, hi=4.0):
        if not np.isfinite(g):
            return 1.0
        return float(max(lo, min(hi, g)))

    MODE_GAINS[0] = cap(g_basic)
    MODE_GAINS[1] = 1.0
    MODE_GAINS[2] = cap(g_di)

calibrate_mode_gains()

# ---------------- Limiter ----------------
limiter_gain = 1.0

def apply_limiter_stereo(L, R):
    global limiter_gain
    peak = float(max(np.max(np.abs(L)), np.max(np.abs(R))) + 1e-12)
    desired = 1.0
    if peak > LIMIT_THRESH:
        desired = LIMIT_THRESH / peak

    if desired < limiter_gain:
        limiter_gain = limiter_gain * (1.0 - ATTACK) + desired * ATTACK
    else:
        limiter_gain = limiter_gain * (1.0 - RELEASE) + desired * RELEASE

    limiter_gain = float(max(0.0, min(1.0, limiter_gain)))
    return L * limiter_gain, R * limiter_gain

def softclip(x):
    return np.tanh(x)

# ---------------- Angles / time ----------------
def clamp_angles(t, p):
    t = t % (2 * np.pi)
    p = max(-np.pi / 2, min(np.pi / 2, p))
    return t, p

def format_time(sec):
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

# ====================== MWF ENGINE ======================
def hann_window(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)

class MultiMWF:
    def __init__(self, sr, hop, nfft, nsrc):
        self.sr = int(sr)
        self.hop = int(hop)
        self.nfft = int(nfft)
        self.nsrc = int(nsrc)

        self.use_mwf = True
        self.loudness_match = True

        self.alpha = 0.01
        self.diag_load = 3e-3
        self.p = 9

        self.win = hann_window(self.nfft).astype(np.float64)
        self.win2 = self.win * self.win
        self.cola_eps = 1e-8

        self.inbuf = np.zeros((self.nfft, 4), dtype=np.float64)
        self.F = self.nfft // 2 + 1
        self.out_ola = np.zeros((self.nsrc, self.nfft), dtype=np.float64)
        self.cola = np.zeros((self.nfft,), dtype=np.float64)

        self.Rs = np.zeros((self.nsrc, self.F, 4, 4), dtype=np.complex128)
        self.Rn = np.zeros((self.nsrc, self.F, 4, 4), dtype=np.complex128)
        self.I4 = np.eye(4, dtype=np.complex128)[None, :, :]
        self.e_ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

        # loudness match per source
        self.rms_ref_ema = np.full((self.nsrc,), 1e-4, dtype=np.float64)
        self.rms_out_ema = np.full((self.nsrc,), 1e-4, dtype=np.float64)
        self.gain_match  = np.ones((self.nsrc,), dtype=np.float64)
        self.rms_alpha = 0.06
        self.gain_alpha = 0.08
        self.gain_min = 0.25
        self.gain_max = 4.0

        self.last_mask_mean = np.zeros((self.nsrc,), dtype=np.float64)

    def reset(self):
        self.inbuf.fill(0.0)
        self.out_ola.fill(0.0)
        self.cola.fill(0.0)
        self.Rs.fill(0.0)
        self.Rn.fill(0.0)
        self.reset_gain_tracking()
        self.last_mask_mean.fill(0.0)

    def reset_gain_tracking(self):
        self.rms_ref_ema.fill(1e-4)
        self.rms_out_ema.fill(1e-4)
        self.gain_match.fill(1.0)

    def _stft(self, frame_wyzx):
        xw = frame_wyzx * self.win[:, None]
        return np.fft.rfft(xw, axis=0)  # (F,4)

    def _istft(self, Y):
        y = np.fft.irfft(Y, n=self.nfft)
        return y * self.win

    def process_hop(self, hop_block_wyzx, Wsrc):
        hop_block_wyzx = hop_block_wyzx.astype(np.float64, copy=False)
        Wsrc_c = Wsrc.astype(np.complex128, copy=False)

        self.inbuf[:-self.hop, :] = self.inbuf[self.hop:, :]
        self.inbuf[-self.hop:, :] = hop_block_wyzx

        X = self._stft(self.inbuf)  # (F,4)

        Ybeams = X @ (Wsrc_c.conj().T)  # (F,K)
        mag = np.abs(Ybeams) + 1e-12
        p = float(self.p)
        magp = mag ** p
        denom = np.sum(magp, axis=1, keepdims=True) + 1e-12
        masks = magp / denom
        self.last_mask_mean = np.mean(masks, axis=0)

        outer = X[:, :, None] * np.conjugate(X[:, None, :])  # (F,4,4)
        a = float(self.alpha)

        mk = masks.T[:, :, None, None]  # (K,F,1,1)
        outerK = outer[None, :, :, :]   # (1,F,4,4)
        self.Rs = (1.0 - a) * self.Rs + a * (mk * outerK)
        self.Rn = (1.0 - a) * self.Rn + a * ((1.0 - mk) * outerK)

        self.cola[:-self.hop] = self.cola[self.hop:]
        self.cola[-self.hop:] = 0.0
        self.cola += self.win2
        nrm = self.cola[:self.hop].copy()

        mono_hops = np.zeros((self.nsrc, self.hop), dtype=np.float64)
        eps = 1e-8

        for k in range(self.nsrc):
            Y_ref = X @ np.conjugate(Wsrc_c[k])
            y_ref_frame = self._istft(Y_ref)
            y_ref_hop = y_ref_frame[:self.hop] / (nrm + self.cola_eps)

            if self.use_mwf:
                Rx = (self.Rs[k] + self.Rn[k]) + (float(self.diag_load) * self.I4)
                v = self.Rs[k] @ self.e_ref
                w = np.linalg.solve(Rx, v[..., None])[..., 0]
                Y_out = np.einsum("fi,fi->f", np.conjugate(w), X)
            else:
                Y_out = Y_ref

            y_out_frame = self._istft(Y_out)

            self.out_ola[k, :-self.hop] = self.out_ola[k, self.hop:]
            self.out_ola[k, -self.hop:] = 0.0
            self.out_ola[k] += y_out_frame

            y_hop = self.out_ola[k, :self.hop].copy()
            y_hop = y_hop / (nrm + self.cola_eps)

            rms_ref = float(np.sqrt(np.mean(y_ref_hop * y_ref_hop) + eps))
            rms_out = float(np.sqrt(np.mean(y_hop * y_hop) + eps))

            ra = float(self.rms_alpha)
            self.rms_ref_ema[k] = (1.0 - ra) * self.rms_ref_ema[k] + ra * rms_ref
            self.rms_out_ema[k] = (1.0 - ra) * self.rms_out_ema[k] + ra * rms_out

            target_gain = self.rms_ref_ema[k] / max(eps, self.rms_out_ema[k])
            target_gain = max(self.gain_min, min(self.gain_max, target_gain))

            ga = float(self.gain_alpha)
            self.gain_match[k] = (1.0 - ga) * self.gain_match[k] + ga * target_gain

            if self.loudness_match:
                mono_hops[k] = y_hop * self.gain_match[k]
            else:
                mono_hops[k] = y_hop

        return mono_hops

# 設定保持用（リアルタイム処理には使わず、UI状態を持つ）
mwf_engine = MultiMWF(sr=SR_TARGET, hop=BLOCK_SIZE, nfft=N_FFT, nsrc=len(SOURCES))
mwf_reset_requested = False  # この版ではリアルタイムMWF計算しないため実質未使用

# ---------------- Precomputed source caches ----------------
# shape: (K, n_frames)
precomp_sources_off = None
precomp_sources_on = None
precomp_sources_hf = None  # 高域補償成分のキャッシュ
bypass_gain_adj = np.ones(len(SOURCES), dtype=np.float32)  # === Added: 先頭バイパス用の音量補正ゲイン ===

def _snapshot_extract_settings():
    with state_lock:
        params = [(b["theta"], b["phi"]) for b in beam_states]
        beam_mode = int(BEAM_MODE)
        lm_on = bool(mwf_engine.loudness_match)
        alpha = float(mwf_engine.alpha)
        diag_load = float(mwf_engine.diag_load)
        p = float(mwf_engine.p)
    return params, beam_mode, lm_on, alpha, diag_load, p

def _set_precompute_status(in_progress=None, progress=None, phase=None, status=None, error=None):
    global precompute_in_progress, precompute_progress, precompute_phase, precompute_status, precompute_error
    with precompute_lock:
        if in_progress is not None:
            precompute_in_progress = bool(in_progress)
        if progress is not None:
            precompute_progress = float(max(0.0, min(1.0, progress)))
        if phase is not None:
            precompute_phase = str(phase)
        if status is not None:
            precompute_status = str(status)
        if error is not None:
            precompute_error = error

def build_precomputed_sources_blocking():
    """
    実行前に、MWF OFF / ON の各ソースmonoを全区間計算して保持する。
    （フェーダー/PAN/mute/soloは再生時に適用）
    """
    global precomp_sources_off, precomp_sources_on, precomp_sources_hf, bypass_gain_adj, cache_dirty, cache_version

    params, beam_mode, lm_on, alpha, diag_load, p = _snapshot_extract_settings()
    K = len(params)
    total_blocks = (n_frames + BLOCK_SIZE - 1) // BLOCK_SIZE
    if total_blocks <= 0:
        raise RuntimeError("Audio frames are empty.")

    _set_precompute_status(in_progress=True, progress=0.0, phase="prepare", status="前計算を開始します…", error=None)
    print("[Precompute] Start building source caches (MWF OFF / ON)...")

    Wsrc = np.stack([get_weight_for_mode(th, ph, beam_mode) for (th, ph) in params], axis=0).astype(np.float64)

    eng_off = MultiMWF(sr=SR_TARGET, hop=BLOCK_SIZE, nfft=N_FFT, nsrc=K)
    eng_on  = MultiMWF(sr=SR_TARGET, hop=BLOCK_SIZE, nfft=N_FFT, nsrc=K)

    # 共通パラメータ
    for eng in (eng_off, eng_on):
        eng.loudness_match = lm_on
        eng.alpha = alpha
        eng.diag_load = diag_load
        eng.p = p
        eng.reset()

    eng_off.use_mwf = False
    eng_on.use_mwf  = True

    out_off = np.zeros((K, n_frames), dtype=np.float32)
    out_on  = np.zeros((K, n_frames), dtype=np.float32)

    t0 = time.time()
    last_print = -1

    for bi in range(total_blocks):
        ptr = bi * BLOCK_SIZE
        endp = min(n_frames, ptr + BLOCK_SIZE)

        hop = data[ptr:endp, :].astype(np.float32, copy=False)
        valid = hop.shape[0]
        if valid < BLOCK_SIZE:
            pad = np.zeros((BLOCK_SIZE - valid, 4), dtype=np.float32)
            hop = np.vstack([hop, pad])

        # OFF / ON を別エンジンで計算して保持
        mono_off = eng_off.process_hop(hop, Wsrc).astype(np.float32, copy=False)
        mono_on  = eng_on.process_hop(hop, Wsrc).astype(np.float32, copy=False)

        out_off[:, ptr:endp] = mono_off[:, :valid]
        out_on[:, ptr:endp]  = mono_on[:, :valid]

        prog = (bi + 1) / total_blocks
        _set_precompute_status(progress=prog * 0.9, phase="compute",
                               status=f"MWF計算中… {prog*100:.1f}% ({bi+1}/{total_blocks} blocks)")

        p10 = int(prog * 100)
        if p10 != last_print and (p10 % 5 == 0 or p10 == 100):
            last_print = p10
            print(f"[Precompute] MWF {prog*100:.1f}%")

    # STFTによる高域補償用トラックの事前一括生成
    _set_precompute_status(progress=0.95, phase="compute", status="高域補償成分を抽出中…")
    print("[Precompute] Extracting HF compensation tracks via full-STFT...")
    
    # オーバーラップアドを用いてノイズのない完璧な高域抽出を行う
    f_stft, t_stft, Zxx = scipy.signal.stft(out_off, fs=SR_TARGET, nperseg=N_FFT, noverlap=N_FFT-BLOCK_SIZE)
    
    mask = np.zeros_like(f_stft, dtype=np.float32)
    f_low = 8000.0
    f_high = 12000.0
    for i, freq in enumerate(f_stft):
        if freq <= f_low:
            mask[i] = 0.0
        elif freq >= f_high:
            mask[i] = 1.0
        else:
            ratio = (freq - f_low) / (f_high - f_low)
            mask[i] = 0.5 - 0.5 * math.cos(math.pi * ratio)
            
    Zxx_hf = Zxx * mask[None, :, None]
    _, out_hf = scipy.signal.istft(Zxx_hf, fs=SR_TARGET, nperseg=N_FFT, noverlap=N_FFT-BLOCK_SIZE)
    
    # 尺を揃える
    if out_hf.shape[1] > n_frames:
        out_hf = out_hf[:, :n_frames]
    elif out_hf.shape[1] < n_frames:
        pad = np.zeros((K, n_frames - out_hf.shape[1]), dtype=np.float32)
        out_hf = np.hstack([out_hf, pad])
        
    precomp_sources_hf = out_hf.astype(np.float32)

    # === Added: 先頭バイパス領域の音量差を計測し、自動補正するゲインを計算 ===
    bypass_samples = int(MWF_START_BYPASS_SEC * SR_TARGET)
    eval_len = int(0.5 * SR_TARGET)  # バイパス直後の0.5秒間を計測して音量マッチング
    for k in range(K):
        end_eval = min(n_frames, bypass_samples + eval_len)
        if end_eval > bypass_samples:
            rms_ref = float(np.sqrt(np.mean(out_off[k, bypass_samples:end_eval]**2) + 1e-12))
            target_eval = out_on[k, bypass_samples:end_eval] + out_hf[k, bypass_samples:end_eval]
            rms_on  = float(np.sqrt(np.mean(target_eval**2) + 1e-12))
            # SHの音量をMWFに合わせるための係数
            bypass_gain_adj[k] = max(0.1, min(10.0, rms_on / rms_ref))
        else:
            bypass_gain_adj[k] = 1.0
    print(f"[Precompute] Bypass Gain Adjustments: {bypass_gain_adj}")
    # ====================================================================

    precomp_sources_off = out_off
    precomp_sources_on = out_on
    cache_dirty = False
    cache_version += 1

    dt = time.time() - t0
    _set_precompute_status(in_progress=False, progress=1.0, phase="done",
                           status=f"前計算完了（{dt:.1f}s）", error=None)
    print(f"[Precompute] Done in {dt:.1f}s")

def rebuild_precomputed_sources_with_pause():
    """
    UIからの設定変更後に手動/自動で再計算。
    再生中のガタつきを避けるため一時停止して同期実行する。
    """
    global paused
    was_paused = paused
    paused = True
    try:
        build_precomputed_sources_blocking()
        request_clickless_update(reset_mwf=False)
    except Exception as e:
        _set_precompute_status(in_progress=False, phase="error", status="前計算エラー", error=str(e))
        print("[Precompute][ERROR]", e)
    finally:
        paused = was_paused

# 起動時に前計算（ユーザー要望: 実行前に保持）
build_precomputed_sources_blocking()

# ---------------- Clickless crossfade state ----------------
xfade_lock = threading.Lock()
xfade_request = False
xfade_remaining = 0
xfade_total = int(max(1, (SR_TARGET * (XFADE_MS / 1000.0)) // BLOCK_SIZE) * BLOCK_SIZE)
_prev_outL = np.zeros((BLOCK_SIZE,), dtype=np.float32)
_prev_outR = np.zeros((BLOCK_SIZE,), dtype=np.float32)

def request_clickless_update(reset_mwf=True):
    global xfade_request
    # reset_mwf 引数は互換目的で残す
    with xfade_lock:
        xfade_request = True

def toggle_play_pause():
    global paused, playback_done, read_ptr
    if playback_done and read_ptr >= n_frames:
        read_ptr = 0
        playback_done = False
        request_clickless_update(reset_mwf=False)
    paused = not paused
    request_clickless_update(reset_mwf=False)

# ---------------- AUDIO CALLBACK ----------------
def _get_cached_source_block(ptr, frames, use_mwf, use_hf):
    """
    ptr..ptr+frames の Kxframes mono block を返す。
    use_mwf=True のときも、先頭0.5秒は MWF OFF を強制使用。
    さらに音量補正とクロスフェード処理を行い、自然に繋げる。
    """
    global precomp_sources_off, precomp_sources_on, precomp_sources_hf, bypass_gain_adj

    K = len(SOURCES)
    out = np.zeros((K, frames), dtype=np.float32)
    if precomp_sources_off is None or precomp_sources_on is None or precomp_sources_hf is None:
        return out

    end_ptr = ptr + frames
    valid = max(0, min(end_ptr, n_frames) - max(ptr, 0))
    if valid <= 0:
        return out

    s0 = max(0, ptr)
    s1 = s0 + valid

    off_blk = precomp_sources_off[:, s0:s1]
    on_blk  = precomp_sources_on[:, s0:s1]

    if not use_mwf:
        out[:, :valid] = off_blk
        return out

    if use_hf:
        hf_blk = precomp_sources_hf[:, s0:s1]
        target_blk = on_blk + hf_blk
    else:
        target_blk = on_blk

    # === Modified: クロスフェード＆音量補正つきのバイパス処理 ===
    bypass_samples = int(MWF_START_BYPASS_SEC * SR_TARGET)
    xfade_samples = int(0.2 * SR_TARGET) # 0.2秒かけて滑らかにクロスフェード
    start_xf = max(0, bypass_samples - xfade_samples)

    local_start = ptr
    local_end = ptr + valid

    if local_end <= start_xf:
        # 完全バイパス領域 (音量補正したSHを流す)
        for k in range(K):
            out[k, :valid] = off_blk[k] * bypass_gain_adj[k]
            
    elif local_start >= bypass_samples:
        # 完全MWF領域
        out[:, :valid] = target_blk
        
    else:
        # クロスフェード移行領域 (ブロックを跨いでもサンプル単位で正確にフェード)
        t = np.arange(local_start, local_end)
        ramp = (t - start_xf) / float(bypass_samples - start_xf)
        ramp = np.clip(ramp, 0.0, 1.0).astype(np.float32)
        
        for k in range(K):
            # 音量補正済みSH(OFF)から、MWF(ON)へ徐々にフェード
            out[k, :valid] = (off_blk[k] * bypass_gain_adj[k] * (1.0 - ramp)) + (target_blk[k] * ramp)

    return out
    # ==============================================================

def audio_callback(outdata, frames, time_info, status):
    global read_ptr, playback_done, limiter_gain, playback_progress
    global meter_peak_db, meter_rms_db, meter_L_db, meter_R_db
    global meter_peak_hold_db, meter_L_hold_db, meter_R_hold_db
    global _meter_peak_hold_t, _meter_L_hold_t, _meter_R_hold_t
    global meter_pk_smooth, meter_rms_smooth, meter_L_smooth, meter_R_smooth, _meter_last_t
    global xfade_request, xfade_remaining, _prev_outL, _prev_outR

    if stop_playback or paused:
        outdata[:] = np.zeros_like(outdata)
        return

    if frames != BLOCK_SIZE:
        outdata[:] = np.zeros_like(outdata)
        return

    # read pointer update only（音源はキャッシュから取得）
    start_ptr = read_ptr
    end_ptr = read_ptr + frames
    if end_ptr <= n_frames:
        read_ptr = end_ptr
        if read_ptr >= n_frames:
            playback_done = True
    else:
        read_ptr = n_frames
        playback_done = True

    playback_progress = (read_ptr / max(1, n_frames))

    # snapshot mixer params
    with state_lock:
        params = [(b["theta"], b["phi"], b["fader_pos"], b["pan"], b["mute"], b["solo"]) for b in beam_states]
        mode_gain = float(MODE_GAINS[BEAM_MODE])
        solo_active = any(b["solo"] for b in beam_states)
        master_pos = master_fader_pos
        use_mwf = bool(mwf_engine.use_mwf)
        use_hf = USE_HF_COMP

    master_gain = fader_pos_to_gain(master_pos)

    # --- precomputed mono sources ---
    mono_hops = _get_cached_source_block(start_ptr, frames, use_mwf=use_mwf, use_hf=use_hf)

    # mix
    mixL = np.zeros((frames,), dtype=np.float32)
    mixR = np.zeros((frames,), dtype=np.float32)
    active_count = 0

    for i, (_th, _ph, fpos, pan, mute, solo) in enumerate(params):
        if solo_active and not solo:
            continue
        if mute:
            continue
        gain = fader_pos_to_gain(fpos)
        if gain <= 0.0:
            continue

        active_count += 1
        mono = mono_hops[i] * mode_gain

        pan = max(-1.0, min(1.0, float(pan)))
        angle = (pan + 1.0) * (math.pi / 4.0)
        gL = math.cos(angle)
        gR = math.sin(angle)

        mixL += (gain * gL) * mono
        mixR += (gain * gR) * mono

    if AUTO_COMPENSATE and active_count >= 2:
        s = (1.0 / math.sqrt(active_count))
        mixL *= s
        mixR *= s

    mixL *= master_gain
    mixR *= master_gain

    # ---- METER (pre-limiter): smooth + hold ----
    pkL = float(np.max(np.abs(mixL)) + 1e-12)
    pkR = float(np.max(np.abs(mixR)) + 1e-12)
    pk  = float(max(pkL, pkR) + 1e-12)
    rms = float(np.sqrt(np.mean(0.5 * (mixL*mixL + mixR*mixR)) + 1e-12))

    now = time.time()
    pk_db  = _safe_db(pk,  floor=-120.0)
    rms_db = _safe_db(rms, floor=-120.0)
    L_db   = _safe_db(pkL, floor=-120.0)
    R_db   = _safe_db(pkR, floor=-120.0)

    if _meter_last_t is None:
        _meter_last_t = now

    dt = max(1e-4, now - _meter_last_t)
    _meter_last_t = now

    tau_a = 0.03
    tau_r = 0.18

    def smooth_db(prev, cur):
        if cur > prev:
            a2 = 1.0 - math.exp(-dt / tau_a)
        else:
            a2 = 1.0 - math.exp(-dt / tau_r)
        return prev + (cur - prev) * a2

    meter_pk_smooth  = smooth_db(meter_pk_smooth,  pk_db)
    meter_rms_smooth = smooth_db(meter_rms_smooth, rms_db)
    meter_L_smooth   = smooth_db(meter_L_smooth,   L_db)
    meter_R_smooth   = smooth_db(meter_R_smooth,   R_db)

    if pk_db >= meter_peak_hold_db or (now - _meter_peak_hold_t) > PEAK_HOLD_SEC:
        meter_peak_hold_db = pk_db
        _meter_peak_hold_t = now
    if L_db >= meter_L_hold_db or (now - _meter_L_hold_t) > PEAK_HOLD_SEC:
        meter_L_hold_db = L_db
        _meter_L_hold_t = now
    if R_db >= meter_R_hold_db or (now - _meter_R_hold_t) > PEAK_HOLD_SEC:
        meter_R_hold_db = R_db
        _meter_R_hold_t = now

    with meter_lock:
        meter_peak_db = meter_pk_smooth
        meter_rms_db  = meter_rms_smooth
        meter_L_db    = meter_L_smooth
        meter_R_db    = meter_R_smooth

    # ---- limiter + softclip ----
    mixL, mixR = apply_limiter_stereo(mixL, mixR)
    if SOFTCLIP_AFTER:
        mixL = softclip(mixL)
        mixR = softclip(mixR)

    if not np.isfinite(mixL).all() or not np.isfinite(mixR).all():
        mixL = np.nan_to_num(mixL, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mixR = np.nan_to_num(mixR, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---- clickless crossfade ----
    with xfade_lock:
        if xfade_request:
            xfade_request = False
            xfade_remaining = xfade_total

    if xfade_remaining > 0:
        t0 = (xfade_total - xfade_remaining) / max(1.0, float(xfade_total))
        t1 = (xfade_total - (xfade_remaining - BLOCK_SIZE)) / max(1.0, float(xfade_total))
        t0 = max(0.0, min(1.0, t0))
        t1 = max(0.0, min(1.0, t1))
        ramp = np.linspace(t0, t1, BLOCK_SIZE, dtype=np.float32)
        outL = (_prev_outL * (1.0 - ramp)) + (mixL.astype(np.float32) * ramp)
        outR = (_prev_outR * (1.0 - ramp)) + (mixR.astype(np.float32) * ramp)
        xfade_remaining -= BLOCK_SIZE
        if xfade_remaining <= 0:
            xfade_remaining = 0
    else:
        outL = mixL.astype(np.float32, copy=False)
        outR = mixR.astype(np.float32, copy=False)

    _prev_outL[:] = outL
    _prev_outR[:] = outR

    out = np.zeros((frames, OUTPUT_CHANNELS), dtype=np.float32)
    out[:, 0] = outL
    out[:, 1] = outR
    outdata[:] = out

# ---------------- AUDIO THREAD ----------------
def audio_thread():
    with sd.OutputStream(
        channels=OUTPUT_CHANNELS,
        samplerate=SR_TARGET,
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
        dtype='float32',
        latency="high",
    ):
        while not stop_playback:
            time.sleep(0.02)

t_audio = threading.Thread(target=audio_thread, daemon=True)
t_audio.start()

# ---------------- PYGAME UI ----------------
pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("AmbiX Multi-Beam Mixer + Precomputed MWF ON/OFF")
font = pygame.font.SysFont(None, 20)
font_small = pygame.font.SysFont(None, 18)
font_big = pygame.font.SysFont(None, 28)
clock = pygame.time.Clock()

mouse_down = False
drag_seek = False
dragging_fader_index = None
dragging_pan_index = None
dragging_beam = False

arrow_times = {pygame.K_LEFT: 0, pygame.K_RIGHT: 0, pygame.K_UP: 0, pygame.K_DOWN: 0}
continuous_flags = {pygame.K_LEFT: False, pygame.K_RIGHT: False, pygame.K_UP: False, pygame.K_DOWN: False}

def _db_to_ratio(db, floor_db=METER_FLOOR_DB):
    db = float(db)
    if db <= floor_db:
        return 0.0
    if db >= 0.0:
        return 1.0
    return (db - floor_db) / (0.0 - floor_db)

def draw_shadowed_round_rect(x, y, w, h, radius, fill, shadow=(0, 0, 0, 120), shadow_off=(0, 3)):
    s = pygame.Surface((w + 16, h + 16), pygame.SRCALPHA)
    pygame.draw.rect(s, shadow, (8 + shadow_off[0], 8 + shadow_off[1], w, h), border_radius=radius)
    screen.blit(s, (x - 8, y - 8))
    pygame.draw.rect(screen, fill, (x, y, w, h), border_radius=radius)

def draw_topbar():
    draw_shadowed_round_rect(14, 12, WIN_W - 28, 78, 16, (20, 24, 30))
    pygame.draw.line(screen, (60, 70, 85), (24, 90), (WIN_W - 24, 90), 1)

def draw_badge(text, x, y, on=True):
    col = (70, 170, 120) if on else (120, 120, 120)
    pygame.draw.rect(screen, col, (x, y, 54, 22), border_radius=10)
    screen.blit(font_small.render(text, True, (255, 255, 255)), (x + 12, y + 3))

def draw_button(rect, label, fill, hover=False):
    c = tuple(min(255, int(v * (1.15 if hover else 1.0))) for v in fill)
    draw_shadowed_round_rect(rect.x, rect.y, rect.w, rect.h, 12, c)
    txt = font.render(label, True, (255, 255, 255))
    screen.blit(txt, (rect.x + (rect.w - txt.get_width()) // 2, rect.y + (rect.h - txt.get_height()) // 2))

def draw_master_meter(x, y, w, h):
    with meter_lock:
        pk_s = float(meter_peak_db)
        rm_s = float(meter_rms_db)
        pk_hold = float(meter_peak_hold_db)
        Lh = float(meter_L_hold_db)
        Rh = float(meter_R_hold_db)

    draw_shadowed_round_rect(x - 14, y - 26, w + 28, h + 80, 16, (20, 24, 30))
    screen.blit(font_small.render("MTR", True, (210, 210, 220)), (x, y - 20))

    pygame.draw.rect(screen, (40, 44, 54), (x, y, w, h), border_radius=12)
    pygame.draw.rect(screen, (70, 80, 95), (x, y, w, h), 1, border_radius=12)

    ratio = _db_to_ratio(pk_s)
    bar_h = int(h * ratio)
    bar_top = y + (h - bar_h)

    if pk_s < METER_GREEN_DB:
        col = (90, 220, 140)
    elif pk_s < METER_ORANGE_DB:
        col = (245, 180, 75)
    else:
        col = (245, 90, 90)

    pygame.draw.rect(screen, col, (x + 3, bar_top + 3, w - 6, max(0, bar_h - 6)), border_radius=10)

    pk_ratio = _db_to_ratio(pk_hold)
    pk_y = y + int(h * (1.0 - pk_ratio))
    pygame.draw.line(screen, (235, 235, 240), (x + 2, pk_y), (x + w - 2, pk_y), 2)

    def tline(s, dy):
        screen.blit(font_small.render(s, True, (220, 220, 230)), (x - 2, y + h + dy))

    tline(f"Pk {pk_s:5.1f} dB", 10)
    tline(f"Rms{rm_s:5.1f} dB", 28)
    tline(f"L {Lh:5.1f}  R {Rh:5.1f}", 46)

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def draw_seekbar(seek_rect, read_ptr_local):
    total = max(1, n_frames)
    rel = clamp01(read_ptr_local / total)
    fill_w = int(seek_rect.w * rel)

    pygame.draw.rect(screen, (40, 44, 54), seek_rect, border_radius=10)
    pygame.draw.rect(screen, (70, 80, 95), seek_rect, 1, border_radius=10)

    fill_rect = pygame.Rect(seek_rect.x, seek_rect.y, fill_w, seek_rect.h)
    pygame.draw.rect(screen, (220, 80, 90), fill_rect, border_radius=10)

    hx = seek_rect.x + fill_w
    hy = seek_rect.centery
    pygame.draw.circle(screen, (235, 235, 240), (hx, hy), 8)
    pygame.draw.circle(screen, (60, 70, 85), (hx, hy), 8, 2)

    cur_sec = (read_ptr_local / SR_TARGET) if SR_TARGET else 0.0
    total_sec = (n_frames / SR_TARGET) if SR_TARGET else 0.0

    screen.blit(font_small.render(format_time(cur_sec), True, (220, 220, 230)),
                (seek_rect.x, seek_rect.y - 20))
    tot = font_small.render(format_time(total_sec), True, (220, 220, 230))
    screen.blit(tot, (seek_rect.right - tot.get_width(), seek_rect.y - 20))

    mx, my = pygame.mouse.get_pos()
    if seek_rect.collidepoint((mx, my)):
        rel_h = clamp01((mx - seek_rect.x) / max(1, seek_rect.w))
        tsec = rel_h * total_sec
        tip = font_small.render(format_time(tsec), True, (255, 255, 255))
        tw, th = tip.get_width(), tip.get_height()
        tx = max(seek_rect.x, min(mx - tw // 2, seek_rect.right - tw))
        ty = seek_rect.y + 18
        pygame.draw.rect(screen, (15, 18, 24), (tx - 8, ty - 6, tw + 16, th + 12), border_radius=10)
        pygame.draw.rect(screen, (70, 80, 95), (tx - 8, ty - 6, tw + 16, th + 12), 1, border_radius=10)
        screen.blit(tip, (tx, ty))

def _theta_phi_to_xy(theta, phi, map_rect):
    x = map_rect.x + int(((math.degrees(theta) % 360.0) / 360.0) * map_rect.w)
    y = map_rect.y + int((0.5 - (math.degrees(phi) / 180.0)) * map_rect.h)
    return x, y

def draw_progress_bar(x, y, w, h, ratio, label, col=(90, 200, 255)):
    ratio = max(0.0, min(1.0, float(ratio)))
    pygame.draw.rect(screen, (40, 44, 54), (x, y, w, h), border_radius=8)
    pygame.draw.rect(screen, (70, 80, 95), (x, y, w, h), 1, border_radius=8)
    fillw = int(w * ratio)
    if fillw > 0:
        pygame.draw.rect(screen, col, (x, y, fillw, h), border_radius=8)
    txt = font_small.render(f"{label} {ratio*100:.1f}%", True, (225, 230, 240))
    screen.blit(txt, (x + 8, y + (h - txt.get_height()) // 2))

def draw_ui(read_ptr_local):
    screen.fill((14, 16, 20))
    draw_topbar()

    mode_names = ["Basic", "MaxRe", "MaxDI"]
    with state_lock:
        mwf_on = mwf_engine.use_mwf
        lm_on = mwf_engine.loudness_match
        a = mwf_engine.alpha
        dl = mwf_engine.diag_load
        pp = mwf_engine.p
        local_beams = [b.copy() for b in beam_states]
        sel_idx = selected_index
        master_pos_local = master_fader_pos
        hf_on = USE_HF_COMP
    with precompute_lock:
        pc_ip = precompute_in_progress
        pc_prog = precompute_progress
        pc_stat = precompute_status
        pc_err = precompute_error
        pc_phase = precompute_phase

    title = "AmbiX Mixer + Precomputed MWF ON/OFF"
    screen.blit(font_big.render(title, True, (240, 240, 245)), (30, 22))
    
    # === 修正①: Mode、hop、nfftのテキストをタイトルの下(y=56)にずらす ===
    screen.blit(font_small.render(f"Mode: {mode_names[BEAM_MODE]}  hop={BLOCK_SIZE} nfft={N_FFT}", True, (180, 190, 205)), (370, 56))

    draw_badge("MWF", 700, 24, mwf_on)
    draw_badge("LM",  760, 24, lm_on)
    draw_badge("HF",  820, 24, hf_on)

    stat = f"alpha={a:.3f}  diag={dl:.3g}  p={pp:.2f}"
    
    # === 修正②: alpha等のテキストを①と同じ高さ(y=56)にずらす ===
    screen.blit(font_small.render(stat, True, (180, 190, 205)), (890, 56))

    btn_rebuild = pygame.Rect(WIN_W - 400, 20, 110, 34)
    btn_rect = pygame.Rect(WIN_W - 280, 20, 110, 34)
    play_rect = pygame.Rect(WIN_W - 160, 20, 110, 34)
    mx, my = pygame.mouse.get_pos()
    draw_button(btn_rebuild, "Rebuild", (120, 140, 220), btn_rebuild.collidepoint((mx, my)))
    draw_button(btn_rect, "Save", (95, 110, 220), btn_rect.collidepoint((mx, my)))
    draw_button(play_rect, "Play/Pause", (90, 200, 120), play_rect.collidepoint((mx, my)))

    map_rect = pygame.Rect(MAP_LEFT, MAP_TOP + 22, MAP_WIDTH, MAP_HEIGHT)  # topbar拡張ぶん下げる
    draw_shadowed_round_rect(map_rect.x, map_rect.y, map_rect.w, map_rect.h, 18, (20, 24, 30))
    inner = map_rect.inflate(-12, -12)
    pygame.draw.rect(screen, (30, 34, 42), inner, border_radius=16)
    pygame.draw.rect(screen, (70, 80, 95), inner, 1, border_radius=16)

    for deg in [0, 90, 180, 270, 360]:
        x = inner.x + int((deg / 360.0) * inner.w)
        pygame.draw.line(screen, (55, 60, 72), (x, inner.y), (x, inner.bottom), 1)
        screen.blit(font_small.render(f"{deg}°", True, (160, 170, 190)), (x - 12, inner.y + 4))
    for deg in [90, 45, 0, -45, -90]:
        y = inner.y + int((0.5 - deg / 180.0) * inner.h)
        pygame.draw.line(screen, (55, 60, 72), (inner.x, y), (inner.right, y), 1)
        screen.blit(font_small.render(f"{deg}°", True, (160, 170, 190)), (inner.x + 6, y - 10))

    for b in local_beams:
        base_deg = (math.degrees(b["base_theta"]) % 360.0)
        sx = inner.x + int((base_deg / 360.0) * inner.w)
        sy = inner.y + int((0.5 - math.degrees(b["base_phi"]) / 180.0) * inner.h)
        pygame.draw.circle(screen, (255, 110, 120), (sx, sy), 4)

    for idx, b in enumerate(local_beams):
        th_deg = (math.degrees(b["theta"]) % 360.0)
        ph_deg = math.degrees(b["phi"])
        cx = inner.x + int((th_deg / 360.0) * inner.w)
        cy = inner.y + int((0.5 - ph_deg / 180.0) * inner.h)
        radius = 11 if idx == sel_idx else 8
        color = (150, 255, 170) if idx == sel_idx else (110, 210, 140)
        pygame.draw.circle(screen, color, (cx, cy), radius)
        pygame.draw.circle(screen, (15, 18, 24), (cx, cy), radius, 2)

        name = font.render(b["name"], True, (245, 245, 235))
        screen.blit(name, (cx + 10, cy - 14))

    if 0 <= sel_idx < len(local_beams):
        b_sel = local_beams[sel_idx]
        db_val = fader_pos_to_db(b_sel["fader_pos"])
        db_text = "-∞ dB" if db_val == float("-inf") else f"{db_val:.1f} dB"
        ms_text = (" [M]" if b_sel["mute"] else "") + (" [S]" if b_sel["solo"] else "")
        text = (f"Selected: {b_sel['name']}{ms_text}   "
                f"θ={math.degrees(b_sel['theta']):.1f}°   "
                f"φ={math.degrees(b_sel['phi']):+.1f}°   "
                f"Fader={db_text}   Pan={b_sel['pan']:.2f}")
        screen.blit(font_small.render(text, True, (210, 245, 225)), (inner.x + 12, inner.y + 12))

    seek_rect = pygame.Rect(inner.x + 40, inner.bottom + 22, inner.w - 80, 18)
    draw_seekbar(seek_rect, read_ptr_local)

    strip_top = map_rect.y
    meter_w = 26
    
    # === 修正③: 横幅拡張に伴い、メーターを右端から少し離してテキスト見切れを防ぐ ===
    meter_x = WIN_W - 90 
    draw_master_meter(meter_x, strip_top + 30, meter_w, 240)

    fader_rects = []
    pan_rects = []
    mute_rects = []
    solo_rects = []

    num = len(local_beams)
    if num > 0:
        GAP_MAP_TO_FADERS = 36
        GAP_FADERS_TO_METER = 20
        RIGHT_PANEL_LEFT = map_rect.right + GAP_MAP_TO_FADERS
        RIGHT_PANEL_RIGHT = (meter_x - GAP_FADERS_TO_METER)

        available_w = max(120, RIGHT_PANEL_RIGHT - RIGHT_PANEL_LEFT)
        base_strip_w = 78
        base_spacing = 14
        total_strips = num + 1  # + Master
        need_w = total_strips * base_strip_w + (total_strips - 1) * base_spacing

        if need_w > available_w:
            scale = available_w / need_w
            strip_width = max(56, int(base_strip_w * scale))
            spacing = max(8, int(base_spacing * scale))
        else:
            strip_width = base_strip_w
            spacing = base_spacing

        total_width = total_strips * strip_width + (total_strips - 1) * spacing
        start_x = RIGHT_PANEL_LEFT
        slack = available_w - total_width
        if slack > 0:
            start_x = RIGHT_PANEL_LEFT + int(slack * 0.35)

        fader_area_top = strip_top + 30
        fader_area_bottom = map_rect.y + MAP_HEIGHT - 62
        fader_height = fader_area_bottom - fader_area_top

        pan_height = 12
        pan_area_top = fader_area_bottom + 12
        label_y = pan_area_top + 18

        with meter_lock:
            Lh = float(meter_L_hold_db)
            Rh = float(meter_R_hold_db)

        for i in range(total_strips):
            x = start_x + i * (strip_width + spacing)
            is_master = (i == num)

            panel = pygame.Rect(x, fader_area_top - 16, strip_width, fader_height + 16 + 70)
            pygame.draw.rect(screen, (20, 24, 30), panel, border_radius=16)
            pygame.draw.rect(screen, (70, 80, 95), panel, 1, border_radius=16)

            f_rect = pygame.Rect(x + 10, fader_area_top, strip_width - 20, fader_height)
            fader_rects.append(f_rect)
            pygame.draw.rect(screen, (30, 34, 42), f_rect, border_radius=14)
            pygame.draw.rect(screen, (60, 70, 85), f_rect, 1, border_radius=14)

            zero_pos = db_to_fader_pos(0.0)
            zero_y = f_rect.y + int((1.0 - zero_pos) * f_rect.h)
            pygame.draw.line(screen, (220, 220, 230), (f_rect.x + 4, zero_y), (f_rect.right - 4, zero_y), 2)
            zlab = font_small.render("0", True, (230, 230, 240))
            screen.blit(zlab, (f_rect.x + 6, zero_y - zlab.get_height() // 2))

            pos = clamp01(local_beams[i]["fader_pos"]) if not is_master else clamp01(master_pos_local)
            knob_y = f_rect.y + int((1.0 - pos) * f_rect.h)
            knob_rect = pygame.Rect(f_rect.x + 8, knob_y - 9, f_rect.w - 16, 18)
            knob_color = (245, 230, 190) if is_master else ((235, 235, 240) if i == sel_idx else (210, 210, 220))
            pygame.draw.rect(screen, knob_color, knob_rect, border_radius=10)
            pygame.draw.rect(screen, (15, 18, 24), knob_rect, 2, border_radius=10)

            if not is_master:
                pan_rect = pygame.Rect(x + 12, pan_area_top, strip_width - 24, pan_height)
                pan_rects.append(pan_rect)
                pygame.draw.rect(screen, (30, 34, 42), pan_rect, border_radius=8)
                pygame.draw.rect(screen, (60, 70, 85), pan_rect, 1, border_radius=8)

                pan_val = max(-1.0, min(1.0, float(local_beams[i]["pan"])))
                rel = (pan_val + 1.0) / 2.0
                knob_x = pan_rect.x + int(rel * pan_rect.w)
                pygame.draw.circle(screen, (235, 235, 240) if i == sel_idx else (210, 210, 220),
                                   (knob_x, pan_rect.centery), 6)
                pygame.draw.circle(screen, (15, 18, 24), (knob_x, pan_rect.centery), 6, 2)
            else:
                pan_rects.append(pygame.Rect(0, 0, 0, 0))

            label_text = local_beams[i]["name"] if not is_master else "Master"
            name_label = font.render(label_text, True, (245, 245, 245))
            screen.blit(name_label, (x + (strip_width - name_label.get_width()) // 2, label_y))

            if is_master:
                lr_txt = font_small.render(f"L {Lh:4.1f}  R {Rh:4.1f}", True, (200, 210, 225))
                screen.blit(lr_txt, (x + (strip_width - lr_txt.get_width()) // 2, label_y + 22))

            if not is_master:
                m_rect = pygame.Rect(x + 16, label_y + 22, 24, 18)
                s_rect = pygame.Rect(x + 48, label_y + 22, 24, 18)
                mute_rects.append(m_rect)
                solo_rects.append(s_rect)

                b = local_beams[i]
                m_on = b["mute"]
                s_on = b["solo"]
                m_color = (220, 90, 100) if m_on else (60, 70, 85)
                s_color = (90, 210, 140) if s_on else (60, 70, 85)

                pygame.draw.rect(screen, m_color, m_rect, border_radius=8)
                pygame.draw.rect(screen, s_color, s_rect, border_radius=8)
                pygame.draw.rect(screen, (15, 18, 24), m_rect, 2, border_radius=8)
                pygame.draw.rect(screen, (15, 18, 24), s_rect, 2, border_radius=8)

                screen.blit(font_small.render("M", True, (255, 255, 255)), (m_rect.x + 7, m_rect.y + 1))
                screen.blit(font_small.render("S", True, (255, 255, 255)), (s_rect.x + 7, s_rect.y + 1))
            else:
                mute_rects.append(pygame.Rect(0, 0, 0, 0))
                solo_rects.append(pygame.Rect(0, 0, 0, 0))

    return btn_rebuild, btn_rect, play_rect, seek_rect, inner, fader_rects, pan_rects, mute_rects, solo_rects

# ---------------- SAVE FUNCTION ----------------
def save_current_mix_dialog():
    global paused
    paused = True

    root = tk.Tk()
    root.withdraw()
    filename = filedialog.asksaveasfilename(
        defaultextension=".wav",
        filetypes=[("WAV files", "*.wav")],
        title="現在の設定でミックスを保存（前計算済み MWF ON/OFF切替を反映）"
    )
    root.destroy()

    if filename:
        with state_lock:
            params = [(b["fader_pos"], b["pan"], b["mute"], b["solo"]) for b in beam_states]
            mode_gain = float(MODE_GAINS[BEAM_MODE])
            solo_active = any(b["solo"] for b in beam_states)
            master_pos = master_fader_pos
            use_mwf = bool(mwf_engine.use_mwf)
            use_hf = USE_HF_COMP

        if precomp_sources_off is None or precomp_sources_on is None:
            print("[Save] キャッシュ未作成のため保存できません。")
            paused = False
            return

        master_gain = fader_pos_to_gain(master_pos)
        outL = np.zeros((n_frames,), dtype=np.float32)
        outR = np.zeros((n_frames,), dtype=np.float32)

        ptr = 0
        t0 = time.time()
        local_limiter_gain = 1.0

        def offline_limiter(L, R):
            nonlocal local_limiter_gain
            peak = float(max(np.max(np.abs(L)), np.max(np.abs(R))) + 1e-12)
            desired = 1.0
            if peak > LIMIT_THRESH:
                desired = LIMIT_THRESH / peak
            if desired < local_limiter_gain:
                local_limiter_gain = local_limiter_gain * (1.0 - ATTACK) + desired * ATTACK
            else:
                local_limiter_gain = local_limiter_gain * (1.0 - RELEASE) + desired * RELEASE
            local_limiter_gain = float(max(0.0, min(1.0, local_limiter_gain)))
            return L * local_limiter_gain, R * local_limiter_gain

        while ptr < n_frames:
            endp = min(n_frames, ptr + BLOCK_SIZE)
            valid = endp - ptr
            mono_hops = _get_cached_source_block(ptr, BLOCK_SIZE, use_mwf=use_mwf, use_hf=use_hf)  # K x BLOCK_SIZE

            mixL = np.zeros((BLOCK_SIZE,), dtype=np.float32)
            mixR = np.zeros((BLOCK_SIZE,), dtype=np.float32)
            active_count = 0

            for i, (fpos, pan, mute, solo) in enumerate(params):
                if solo_active and not solo:
                    continue
                if mute:
                    continue
                gain = fader_pos_to_gain(fpos)
                if gain <= 0.0:
                    continue

                active_count += 1
                mono = mono_hops[i] * mode_gain

                pan = max(-1.0, min(1.0, float(pan)))
                angle = (pan + 1.0) * (math.pi / 4.0)
                gL = math.cos(angle)
                gR = math.sin(angle)
                mixL += (gain * gL) * mono
                mixR += (gain * gR) * mono

            if AUTO_COMPENSATE and active_count >= 2:
                s = (1.0 / math.sqrt(active_count))
                mixL *= s
                mixR *= s

            mixL *= master_gain
            mixR *= master_gain

            mixL, mixR = offline_limiter(mixL, mixR)
            if SOFTCLIP_AFTER:
                mixL = softclip(mixL)
                mixR = softclip(mixR)

            outL[ptr:endp] = mixL[:valid]
            outR[ptr:endp] = mixR[:valid]
            ptr = endp

            # 保存時進捗
            if ptr % max(1, int(SR_TARGET * 1)) < BLOCK_SIZE:
                print(f"[Save] {ptr/n_frames*100:.1f}%")

        stereo = np.stack([outL, outR], axis=1).astype(np.float32, copy=False)
        sf.write(filename, stereo, SR_TARGET)
        print(f"保存完了: {filename}   (took {time.time()-t0:.1f}s)")
        request_clickless_update(reset_mwf=False)

    paused = False

# ---------------- MAIN LOOP ----------------
RUNNING = True

while RUNNING:
    mouse_pos = pygame.mouse.get_pos()
    btn_rebuild, btn_rect, play_rect, seek_rect, map_inner, fader_rects, pan_rects, mute_rects, solo_rects = draw_ui(read_ptr)
    pygame.display.flip()
    clock.tick(60)

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            RUNNING = False

        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mouse_down = True

            if btn_rebuild.collidepoint(ev.pos):
                rebuild_precomputed_sources_with_pause()

            elif btn_rect.collidepoint(ev.pos):
                save_current_mix_dialog()

            elif play_rect.collidepoint(ev.pos):
                toggle_play_pause()

            elif seek_rect.collidepoint(ev.pos):
                drag_seek = True
                rel = (ev.pos[0] - seek_rect.x) / max(1, seek_rect.w)
                rel = max(0.0, min(1.0, rel))
                read_ptr = int(rel * n_frames)
                read_ptr = max(0, min(read_ptr, n_frames - 1))
                playback_done = False
                request_clickless_update(reset_mwf=False)

            else:
                dragging_fader_index = None
                dragging_pan_index = None
                dragging_beam = False

                for i, rect in enumerate(pan_rects):
                    if rect.w > 0 and rect.collidepoint(ev.pos) and i < len(beam_states):
                        dragging_pan_index = i
                        break

                if dragging_pan_index is None:
                    for i, rect in enumerate(fader_rects):
                        if rect.w > 0 and rect.collidepoint(ev.pos):
                            dragging_fader_index = i
                            break

                if dragging_fader_index is None and dragging_pan_index is None:
                    for i, rect in enumerate(mute_rects):
                        if rect.w > 0 and rect.collidepoint(ev.pos) and i < len(beam_states):
                            with state_lock:
                                beam_states[i]["mute"] = not beam_states[i]["mute"]
                            request_clickless_update(reset_mwf=False)
                            break
                    else:
                        for i, rect in enumerate(solo_rects):
                            if rect.w > 0 and rect.collidepoint(ev.pos) and i < len(beam_states):
                                with state_lock:
                                    beam_states[i]["solo"] = not beam_states[i]["solo"]
                                request_clickless_update(reset_mwf=False)
                                break
                        else:
                            with state_lock:
                                local_beams = [b.copy() for b in beam_states]
                            clicked_idx = None
                            for i, b in enumerate(local_beams):
                                cx, cy = _theta_phi_to_xy(b["theta"], b["phi"], map_inner)
                                if (ev.pos[0] - cx) ** 2 + (ev.pos[1] - cy) ** 2 < (14 ** 2):
                                    clicked_idx = i
                                    break
                            if clicked_idx is not None:
                                with state_lock:
                                    selected_index = clicked_idx
                                dragging_beam = True

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            mouse_down = False
            drag_seek = False
            dragging_fader_index = None
            dragging_pan_index = None
            dragging_beam = False

        elif ev.type == pygame.MOUSEMOTION:
            if drag_seek:
                rel = (mouse_pos[0] - seek_rect.x) / max(1, seek_rect.w)
                rel = max(0.0, min(1.0, rel))
                read_ptr = int(rel * n_frames)
                read_ptr = max(0, min(read_ptr, n_frames - 1))
                playback_done = False
                request_clickless_update(reset_mwf=False)

            elif dragging_fader_index is not None:
                i = dragging_fader_index
                if 0 <= i < len(fader_rects):
                    rect = fader_rects[i]
                    rel = (mouse_pos[1] - rect.y) / max(1, rect.h)
                    rel = max(0.0, min(1.0, rel))
                    fpos = 1.0 - rel
                    with state_lock:
                        if i < len(beam_states):
                            beam_states[i]["fader_pos"] = fpos
                        else:
                            master_fader_pos = fpos

            elif dragging_pan_index is not None:
                i = dragging_pan_index
                if 0 <= i < len(pan_rects) and i < len(beam_states):
                    rect = pan_rects[i]
                    if rect.w > 0:
                        rel = (mouse_pos[0] - rect.x) / max(1, rect.w)
                        rel = max(0.0, min(1.0, rel))
                        pan_val = rel * 2.0 - 1.0
                        with state_lock:
                            beam_states[i]["pan"] = pan_val

            elif dragging_beam and mouse_down:
                with state_lock:
                    if 0 <= selected_index < len(beam_states):
                        theta_new = ((mouse_pos[0] - map_inner.x) / max(1, map_inner.w)) * 2 * np.pi
                        phi_new = ((0.5 - (mouse_pos[1] - map_inner.y) / max(1, map_inner.h)) * np.pi)
                        theta_new, phi_new = clamp_angles(theta_new, phi_new)
                        beam_states[selected_index]["theta"] = theta_new
                        beam_states[selected_index]["phi"] = phi_new
                # 方向変更はキャッシュ内容に影響するので dirty
                cache_dirty = True

        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                RUNNING = False

            elif ev.key == pygame.K_SPACE:
                toggle_play_pause()

            elif ev.key == pygame.K_r:
                # 手動再計算ショートカット
                rebuild_precomputed_sources_with_pause()

            elif ev.key == pygame.K_1:
                BEAM_MODE = 0
                print("Beam mode: Basic")
                cache_dirty = True
            elif ev.key == pygame.K_2:
                BEAM_MODE = 1
                print("Beam mode: MaxRe")
                cache_dirty = True
            elif ev.key == pygame.K_3:
                BEAM_MODE = 2
                print("Beam mode: MaxDI")
                cache_dirty = True

            elif ev.key == pygame.K_m:
                with state_lock:
                    mwf_engine.use_mwf = not mwf_engine.use_mwf
                print("MWF:", "ON" if mwf_engine.use_mwf else "OFF")
                request_clickless_update(reset_mwf=False)
                
            elif ev.key == pygame.K_h:
                with state_lock:
                    USE_HF_COMP = not USE_HF_COMP
                print("HF Compensation:", "ON" if USE_HF_COMP else "OFF")
                request_clickless_update(reset_mwf=False)

            elif ev.key == pygame.K_l:
                with state_lock:
                    mwf_engine.loudness_match = not mwf_engine.loudness_match
                    mwf_engine.reset_gain_tracking()
                print("Loudness Match:", "ON" if mwf_engine.loudness_match else "OFF")
                cache_dirty = True

            elif ev.key == pygame.K_LEFTBRACKET:
                with state_lock:
                    mwf_engine.diag_load = max(1e-6, mwf_engine.diag_load * 0.7)
                cache_dirty = True
            elif ev.key == pygame.K_RIGHTBRACKET:
                with state_lock:
                    mwf_engine.diag_load = min(1e-1, mwf_engine.diag_load * 1.4)
                cache_dirty = True
            elif ev.key == pygame.K_MINUS:
                with state_lock:
                    mwf_engine.alpha = max(0.005, mwf_engine.alpha * 0.85)
                cache_dirty = True
            elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS):
                with state_lock:
                    mwf_engine.alpha = min(0.20, mwf_engine.alpha * 1.15)
                cache_dirty = True
            elif ev.key == pygame.K_COMMA:
                with state_lock:
                    mwf_engine.p = max(0.8, mwf_engine.p - 0.2)
                cache_dirty = True
            elif ev.key == pygame.K_PERIOD:
                with state_lock:
                    mwf_engine.p = min(100, mwf_engine.p + 0.2)
                cache_dirty = True

            elif ev.key == pygame.K_RETURN and cache_dirty:
                # 設定変更後の再計算を Enter で実行
                rebuild_precomputed_sources_with_pause()

    keys = pygame.key.get_pressed()
    current_time = time.time()
    with state_lock:
        if 0 <= selected_index < len(beam_states):
            b = beam_states[selected_index]
            changed = False
            for k in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                if keys[k]:
                    if arrow_times[k] == 0:
                        arrow_times[k] = current_time
                        continuous_flags[k] = False
                        if k == pygame.K_LEFT:
                            b["theta"] -= MOVE_STEP; changed = True
                        elif k == pygame.K_RIGHT:
                            b["theta"] += MOVE_STEP; changed = True
                        elif k == pygame.K_UP:
                            b["phi"] += MOVE_STEP; changed = True
                        elif k == pygame.K_DOWN:
                            b["phi"] -= MOVE_STEP; changed = True
                    else:
                        if (not continuous_flags[k] and current_time - arrow_times[k] >= LONGPRESS_DELAY):
                            continuous_flags[k] = True
                        if continuous_flags[k]:
                            if k == pygame.K_LEFT:
                                b["theta"] -= MOVE_STEP; changed = True
                            elif k == pygame.K_RIGHT:
                                b["theta"] += MOVE_STEP; changed = True
                            elif k == pygame.K_UP:
                                b["phi"] += MOVE_STEP; changed = True
                            elif k == pygame.K_DOWN:
                                b["phi"] -= MOVE_STEP; changed = True
                else:
                    arrow_times[k] = 0
                    continuous_flags[k] = False

            b["theta"], b["phi"] = clamp_angles(b["theta"], b["phi"])
            if changed:
                cache_dirty = True

    # dirty 状態をUI表示用に status へ反映（自動再計算はしない）
    if cache_dirty:
        with precompute_lock:
            if not precompute_in_progress:
                precompute_phase = "dirty"
                precompute_status = "設定変更あり（R or Rebuild/Enter で前計算更新）"

stop_playback = True
pygame.quit()
t_audio.join()
print("アプリ終了、再生停止")
