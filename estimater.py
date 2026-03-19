# beam_search_realtime_maxdi_sources_export.py
# AmbiX (W,Y,Z,X) FOA -> realtime MaxDI steering + 2-stage direction search
# UI:
#  - Play / Pause / Stop buttons
#  - Add button: save current PLAY direction with a name (text input)
#  - Save button: export SOURCES list to .txt in the requested format
#
# Visual:
#  - Draw only 2 search points: Stage1 best (small) + FINAL best (bigger), both labeled
#  - Labels are clamped so they don't go off-screen
#  - Saved sources are drawn and listed

import argparse
import math
import threading
import time
from datetime import datetime

import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf
import scipy.special

# ================= USER SETTINGS =================
WAV_PATH = "方向推定用.wav"     # ★ここを変更すればOK（スクリプトと同じフォルダ推奨）
DEFAULT_SECONDS = 0.5         # 推定に使う区間長（秒）
DEFAULT_BLOCK = 128           # 再生ブロック
DEFAULT_FRAME_MS = 20.0       # 音圧メトリクスのフレーム長
DEFAULT_TOP_PERCENT = 20.0    # 上位何%のフレームRMSを平均するか

OUTPUT_TXT_DEFAULT = "sources_export.txt"  # ★保存先（同フォルダに出る）
# =================================================


# ----------------- SH (same style as your previous program) -----------------
def eval_sh(max_order, dirs_sph):
    """
    channel order (ACN): [Y_0^0, Y_1^-1, Y_1^0, Y_1^1] = [W, Y, Z, X]
    dirs_sph: [[azi(theta), ele(phi)]]
    """
    dirs_sph = np.array(dirs_sph, dtype=np.float64)
    if dirs_sph.ndim == 1:
        dirs_sph = dirs_sph.reshape((1, 2))
    num_sh_channels = (max_order + 1) ** 2
    num_dir = dirs_sph.shape[0]
    azi = dirs_sph[:, 0]
    ele = dirs_sph[:, 1]

    Y = np.zeros((num_dir, num_sh_channels), dtype=np.float64)
    for n in range(0, max_order + 1):
        for m in range(-n, n + 1):
            i = n + n ** 2 + m  # ACN index
            if m < 0:
                Yazi = np.sqrt(2) * np.sin(azi * np.abs(m))
            elif m > 0:
                Yazi = np.sqrt(2) * np.cos(azi * m)
            else:
                Yazi = np.ones(num_dir)

            Yzen = np.zeros(num_dir, dtype=np.float64)
            for iDir in range(num_dir):
                Yzen[iDir] = (-1) ** m * scipy.special.lpmv(
                    np.abs(m), n, np.cos(np.pi / 2 - ele[iDir])
                )

            normlz = np.sqrt(
                (2 * n + 1) * math.factorial(n - abs(m)) /
                (4 * np.pi * math.factorial(n + abs(m)))
            )
            Y[:, i] = Yazi * Yzen * normlz
    return Y


# ----------------- Utils -----------------
def clamp_angles(theta, phi):
    theta = theta % (2 * np.pi)
    phi = max(-np.pi / 2, min(np.pi / 2, phi))
    return theta, phi

def chunk_rms(x, frame_len):
    n = len(x)
    if n < frame_len:
        x = np.concatenate([x, np.zeros(frame_len - n, dtype=x.dtype)])
        n = len(x)
    n_frames = n // frame_len
    if n_frames <= 0:
        return np.array([], dtype=np.float64)
    x2 = x[:n_frames * frame_len].reshape(n_frames, frame_len).astype(np.float64)
    return np.sqrt(np.mean(x2 * x2, axis=1) + 1e-12)

def top_percentile_rms_db(x, sr, frame_ms=20.0, top_percent=20.0):
    frame_len = max(16, int(round(sr * frame_ms / 1000.0)))
    r = chunk_rms(np.asarray(x, dtype=np.float32), frame_len)
    if r.size == 0:
        return -1e18
    top_percent = float(np.clip(top_percent, 1.0, 100.0))
    k = max(1, int(round(r.size * (top_percent / 100.0))))
    idx = np.argpartition(r, -k)[-k:]
    mean_top = float(np.mean(r[idx]) + 1e-12)
    return 20.0 * math.log10(mean_top)


# ----------------- MaxDI beamformer (FOA) -----------------
def beamformer_max_di_foa_block(block_wyzx, theta, phi):
    """
    block_wyzx: (N,4) [W,Y,Z,X]
    MaxDI(FOA) -> hypercardioid-equivalent pattern for 1st order.
    Use steering vector normalized to unity at look direction.
    """
    input_signal = block_wyzx.astype(np.float64, copy=False)  # (N,4)
    y = eval_sh(1, np.array([[theta, phi]], dtype=np.float64))[0]  # (4,)
    denom = float(np.dot(y, y) + 1e-12)
    w = y / denom
    mono = input_signal @ w  # (N,)
    return mono.astype(np.float32)

def beamformer_max_di_from_components(W, Y, Z, X, theta, phi):
    block = np.stack([W, Y, Z, X], axis=1)
    return beamformer_max_di_foa_block(block, theta, phi)


# ----------------- Search -----------------
def generate_stage1_grids(theta_step=30):
    thetas_deg = list(range(0, 360, theta_step))
    phis_deg = list(range(-90, 91, 30))  # -90..+90 step 30
    return thetas_deg, phis_deg

def theta_candidates_for_phi(thetas_deg, phi_deg):
    return [0] if abs(phi_deg) == 90 else thetas_deg

def search_direction_two_stage_maxdi(W, Y, Z, X, sr, start_sample, seconds,
                                    coarse_theta_step=30,
                                    fine_half_range_deg=30,
                                    fine_step_deg=5,
                                    frame_ms=20.0,
                                    top_percent=20.0):
    seg_len = int(round(seconds * sr))
    s0 = max(0, int(start_sample))
    s1 = min(len(W), s0 + seg_len)
    if s1 <= s0 + 32:
        raise RuntimeError("探索区間が短すぎます。seconds を増やすか、再生位置を前にしてください。")

    Wseg = W[s0:s1]
    Yseg = Y[s0:s1]
    Zseg = Z[s0:s1]
    Xseg = X[s0:s1]

    # Stage1
    thetas_deg, phis_deg = generate_stage1_grids(coarse_theta_step)
    best1 = {"theta_deg": 0, "phi_deg": 0, "score_db": -1e18}

    for ph_deg in phis_deg:
        for th_deg in theta_candidates_for_phi(thetas_deg, ph_deg):
            th = math.radians(th_deg % 360)
            ph = math.radians(ph_deg)
            mono = beamformer_max_di_from_components(Wseg, Yseg, Zseg, Xseg, th, ph)
            sc = top_percentile_rms_db(mono, sr, frame_ms=frame_ms, top_percent=top_percent)
            if sc > best1["score_db"]:
                best1.update({"theta_deg": th_deg % 360, "phi_deg": ph_deg, "score_db": sc})

    # Stage2
    center_th = best1["theta_deg"]
    center_ph = best1["phi_deg"]

    th_candidates_raw = list(range(center_th - fine_half_range_deg, center_th + fine_half_range_deg + 1, fine_step_deg))
    ph_candidates_raw = list(range(center_ph - fine_half_range_deg, center_ph + fine_half_range_deg + 1, fine_step_deg))
    ph_candidates = [max(-90, min(90, p)) for p in ph_candidates_raw]
    seen = set()
    ph_candidates = [p for p in ph_candidates if (p not in seen and not seen.add(p))]

    best2 = {"theta_deg": center_th % 360, "phi_deg": center_ph, "score_db": -1e18}

    for ph_deg in ph_candidates:
        for th_deg in th_candidates_raw:
            th_deg_wrapped = th_deg % 360
            if abs(ph_deg) == 90 and th_deg_wrapped != 0:
                continue
            th = math.radians(th_deg_wrapped)
            ph = math.radians(ph_deg)
            mono = beamformer_max_di_from_components(Wseg, Yseg, Zseg, Xseg, th, ph)
            sc = top_percentile_rms_db(mono, sr, frame_ms=frame_ms, top_percent=top_percent)
            if sc > best2["score_db"]:
                best2.update({"theta_deg": th_deg_wrapped, "phi_deg": ph_deg, "score_db": sc})

    th1, ph1 = math.radians(best1["theta_deg"]), math.radians(best1["phi_deg"])
    th2, ph2 = math.radians(best2["theta_deg"]), math.radians(best2["phi_deg"])
    return (th1, ph1, best1["score_db"]), (th2, ph2, best2["score_db"])


# ----------------- Realtime App -----------------
class RealtimeBeamApp:
    def __init__(self, wav_path, seconds=1.0, block_size=128, move_step_deg=3.0,
                 frame_ms=20.0, top_percent=20.0, output_txt=OUTPUT_TXT_DEFAULT):
        self.wav_path = wav_path
        self.seconds = float(seconds)
        self.block_size = int(block_size)
        self.move_step = math.radians(float(move_step_deg))
        self.frame_ms = float(frame_ms)
        self.top_percent = float(top_percent)
        self.output_txt = str(output_txt)

        self.data, self.sr = sf.read(self.wav_path, always_2d=True)
        if self.data.shape[1] != 4:
            raise RuntimeError("WAV must be 4 channels (AmbiX order: W,Y,Z,X)")
        self.n_frames = self.data.shape[0]

        self.read_ptr = 0
        self.paused = True  # start paused
        self.stop_flag = False

        self.theta = 0.0
        self.phi = 0.0

        self.est_theta = None
        self.est_phi = None

        self.compare_mode = False
        self.off_theta = 0.0
        self.off_phi = 0.0

        self.angle_lock = threading.Lock()

        # draw only 2 search points
        self.stage1_best = None
        self.stage2_best = None

        # saved sources
        self.sources = []  # list of {"name": str, "theta": float(rad), "phi": float(rad)}
        self.pending_add_theta_phi = None  # (theta, phi) captured when Add pressed

        # input mode (naming)
        self.input_mode = False
        self.input_text = ""
        self.input_hint = "Name?"
        self.flash_t = 0.0

        # UI / window
        self.WIN_W, self.WIN_H = 940, 600
        self.MAP_H = 440
        self.font = None
        self.font_small = None
        self.clock = None
        self.screen = None

        # buttons
        yb = self.MAP_H + 10
        self.btn_play  = pygame.Rect(10,  yb, 85, 26)
        self.btn_pause = pygame.Rect(100, yb, 85, 26)
        self.btn_stop  = pygame.Rect(190, yb, 85, 26)
        self.btn_add   = pygame.Rect(290, yb, 90, 26)
        self.btn_save  = pygame.Rect(390, yb, 90, 26)

        # messages
        self.toast = ""
        self.toast_until = 0.0

        # label option
        self.show_point_labels = True

    def toast_msg(self, msg, sec=2.0):
        self.toast = msg
        self.toast_until = time.time() + sec

    def get_play_theta_phi(self):
        with self.angle_lock:
            if self.compare_mode and self.est_theta is not None:
                th = self.est_theta + self.off_theta
                ph = self.est_phi + self.off_phi
                return clamp_angles(th, ph)
            return clamp_angles(self.theta, self.phi)

    def audio_callback(self, outdata, frames, time_info, status):
        out = np.zeros((frames, 2), dtype=np.float32)

        if self.stop_flag or self.paused:
            outdata[:] = out
            return

        end_ptr = self.read_ptr + frames
        if end_ptr <= self.n_frames:
            block = self.data[self.read_ptr:end_ptr, :]
            self.read_ptr = end_ptr
        else:
            valid = max(0, self.n_frames - self.read_ptr)
            tail = self.data[self.read_ptr:self.read_ptr + valid, :] if valid > 0 else np.zeros((0, 4))
            pad = np.zeros((frames - valid, 4), dtype=tail.dtype)
            block = np.vstack([tail, pad])
            self.read_ptr = self.n_frames
            self.paused = True  # auto pause at end

        W = block[:, 0].astype(np.float32)
        Y = block[:, 1].astype(np.float32)
        Z = block[:, 2].astype(np.float32)
        X = block[:, 3].astype(np.float32)

        th, ph = self.get_play_theta_phi()
        mono = beamformer_max_di_from_components(W, Y, Z, X, th, ph)

        m = float(np.max(np.abs(mono)) + 1e-9)
        if m > 1.0:
            mono = mono / m

        out[:, 0] = mono
        out[:, 1] = mono
        outdata[:] = out

    def audio_thread(self):
        with sd.OutputStream(
            channels=2,
            samplerate=self.sr,
            blocksize=self.block_size,
            callback=self.audio_callback,
            dtype="float32",
        ):
            while not self.stop_flag:
                time.sleep(0.05)

    def theta_phi_to_xy(self, theta, phi):
        x = int((math.degrees(theta) % 360) / 360.0 * self.WIN_W)
        y = int((0.5 - (math.degrees(phi) / 180.0)) * self.MAP_H)
        return x, y

    def xy_to_theta_phi(self, x, y):
        theta = (x / self.WIN_W) * 2 * math.pi
        phi = (0.5 - (y / self.MAP_H)) * math.pi
        return clamp_angles(theta, phi)

    def draw_button(self, rect, label, enabled=True):
        bg = (40, 40, 40) if enabled else (25, 25, 25)
        fg = (235, 235, 235) if enabled else (140, 140, 140)
        pygame.draw.rect(self.screen, bg, rect, border_radius=6)
        pygame.draw.rect(self.screen, (90, 90, 90), rect, 1, border_radius=6)
        txt = self.font.render(label, True, fg)
        self.screen.blit(txt, (rect.x + 10, rect.y + 5))

    def draw_label_clamped(self, text, x, y, color=(230, 230, 230), area=(0, 0, 940, 440)):
        # shadow
        txt = self.font.render(text, True, color)
        w, h = txt.get_width(), txt.get_height()
        ax, ay, aw, ah = area

        # clamp so the text box stays within area
        x = max(ax + 2, min(ax + aw - w - 2, x))
        y = max(ay + 2, min(ay + ah - h - 2, y))

        shadow = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(shadow, (x + 2, y + 2))
        self.screen.blit(txt, (x, y))

    def export_sources_txt(self):
        # requested format
        lines = []
        lines.append("SOURCES = [")
        for s in self.sources:
            name = s["name"].replace('"', '\\"')
            th_deg = math.degrees(s["theta"])
            ph_deg = math.degrees(s["phi"])
            # keep radians call in output, as requested
            lines.append(f'    {{"name": "{name}", "theta": math.radians({th_deg:.3f}), "phi": math.radians({ph_deg:.3f})}},')
        lines.append("]")
        content = "\n".join(lines) + "\n"

        with open(self.output_txt, "w", encoding="utf-8") as f:
            f.write(content)

        self.toast_msg(f"Saved: {self.output_txt}", sec=2.5)

    def begin_name_input_for_add(self):
        th, ph = self.get_play_theta_phi()  # save current PLAY direction
        self.pending_add_theta_phi = (th, ph)
        self.input_mode = True
        self.input_text = f"Source{len(self.sources)+1}"
        self.input_hint = "Enter name (Enter=OK, Esc=Cancel)"
        self.flash_t = 0.0
        self.toast_msg("Type a name and press Enter", sec=2.0)

    def confirm_name_input(self):
        if self.pending_add_theta_phi is None:
            self.input_mode = False
            return
        name = self.input_text.strip()
        if name == "":
            name = f"Source{len(self.sources)+1}"

        th, ph = self.pending_add_theta_phi
        self.sources.append({"name": name, "theta": float(th), "phi": float(ph)})

        self.pending_add_theta_phi = None
        self.input_mode = False
        self.toast_msg(f"Added: {name}", sec=2.0)

    def cancel_name_input(self):
        self.pending_add_theta_phi = None
        self.input_mode = False
        self.toast_msg("Add canceled", sec=1.5)

    def run_search(self):
        start_sample = self.read_ptr

        W = self.data[:, 0].astype(np.float32)
        Y = self.data[:, 1].astype(np.float32)
        Z = self.data[:, 2].astype(np.float32)
        X = self.data[:, 3].astype(np.float32)

        (th1, ph1, sc1), (th2, ph2, sc2) = search_direction_two_stage_maxdi(
            W=W, Y=Y, Z=Z, X=X,
            sr=self.sr,
            start_sample=start_sample,
            seconds=self.seconds,
            coarse_theta_step=30,
            fine_half_range_deg=30,
            fine_step_deg=5,
            frame_ms=self.frame_ms,
            top_percent=self.top_percent
        )

        self.stage1_best = (th1, ph1)
        self.stage2_best = (th2, ph2)

        with self.angle_lock:
            self.est_theta = th2
            self.est_phi = ph2
            self.off_theta = 0.0
            self.off_phi = 0.0

        # ONLY 2 lines output
        print(f"[Stage1 BEST] theta={math.degrees(th1):.1f} deg, phi={math.degrees(ph1):+.1f} deg  ({sc1:.2f} dBFS)")
        print(f"[FINAL  BEST] theta={math.degrees(th2):.1f} deg, phi={math.degrees(ph2):+.1f} deg  ({sc2:.2f} dBFS)")

        self.toast_msg("Search done. Adjust then press Add.", sec=2.0)

    def draw(self):
        self.screen.fill((25, 25, 25))

        # grid
        for deg in [0, 90, 180, 270, 360]:
            x = int((deg / 360.0) * self.WIN_W)
            pygame.draw.line(self.screen, (60, 60, 60), (x, 0), (x, self.MAP_H), 1)
            self.screen.blit(self.font.render(f"{deg}°", True, (180, 180, 180)), (x - 12, 6))

        for deg in [90, 60, 30, 0, -30, -60, -90]:
            y = int((0.5 - (deg / 180.0)) * self.MAP_H)
            pygame.draw.line(self.screen, (60, 60, 60), (0, y), (self.WIN_W, y), 1)
            self.screen.blit(self.font.render(f"{deg:+d}°", True, (180, 180, 180)), (6, y - 10))

        # saved sources (small dots + labels)
        for i, s in enumerate(self.sources):
            th, ph = s["theta"], s["phi"]
            x, y = self.theta_phi_to_xy(th, ph)
            pygame.draw.circle(self.screen, (210, 210, 210), (x, y), 4)
            if self.show_point_labels:
                self.draw_label_clamped(
                    s["name"], x + 8, y - 18, (220, 220, 220),
                    area=(0, 0, self.WIN_W, self.MAP_H)
                )

        # ONLY 2 search points + labels (clamped)
        if self.stage1_best is not None:
            th, ph = self.stage1_best
            x, y = self.theta_phi_to_xy(th, ph)
            pygame.draw.circle(self.screen, (255, 170, 70), (x, y), 5)  # small
            if self.show_point_labels:
                self.draw_label_clamped(
                    "Stage1 best", x + 8, y - 18, (255, 200, 120),
                    area=(0, 0, self.WIN_W, self.MAP_H)
                )

        if self.stage2_best is not None:
            th, ph = self.stage2_best
            x, y = self.theta_phi_to_xy(th, ph)

            # ★ FINAL best の色を「普通な青」に変更
            final_dot_color = (80, 160, 255)
            final_label_color = (180, 210, 255)

            pygame.draw.circle(self.screen, final_dot_color, (x, y), 9)  # big
            if self.show_point_labels:
                self.draw_label_clamped(
                    "FINAL best", x + 8, y - 18, final_label_color,
                    area=(0, 0, self.WIN_W, self.MAP_H)
                )

        # estimated ring (final)
        if self.est_theta is not None:
            x, y = self.theta_phi_to_xy(self.est_theta, self.est_phi)
            # ★ リングも FINAL best と同系色に
            pygame.draw.circle(self.screen, (80, 160, 255), (x, y), 10, 2)

        # play direction (green)
        thp, php = self.get_play_theta_phi()
        xg, yg = self.theta_phi_to_xy(thp, php)
        pygame.draw.circle(self.screen, (90, 255, 120), (xg, yg), 9)
        if self.show_point_labels:
            self.draw_label_clamped(
                "PLAY", xg + 8, yg - 18, (160, 255, 190),
                area=(0, 0, self.WIN_W, self.MAP_H)
            )

        # bottom panel
        panel_y = self.MAP_H
        pygame.draw.rect(self.screen, (18, 18, 18), (0, panel_y, self.WIN_W, self.WIN_H - panel_y))
        pygame.draw.line(self.screen, (60, 60, 60), (0, panel_y), (self.WIN_W, panel_y), 1)

        # buttons
        self.draw_button(self.btn_play, "Play", enabled=True)
        self.draw_button(self.btn_pause, "Pause", enabled=True)
        self.draw_button(self.btn_stop, "Stop", enabled=True)
        self.draw_button(self.btn_add, "Add", enabled=True)
        self.draw_button(self.btn_save, "Save", enabled=(len(self.sources) > 0))

        # info
        t_cur = self.read_ptr / self.sr
        state = "PAUSED" if self.paused else "PLAYING"
        info1 = f"[Space]Play/Pause  [R]Search@Playhead  [E]JumpToEst  [Tab]Compare   state={state}"
        info2 = f"Play θ={math.degrees(thp):6.1f}° φ={math.degrees(php):+6.1f}° | time={t_cur:7.2f}s | seconds={self.seconds:.2f}"
        info3 = f"WAV: {self.wav_path} | Beamformer: MaxDI(FOA) | Metric topRMS {self.frame_ms:.1f}ms top{self.top_percent:.1f}%"
        self.screen.blit(self.font.render(info1, True, (220, 220, 220)), (10, panel_y + 44))
        self.screen.blit(self.font.render(info2, True, (180, 255, 200)), (10, panel_y + 66))
        self.screen.blit(self.font.render(info3, True, (220, 220, 220)), (10, panel_y + 88))

        # list sources (right side)
        lx, ly = 520, panel_y + 10
        self.screen.blit(self.font.render(f"Saved sources ({len(self.sources)})", True, (230, 230, 230)), (lx, ly))
        ly += 20
        show_n = 6
        for i, s in enumerate(self.sources[-show_n:]):
            thd = math.degrees(s["theta"])
            phd = math.degrees(s["phi"])
            self.screen.blit(
                self.font_small.render(f"- {s['name']}: θ={thd:.1f}°, φ={phd:+.1f}°", True, (210, 210, 210)),
                (lx, ly)
            )
            ly += 16
        if len(self.sources) > show_n:
            self.screen.blit(self.font_small.render("... (older hidden)", True, (150, 150, 150)), (lx, ly))

        # seek bar
        bar_x, bar_y, bar_w, bar_h = 60, self.WIN_H - 22, self.WIN_W - 120, 10
        pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))
        pos = int((self.read_ptr / max(1, self.n_frames)) * bar_w)
        pygame.draw.rect(self.screen, (220, 80, 80), (bar_x, bar_y, pos, bar_h))
        self.screen.blit(self.font.render("Seek", True, (200, 200, 200)), (10, self.WIN_H - 26))

        # toast
        if time.time() < self.toast_until and self.toast:
            self.screen.blit(self.font.render(self.toast, True, (255, 230, 150)), (10, panel_y + 110))

        # input box overlay
        if self.input_mode:
            box = pygame.Rect(120, 160, self.WIN_W - 240, 90)
            pygame.draw.rect(self.screen, (10, 10, 10), box, border_radius=10)
            pygame.draw.rect(self.screen, (120, 120, 120), box, 2, border_radius=10)

            hint = self.font.render(self.input_hint, True, (220, 220, 220))
            self.screen.blit(hint, (box.x + 14, box.y + 12))

            # blinking cursor
            self.flash_t += 1.0 / 60.0
            cursor = "|" if int(self.flash_t * 2) % 2 == 0 else " "
            txt = self.font.render(self.input_text + cursor, True, (255, 255, 255))
            self.screen.blit(txt, (box.x + 14, box.y + 44))

        return pygame.Rect(bar_x, bar_y, bar_w, bar_h)


    def run(self):
        th = threading.Thread(target=self.audio_thread, daemon=True)
        th.start()

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))
        pygame.display.set_caption("FOA MaxDI Beam Search + Sources Save (AmbiX W,Y,Z,X)")
        self.font = pygame.font.SysFont(None, 20)
        self.font_small = pygame.font.SysFont(None, 16)
        self.clock = pygame.time.Clock()

        dragging = False
        seek_drag = False
        RUNNING = True

        while RUNNING:
            seek_rect = self.draw()
            pygame.display.flip()
            self.clock.tick(60)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    RUNNING = False
                    continue

                # ---- input mode: capture typing ----
                if self.input_mode:
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_ESCAPE:
                            self.cancel_name_input()
                        elif ev.key == pygame.K_RETURN:
                            self.confirm_name_input()
                        elif ev.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        else:
                            # accept printable characters
                            ch = ev.unicode
                            if ch and ch.isprintable():
                                # basic length cap
                                if len(self.input_text) < 32:
                                    self.input_text += ch
                    continue  # block other controls while naming

                # ---- normal mode ----
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        RUNNING = False
                    elif ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif ev.key == pygame.K_r:
                        was_paused = self.paused
                        self.paused = True
                        try:
                            self.run_search()
                        except Exception as e:
                            print("[SEARCH ERROR]", e)
                            self.toast_msg(f"Search error: {e}", sec=2.5)
                        self.paused = was_paused
                    elif ev.key == pygame.K_e:
                        if self.est_theta is not None:
                            with self.angle_lock:
                                self.theta = self.est_theta
                                self.phi = self.est_phi
                            self.toast_msg("Jumped to estimate", sec=1.2)
                    elif ev.key == pygame.K_TAB:
                        if self.est_theta is not None:
                            self.compare_mode = not self.compare_mode
                            if self.compare_mode:
                                self.off_theta = 0.0
                                self.off_phi = 0.0

                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 1:
                        # buttons
                        if self.btn_play.collidepoint(ev.pos):
                            self.paused = False
                            continue
                        if self.btn_pause.collidepoint(ev.pos):
                            self.paused = True
                            continue
                        if self.btn_stop.collidepoint(ev.pos):
                            self.paused = True
                            self.read_ptr = 0
                            continue
                        if self.btn_add.collidepoint(ev.pos):
                            self.begin_name_input_for_add()
                            continue
                        if self.btn_save.collidepoint(ev.pos):
                            if len(self.sources) > 0:
                                try:
                                    self.export_sources_txt()
                                except Exception as e:
                                    self.toast_msg(f"Save error: {e}", sec=2.5)
                            else:
                                self.toast_msg("No sources to save", sec=1.5)
                            continue

                        # seek vs drag
                        if seek_rect.collidepoint(ev.pos):
                            seek_drag = True
                        else:
                            mx, my = ev.pos
                            if my < self.MAP_H:
                                dragging = True
                                th0, ph0 = self.xy_to_theta_phi(mx, my)
                                with self.angle_lock:
                                    self.theta, self.phi = th0, ph0
                                self.compare_mode = False

                elif ev.type == pygame.MOUSEBUTTONUP:
                    if ev.button == 1:
                        dragging = False
                        seek_drag = False

                elif ev.type == pygame.MOUSEMOTION:
                    mx, my = ev.pos
                    if dragging and my < self.MAP_H:
                        th0, ph0 = self.xy_to_theta_phi(mx, my)
                        with self.angle_lock:
                            self.theta, self.phi = th0, ph0
                    if seek_drag:
                        r = (mx - seek_rect.x) / max(1, seek_rect.width)
                        r = max(0.0, min(1.0, r))
                        self.read_ptr = int(r * (self.n_frames - 1))

            keys = pygame.key.get_pressed()

            if self.compare_mode and self.est_theta is not None:
                step = math.radians(5)
                if keys[pygame.K_a]:
                    self.off_theta -= step
                if keys[pygame.K_d]:
                    self.off_theta += step
                if keys[pygame.K_w]:
                    self.off_phi += step
                if keys[pygame.K_s]:
                    self.off_phi -= step
                self.off_theta, self.off_phi = clamp_angles(self.off_theta, self.off_phi)

            else:
                with self.angle_lock:
                    if keys[pygame.K_LEFT]:
                        self.theta -= self.move_step
                    if keys[pygame.K_RIGHT]:
                        self.theta += self.move_step
                    if keys[pygame.K_UP]:
                        self.phi += self.move_step
                    if keys[pygame.K_DOWN]:
                        self.phi -= self.move_step
                    self.theta, self.phi = clamp_angles(self.theta, self.phi)

        self.stop_flag = True
        pygame.quit()
        th.join(timeout=1.0)
        print("App closed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, default=None, help="override WAV_PATH if set")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    ap.add_argument("--frame_ms", type=float, default=DEFAULT_FRAME_MS)
    ap.add_argument("--top_percent", type=float, default=DEFAULT_TOP_PERCENT)
    ap.add_argument("--out", type=str, default=OUTPUT_TXT_DEFAULT, help="output txt filename")
    args = ap.parse_args()

    wav = args.wav if args.wav else WAV_PATH
    app = RealtimeBeamApp(
        wav_path=wav,
        seconds=args.seconds,
        block_size=args.block,
        move_step_deg=3.0,
        frame_ms=args.frame_ms,
        top_percent=args.top_percent,
        output_txt=args.out
    )
    app.run()


if __name__ == "__main__":
    main()
