"""
cognitive_widget.py  —  proactive eye-strain build
═══════════════════════════════════════════════════
New in this version
───────────────────
EyeStrain model (3 components, all computed proactively):

  1. CiliaryStrain ODE
       Ciliary muscles control lens accommodation (near ↔ far focus).
       They fatigue during sustained near-focus work (READING / TYPING /
       SCROLLING / CORRECTIVE) and recover during IDLE / RECOVERY / break.
         dF/dt = +K_FAT * activity_weight   (fatigue accumulation)
         dF/dt = -K_REC * F                 (exponential recovery)
       F in [0, 100].  F > 70 triggers a proactive break.

  2. BlinkSuppression proxy
       Normal blink rate ~15-17 / min; falls to ~4-6 during screen focus.
       Proxy: sustained absence of keyboard events + near-focus state
       means eyes are open, dry and staring.  Score rises linearly with
       stare time, resets on any key event.

  3. SaccadicTremor index
       Online Welford variance of mouse X+Y position.
       High variance = large fixation jitter = tired saccadic system.
       Normalised to [0, 100] against a 500 px^2 reference.

Composite strain score = weighted average of all three + rhodopsin.

TwentyTwenty scheduler (20-20-20 rule, proactive)
──────────────────────────────────────────────────
  Every 20 min of near-focus → 20-second far-focus break (widget hides).
  If CiliaryStrain > 70 OR composite strain > 75, break fires early.
  After break: ciliary ODE steps forward in recovery mode for the 20 s.
  "20/20 remaining" displayed as a depleting arc (MM:SS countdown).

Overlay on every visible frame:
  • Thin arc (radius W*0.46) depleting CW, colour green -> amber -> red.
  • Centre text: MM:SS remaining until next break.
  • Bottom stripe: composite strain fill.
  • Top-right dot: ciliary fatigue level.
  • When < 60 s remaining, arc pulses.

All previous fixes retained:
  t_var mod 2pi (LRU cache cycles), Welford M2 fix, CSV daemon thread,
  t_b key wraps mod TCYCLE, optomotor flee, 4-Hz idle while hidden.
"""

import pygame
import pygame._sdl2.video as sdl2_video
import math, sys, os, time, csv, threading
from collections import deque, OrderedDict
from pynput import mouse, keyboard

try:
    from AppKit import NSApp, NSApplicationActivationPolicyAccessory
    _MAC = True
except ImportError:
    _MAC = False

os.environ["SDL_VIDEO_WINDOW_POS"] = "20,20"
pygame.init()
pygame.font.init()

_info    = pygame.display.Info()
SCREEN_W = _info.current_w
SCREEN_H = _info.current_h
W = H    = int(SCREEN_W * 0.15)
CX, CY   = W >> 1, H >> 1
SCALE    = W / 300.0
FOCAL    = W * 1.6
_TWO_PI  = 2.0 * math.pi

screen = pygame.display.set_mode(
    (W, H), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)
window = sdl2_video.Window.from_display_module()

if _MAC:
    try:
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        NSApp.windows()[0].setLevel_(3)
    except Exception:
        pass

# ── Fonts ─────────────────────────────────────────────────────────────────
_FONT_SM = pygame.font.SysFont("monospace", max(9,  int(W * 0.10)), bold=True)
_FONT_XS = pygame.font.SysFont("monospace", max(7,  int(W * 0.07)))

# ── Colors ────────────────────────────────────────────────────────────────
C_IDLE     = (35,   0,  50)
C_FOCUS    = (255, 120,   0)
C_STRESS   = (0,  255, 180)
C_RECOVERY = (120,  30,   5)
C_SCROLL   = (180,  60, 220)
BLACK      = (0,    0,   0)

# ── Trig LUT ──────────────────────────────────────────────────────────────
_LUT   = 7200
_LUT_K = _LUT / _TWO_PI
_GOLD  = math.pi * (3.0 - math.sqrt(5.0))
_FIBPH = math.pi * (1.0 + math.sqrt(5.0))
_COS   = [math.cos(i * _TWO_PI / _LUT) for i in range(_LUT)]
_SIN   = [math.sin(i * _TWO_PI / _LUT) for i in range(_LUT)]
def _cos(r): return _COS[int(r * _LUT_K) % _LUT]
def _sin(r): return _SIN[int(r * _LUT_K) % _LUT]

# ── 20-20-20 / break config ───────────────────────────────────────────────
NEAR_FOCUS_BUDGET   = 20 * 60.0   # seconds of near-focus before break
BREAK_DURATION      = 20.0        # seconds of far-focus rest
RECOVERY_CHECK_SECS = 2.0         # RH / strain check interval while hidden
CILIARY_BREAK_EARLY = 70.0        # CiliaryStrain early-break threshold
STRAIN_BREAK_EARLY  = 75.0        # composite strain early-break threshold

# ── Optomotor reflex config ───────────────────────────────────────────────
OPTOMOTOR_R       = W * 2.0
OPTOMOTOR_DWELL_T = 0.35
OPTOMOTOR_COOL    = 2.5
_ANCHOR_L = 20.0
_ANCHOR_R = float(SCREEN_W - W - 20)

# ── State ─────────────────────────────────────────────────────────────────
class S:
    cr=35.0; cg=0.0; cb=50.0
    tr=35.0; tg=0.0; tb=50.0
    p_n=50.0;  t_n=50.0
    p_r=5.0;   t_r=5.0
    p_wx=10.0; t_wx=10.0
    p_wy=10.0; t_wy=10.0
    p_sp=0.1;  t_sp=0.1
    lerp=0.05
    mode=0; t_mode=0
    win_x=20.0; tgt_x=20.0
    win_y=20.0; tgt_y=20.0
    left=True;  swap_t=time.time()
    tier=0; stable=0
    state="BOOTING"
    t_var=0.0
    hidden=False; hide_until=0.0; hide_reason=""
    opto_last_t=0.0; mouse_in_t=0.0; mouse_in=False

# ── Photo (rhodopsin) ODE ─────────────────────────────────────────────────
class Photo:
    BC=100.0; RL=80.0; CI=70.0; RH=100.0
    EFF=1.0; pred_f=100.0
    K_BCO1=0.00015; K_BCR=0.00004; K_RPE65=0.0025
    K_BIND=0.007;   K_BL=0.0005;   K_SCR=0.0008; K_DK=0.0018
    @classmethod
    def step(cls, I, Sc):
        D  = max(0.0, 100.0 - cls.RH) / 100.0
        In = min(I  / 3000.0, 1.0)
        Sn = min(Sc / 200.0,  1.0)
        dBC = -cls.K_BCO1*In*cls.BC + cls.K_BCR*(100.0-cls.BC)
        dRL =  cls.K_BCO1*In*cls.BC - cls.K_RPE65*cls.RL*D
        dCI =  cls.K_RPE65*cls.RL*D - cls.K_BIND*cls.CI*D
        dRH = (cls.K_BIND*cls.CI*D - cls.K_BL*In*cls.RH
               - cls.K_SCR*Sn*cls.RH + cls.K_DK*D*100.0)
        cls.BC = max(0.0, min(100.0, cls.BC+dBC))
        cls.RL = max(0.0, min(100.0, cls.RL+dRL))
        cls.CI = max(0.0, min(100.0, cls.CI+dCI))
        cls.RH = max(0.0, min(100.0, cls.RH+dRH))
        cls.EFF = ((cls.BC*cls.RL*cls.CI*cls.RH) / (100.0**4)) ** 0.25

# ── EyeStrain model ───────────────────────────────────────────────────────
# Near-focus activity weight per cognitive state (0=rest, 1=max effort)
_CIL_W = {
    "READING"    : 1.00,
    "TYPING"     : 0.85,
    "SCROLLING"  : 0.70,
    "CORRECTIVE" : 0.90,
    "IDLE"       : 0.10,
    "RECOVERY"   : 0.00,
    "BOOTING"    : 0.00,
}
_K_CIL_FAT = 0.08   # fatigue rate at weight=1 (per second)
_K_CIL_REC = 0.04   # exponential recovery constant (per second)

class EyeStrain:
    ciliary     = 0.0   # accommodation / ciliary muscle fatigue [0-100]
    blink_supp  = 0.0   # blink suppression proxy [0-100]
    tremor      = 0.0   # saccadic tremor index [0-100]
    composite   = 0.0   # weighted composite [0-100]
    _stare_secs = 0.0   # continuous stare accumulator

    @classmethod
    def step(cls, state_name, vx, vy, dt, last_key_age):
        w = _CIL_W.get(state_name, 0.0)

        # 1. Ciliary ODE
        if w > 0.01:
            cls.ciliary = min(100.0, cls.ciliary + _K_CIL_FAT * w * dt)
        else:
            cls.ciliary = max(0.0,   cls.ciliary - _K_CIL_REC * cls.ciliary * dt)

        # 2. Blink suppression: accrues when no key for >2 s in near-focus state
        if last_key_age > 2.0 and w >= 0.70:
            cls._stare_secs += dt
        else:
            cls._stare_secs = max(0.0, cls._stare_secs - dt * 4.0)
        cls.blink_supp = min(100.0, cls._stare_secs / 60.0 * 100.0)

        # 3. Saccadic tremor from mouse position variance
        cls.tremor = min(100.0, (vx + vy) * 0.5 / 500.0 * 100.0)

        # 4. Composite (includes rhodopsin depletion)
        cls.composite = (
            0.35 * cls.ciliary
          + 0.25 * cls.blink_supp
          + 0.20 * cls.tremor
          + 0.20 * (100.0 - Photo.RH)
        )

    @classmethod
    def recover_step(cls, dt):
        """Full recovery mode — called while widget is hidden."""
        cls.ciliary     = max(0.0, cls.ciliary    - _K_CIL_REC * cls.ciliary * dt * 3.0)
        cls._stare_secs = max(0.0, cls._stare_secs - dt * 8.0)
        cls.blink_supp  = min(100.0, cls._stare_secs / 60.0 * 100.0)
        cls.composite   = (
            0.35 * cls.ciliary
          + 0.25 * cls.blink_supp
          + 0.20 * cls.tremor
          + 0.20 * (100.0 - Photo.RH)
        )

# ── 20-20-20 Scheduler ────────────────────────────────────────────────────
class TwentyTwenty:
    near_secs    = 0.0
    budget_secs  = NEAR_FOCUS_BUDGET
    breaks_taken = 0
    last_break_t = time.time()
    early_fired  = False

    @classmethod
    def remaining(cls):
        return max(0.0, cls.budget_secs - cls.near_secs)

    @classmethod
    def fraction_used(cls):
        return min(1.0, cls.near_secs / cls.budget_secs)

    @classmethod
    def accrue(cls, state_name, dt):
        w = _CIL_W.get(state_name, 0.0)
        if w > 0.05:
            cls.near_secs += dt * w

    @classmethod
    def should_break(cls):
        if cls.near_secs >= cls.budget_secs:
            return True, "20-min near-focus budget"
        if EyeStrain.ciliary >= CILIARY_BREAK_EARLY and not cls.early_fired:
            return True, f"ciliary strain {EyeStrain.ciliary:.0f}%"
        if EyeStrain.composite >= STRAIN_BREAK_EARLY and not cls.early_fired:
            return True, f"composite strain {EyeStrain.composite:.0f}%"
        return False, ""

    @classmethod
    def reset_after_break(cls):
        cls.near_secs    = 0.0
        cls.early_fired  = False
        cls.breaks_taken += 1
        cls.last_break_t  = time.time()

# ── Geometry cache ────────────────────────────────────────────────────────
_BKT_N=5; _BKT_A=10
class Geo:
    _k2=None; _k3=None; _n2=0; _n3=0
    geo2=[]; geo3=[]
    @classmethod
    def _key(cls,n,wx,wy):
        return (round(n/_BKT_N)*_BKT_N,
                round(wx/_BKT_A)*_BKT_A,
                round(wy/_BKT_A)*_BKT_A)
    @classmethod
    def refresh_2d(cls,n,wx,wy):
        ni=int(n); k=cls._key(ni,wx,wy)
        if k==cls._k2 and ni==cls._n2: return
        cls._k2=k; cls._n2=ni
        cls.geo2=[(i*_GOLD, float(i*500//max(ni,1)%500)) for i in range(ni)]
    @classmethod
    def refresh_3d(cls,n):
        ni=int(n)
        if ni==cls._n3: return
        cls._n3=ni
        pts=[]
        for i in range(ni):
            phi=math.acos(max(-1.0,min(1.0,1.0-2.0*(i+0.5)/ni)))
            th=_FIBPH*i
            pts.append((math.sin(phi)*math.cos(th),
                        math.sin(phi)*math.sin(th),
                        math.cos(phi)))
        pts.sort(key=lambda p:p[2])
        cls.geo3=pts

# ── Surface frame cache (LRU-128) ─────────────────────────────────────────
_FCACHE  = OrderedDict()
_FCAP    = 128
_T_QUANT = 0.04
_TCYCLE  = int(_TWO_PI / _T_QUANT) + 1    # ~158 — keys cycle

def _make_surface(key, t, ax, ay, br, fr, fg, fb, mode):
    surf = pygame.Surface((W, H))
    surf.fill(BLACK)
    _dc  = pygame.draw.circle
    sr   = W * 0.38

    def _d2(n_cap=0):
        geo=Geo.geo2; n=min(n_cap if n_cap>0 else len(geo),len(geo))
        for i in range(n):
            ba,bd=geo[i]
            depth=(bd-t*200.0)%500.0
            if depth<0.1: depth=0.1
            sc=(300.0/depth)*SCALE
            angle=ba+t*0.5
            x=CX+int(_cos(angle)*ax*sc)
            y=CY+int(_sin(angle)*ay*sc)
            if 0<=x<=W and 0<=y<=H:
                fd=1.0-depth/500.0
                _dc(surf,(int(fr*fd),int(fg*fd),int(fb*fd)),
                    (x,y),max(2,int(br*sc)))

    def _d3(n_cap=0):
        geo=Geo.geo3; n=min(n_cap if n_cap>0 else len(geo),len(geo))
        dist=2.8
        cy_=_cos(t*0.55); sy_=_sin(t*0.55)
        cx_=_cos(t*0.18); sx_=_sin(t*0.18)
        for i in range(n):
            nx,ny,nz=geo[i]
            rx=nx*cy_+nz*sy_; rz=-nx*sy_+nz*cy_; ry=ny
            ry2=ry*cx_-rz*sx_; rz2=ry*sx_+rz*cx_; rx2=rx
            zc=rz2+dist
            if zc<0.05: continue
            proj=FOCAL/zc
            px=int(CX+rx2*proj*sr/W)
            py=int(CY+ry2*proj*sr/W)
            if 0<=px<=W and 0<=py<=H:
                fd=max(0.15,(rz2+1.5)/3.0)
                sz=max(2,int(sr*proj*0.012*SCALE))
                _dc(surf,(int(fr*fd),int(fg*fd),int(fb*fd)),(px,py),sz)

    if mode==0:   _d2()
    elif mode==1: _d3()
    else:
        half=max(4,int(S.p_n)>>1)
        _d2(half); _d3(half)
    return surf

def get_frame(t, ax, ay, br, fr, fg, fb, mode):
    t_b  = int(t / _T_QUANT) % _TCYCLE
    n_b  = round(S.p_n / _BKT_N) * _BKT_N
    wx_b = round(ax  / _BKT_A) * _BKT_A
    wy_b = round(ay  / _BKT_A) * _BKT_A
    key  = (mode, n_b, wx_b, wy_b,
            int(fr)&0xF0, int(fg)&0xF0, int(fb)&0xF0, t_b)
    if key in _FCACHE:
        _FCACHE.move_to_end(key)
        return _FCACHE[key]
    Geo.refresh_2d(S.p_n, ax, ay)
    Geo.refresh_3d(S.p_n)
    surf = _make_surface(key, t_b * _T_QUANT, ax, ay, br, fr, fg, fb, mode)
    _FCACHE[key] = surf
    if len(_FCACHE) > _FCAP:
        _FCACHE.popitem(last=False)
    return surf

# ── Overlay renderer ──────────────────────────────────────────────────────
_ARC_R   = int(W * 0.46)
_ARC_T   = max(3, int(W * 0.025))
_ARC_REF = pygame.Rect(CX-_ARC_R, CY-_ARC_R, _ARC_R*2, _ARC_R*2)

def _arc_color(frac):
    if frac < 0.6:
        t = frac / 0.6
        return (int(60+195*t), int(220-20*t), int(80-60*t))
    else:
        t = (frac - 0.6) / 0.4
        return (255, int(200-150*t), int(20-20*t))

def _draw_overlay(surface, now):
    frac = TwentyTwenty.fraction_used()
    rem  = TwentyTwenty.remaining()

    pulse = (0.55 + 0.45 * math.sin(now * 6.0)) if rem < 60.0 else 1.0
    col   = _arc_color(frac)
    ca    = tuple(int(c * pulse) for c in col)

    # Background ring (dim full circle)
    pygame.draw.circle(surface, (30, 30, 30), (CX, CY), _ARC_R, _ARC_T)

    # Depleting arc (CW from top)
    if frac > 0.01:
        arc_rad = frac * _TWO_PI
        start_a = math.pi * 0.5 - arc_rad
        end_a   = math.pi * 0.5
        pygame.draw.arc(surface, ca, _ARC_REF, start_a, end_a, _ARC_T + 1)

    # Countdown text MM:SS
    mins = int(rem) // 60
    secs = int(rem) % 60
    txt  = _FONT_SM.render(f"{mins:02d}:{secs:02d}", True, ca)
    tw, th = txt.get_size()
    surface.blit(txt, (CX - tw//2, CY - th//2))

    # Bottom composite-strain fill bar
    bar_w = int(EyeStrain.composite / 100.0 * W)
    bar_y = H - max(3, int(H * 0.04))
    if bar_w > 0:
        pygame.draw.rect(surface, ca, (0, bar_y, bar_w, H - bar_y))

    # Top-right ciliary fatigue dot
    dot_r = max(2, int(EyeStrain.ciliary / 100.0 * W * 0.06))
    pygame.draw.circle(surface, ca, (W - dot_r - 3, dot_r + 3), dot_r)

# ── Input listeners ───────────────────────────────────────────────────────
_mx = _my = 0
_last_key   = 0.0
_scroll_acc = 0.0

def _on_move(x, y):        global _mx, _my; _mx=x; _my=y
def _on_click(x,y,b,pr):   global _mx, _my; _mx=x; _my=y
def _on_scroll(x,y,dx,dy): global _scroll_acc; _scroll_acc+=abs(dy)
def _on_press(_k):          global _last_key;  _last_key=time.time()

_kb = keyboard.Listener(on_press=_on_press)
_ml = mouse.Listener(on_move=_on_move, on_click=_on_click, on_scroll=_on_scroll)
_kb.start(); _ml.start()

# ── Welford online variance ───────────────────────────────────────────────
class Welford:
    __slots__ = ('n','mean','M2')
    def __init__(self): self.n=0; self.mean=0.0; self.M2=0.0
    def update(self, x):
        self.n += 1; d = x - self.mean; self.mean += d / self.n
        self.M2 += d * (x - self.mean)
    def var(self): return self.M2 / self.n if self.n > 1 else 0.0
    def reset(self): self.n=0; self.mean=0.0; self.M2=0.0

_wfx = Welford(); _wfy = Welford()
_prev_mx = _prev_my = 0
_dist_acc = 0.0; _samp_n = 0

# ── Histories ─────────────────────────────────────────────────────────────
speed_hist  = deque(maxlen=20)
rhod_hist   = deque(maxlen=20)
scroll_hist = deque(maxlen=40)

# ── CSV ───────────────────────────────────────────────────────────────────
_LOG      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cognitive_log.csv")
_LOG_EVERY= 5.0; _log_buf=[]; _last_fl=time.time()
_lf=open(_LOG,"w",newline="",buffering=8192); _lw=csv.writer(_lf)
_lw.writerow(["ts","state","tier","mode",
              "BC","RL","CI","RH","efficacy","pred_fatigue",
              "mouse_speed","scroll_stress","var_x","var_y","side","hidden",
              "ciliary","blink_supp","tremor","strain",
              "near_secs","break_rem","breaks"])

def _flush_csv(rows): _lw.writerows(rows); _lf.flush()
def _maybe_flush(now):
    global _last_fl
    if now - _last_fl >= _LOG_EVERY and _log_buf:
        rows=_log_buf.copy(); _log_buf.clear(); _last_fl=now
        threading.Thread(target=_flush_csv, args=(rows,), daemon=True).start()

# ── Linear regression (5-tier) ────────────────────────────────────────────
def _predict(data, future=0, tier=0):
    nt=len(data)
    if nt<4: return data[-1] if nt else 0.0
    s=list(data)[-4:] if tier>=4 else list(data)[::tier+1]
    n=len(s); sx=n*(n-1)//2; sx2=n*(n-1)*(2*n-1)//6; sy=sxy=0.0
    for i,v in enumerate(s): sy+=v; sxy+=i*v
    d=n*sx2-sx*sx
    if d==0: return sy/n
    slope=(n*sxy-sx*sy)/d
    return max(0.0, slope*(n+future)+(sy-slope*sx)/n)

# ── Window hide / show ────────────────────────────────────────────────────
_OFF_X = SCREEN_W + 200

def _hide_window(reason, duration):
    if S.hidden: return
    S.hidden=True; S.hide_until=time.time()+duration; S.hide_reason=reason
    window.position=(_OFF_X, 0)
    print(f"[widget] HIDDEN {duration:.0f}s — {reason}")

def _show_window():
    S.hidden=False; window.position=(int(S.win_x), int(S.win_y))
    print(f"[widget] VISIBLE — cil={EyeStrain.ciliary:.1f}  strain={EyeStrain.composite:.1f}")

# ── Biometrics (4 Hz) ─────────────────────────────────────────────────────
_last_eval_t = time.time()

def compute(now):
    global _scroll_acc, _dist_acc, _samp_n, _prev_mx, _prev_my, _last_eval_t

    dt       = max(0.001, now - _last_eval_t)
    _last_eval_t = now

    vx=_wfx.var(); vy=_wfy.var(); speed=_dist_acc
    _wfx.reset(); _wfy.reset(); _dist_acc=0.0; _samp_n=0

    speed_hist.append(speed)
    pred_s=_predict(list(speed_hist),0,S.tier)
    err=abs(pred_s-speed)
    if err>500.0: S.tier=0; S.stable=0
    else:
        S.stable+=1
        if S.stable>8 and S.tier<4: S.tier+=1; S.stable=0

    sc=_scroll_acc*4.0; _scroll_acc=0.0
    scroll_hist.append(sc)
    pred_sc=_predict(list(scroll_hist),5,S.tier)

    Photo.step(pred_s, pred_sc)
    rhod_hist.append(Photo.RH)
    Photo.pred_f=min(100.0,_predict(list(rhod_hist),15,S.tier))

    E=Photo.EFF; ma=W*0.4*(0.5+0.5*(1.0-E))
    last_key_age=now-_last_key
    rh_low=Photo.RH<35.0 or Photo.CI<25.0
    fatigue_pred=Photo.pred_f<40.0
    scroll_active=pred_sc>30.0

    # ── Cognitive state ───────────────────────────────────────────────────
    if rh_low or fatigue_pred or (S.state=="RECOVERY" and Photo.RH<80.0):
        S.state="RECOVERY"
        S.tr,S.tg,S.tb=C_RECOVERY
        S.t_n,S.t_r=25.0,12.0
        S.t_wx,S.t_wy,S.t_sp=ma*0.6,ma*0.6,-0.03
        S.t_mode=1; S.lerp=0.02; S.tgt_y=20.0
    elif scroll_active:
        S.state="SCROLLING"
        S.tr,S.tg,S.tb=C_SCROLL
        sf=min(1.0,pred_sc/150.0)
        S.t_n=35.0+35.0*sf; S.t_r=6.0+8.0*sf; S.t_sp=0.25+0.75*sf
        S.t_wx=ma*(0.4+0.6*sf); S.t_wy=ma*(0.4+0.6*sf)
        S.t_mode=1; S.lerp=0.15; S.tgt_y=SCREEN_H*0.35
    elif last_key_age<1.5:
        S.state="TYPING"
        S.tr,S.tg,S.tb=C_IDLE
        S.t_n,S.t_r=15.0,3.0
        S.t_wx,S.t_wy,S.t_sp=ma*0.15,ma*0.15,0.04
        S.t_mode=0; S.lerp=0.05; S.tgt_y=20.0
    elif vx>vy*3 and 150<=pred_s<1500:
        S.state="READING"
        S.tr,S.tg,S.tb=C_FOCUS
        S.t_n,S.t_r=35.0,5.0
        S.t_wx,S.t_wy,S.t_sp=ma*0.9,ma*0.15,0.35
        S.t_mode=0; S.lerp=0.1; S.tgt_y=20.0
    elif pred_s>=1500:
        S.state="CORRECTIVE"
        S.tr,S.tg,S.tb=C_STRESS
        af=min(1.0,pred_s/3000.0); da=ma*0.5+ma*0.5*af
        S.t_n,S.t_r=60.0+30.0*af,4.0
        S.t_wx,S.t_wy,S.t_sp=da,da,1.2+af
        S.t_mode=2; S.lerp=0.2; S.tgt_y=20.0
    else:
        S.state="IDLE"
        S.tr,S.tg,S.tb=C_IDLE
        S.t_n,S.t_r=30.0,5.0
        S.t_wx,S.t_wy,S.t_sp=ma*0.35,ma*0.35,0.12
        S.t_mode=2; S.lerp=0.05; S.tgt_y=20.0

    S.mode=S.t_mode

    # ── Eye strain + 20-20-20 ─────────────────────────────────────────────
    EyeStrain.step(S.state, vx, vy, dt, last_key_age)
    TwentyTwenty.accrue(S.state, dt)

    # ── Proactive break decision ──────────────────────────────────────────
    should, reason = TwentyTwenty.should_break()
    if should or rh_low or fatigue_pred:
        full_reason = (reason if should
                       else ("fatigue predicted" if fatigue_pred else "rhodopsin low"))
        TwentyTwenty.early_fired = should and not (rh_low or fatigue_pred)
        _hide_window(full_reason, BREAK_DURATION)

    # Saccadic swap (visible only)
    if not S.hidden:
        if (now-S.swap_t>45.0 or
            (pred_sc>80 and now-S.swap_t>8.0) or
            (pred_s>4000 and now-S.swap_t>10.0)):
            S.left=not S.left; S.swap_t=now
            S.tgt_x=20.0 if S.left else float(SCREEN_W-W-20)

    _log_buf.append((
        f"{now:.3f}",S.state,S.tier,S.mode,
        f"{Photo.BC:.2f}",f"{Photo.RL:.2f}",f"{Photo.CI:.2f}",
        f"{Photo.RH:.2f}",f"{Photo.EFF:.4f}",f"{Photo.pred_f:.2f}",
        f"{speed:.1f}",f"{sc:.1f}",f"{vx:.1f}",f"{vy:.1f}",
        "L" if S.left else "R","1" if S.hidden else "0",
        f"{EyeStrain.ciliary:.2f}",f"{EyeStrain.blink_supp:.2f}",
        f"{EyeStrain.tremor:.2f}",f"{EyeStrain.composite:.2f}",
        f"{TwentyTwenty.near_secs:.1f}",f"{TwentyTwenty.remaining():.1f}",
        TwentyTwenty.breaks_taken,
    ))

# ── Optomotor flee ────────────────────────────────────────────────────────
def _optomotor_flee(now):
    if S.hidden: return
    if now-S.opto_last_t<OPTOMOTOR_COOL: return
    wx=S.win_x; wy=S.win_y
    wcx=wx+W*0.5; wcy=wy+H*0.5
    dx=_mx-wcx; dy=_my-wcy
    dist=math.sqrt(dx*dx+dy*dy)
    in_rect=(wx<=_mx<=wx+W) and (wy<=_my<=wy+H)
    if in_rect:
        if not S.mouse_in: S.mouse_in=True; S.mouse_in_t=now
    else:
        S.mouse_in=False
    proximity_trigger=dist<OPTOMOTOR_R
    dwell_trigger=S.mouse_in and (now-S.mouse_in_t>=OPTOMOTOR_DWELL_T)
    if not (proximity_trigger or dwell_trigger): return
    new_x=_ANCHOR_R if S.win_x<SCREEN_W*0.5 else _ANCHOR_L
    if abs(_mx-(new_x+W*0.5))<OPTOMOTOR_R:
        mid_y=SCREEN_H*0.5
        S.tgt_y=(SCREEN_H*0.65) if _my<mid_y else 20.0
    else:
        S.tgt_x=new_x; S.left=(new_x==_ANCHOR_L)
    S.opto_last_t=now; S.mouse_in=False
    print(f"[optomotor] flee → x={S.tgt_x:.0f}  dist={dist:.0f} dwell={dwell_trigger}")

# ── Main loop ─────────────────────────────────────────────────────────────
clock      = pygame.time.Clock()
last_eval  = time.time()
_prev_mx   = _mx; _prev_my=_my

print(f"[widget] proactive 20-20-20 | ciliary ODE | blink proxy | optomotor | log→{_LOG}")
print(f"[widget] budget={NEAR_FOCUS_BUDGET/60:.0f} min | break={BREAK_DURATION:.0f} s | "
      f"ciliary-thresh={CILIARY_BREAK_EARLY:.0f} | strain-thresh={STRAIN_BREAK_EARLY:.0f}")

running=True
while running:
    now=time.time()

    for ev in pygame.event.get():
        if ev.type==pygame.QUIT: running=False
        elif ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE: running=False

    # ── Hidden / break mode ───────────────────────────────────────────────
    if S.hidden:
        if now>=S.hide_until:
            if Photo.RH>=70.0 and EyeStrain.ciliary<50.0:
                TwentyTwenty.reset_after_break()
                _show_window()
            else:
                S.hide_until=now+10.0
                print(f"[widget] extending — RH={Photo.RH:.1f} cil={EyeStrain.ciliary:.1f}")

        dx=_mx-_prev_mx; dy=_my-_prev_my
        _dist_acc+=math.sqrt(dx*dx+dy*dy)
        _wfx.update(_mx); _wfy.update(_my)
        _prev_mx=_mx; _prev_my=_my
        EyeStrain.recover_step(RECOVERY_CHECK_SECS)

        if now-last_eval>=RECOVERY_CHECK_SECS:
            compute(now); last_eval=now

        _maybe_flush(now)
        clock.tick(4)
        continue

    # ── Visible path ──────────────────────────────────────────────────────
    dx=_mx-_prev_mx; dy=_my-_prev_my
    _dist_acc+=math.sqrt(dx*dx+dy*dy)
    _wfx.update(_mx); _wfy.update(_my)
    _prev_mx=_mx; _prev_my=_my

    if now-last_eval>=0.25:
        compute(now); last_eval=now

    _maybe_flush(now)
    _optomotor_flee(now)

    S.win_x+=(S.tgt_x-S.win_x)*0.05
    S.win_y+=(S.tgt_y-S.win_y)*0.05
    window.position=(int(S.win_x),int(S.win_y))

    ls=S.lerp
    S.cr+=(S.tr-S.cr)*ls; S.cg+=(S.tg-S.cg)*ls; S.cb+=(S.tb-S.cb)*ls
    S.p_n +=(S.t_n -S.p_n )*ls; S.p_r +=(S.t_r -S.p_r )*ls
    S.p_wx+=(S.t_wx-S.p_wx)*ls; S.p_wy+=(S.t_wy-S.p_wy)*ls
    S.p_sp+=(S.t_sp-S.p_sp)*ls
    S.t_var=(S.t_var+0.02*S.p_sp)%_TWO_PI

    # Blit cached base frame then draw overlay on a copy (cache untouched)
    base=get_frame(S.t_var,S.p_wx,S.p_wy,S.p_r,S.cr,S.cg,S.cb,S.mode)
    frame=base.copy()
    _draw_overlay(frame, now)

    screen.blit(frame,(0,0))
    pygame.display.flip()
    clock.tick(60)

# ── Shutdown ──────────────────────────────────────────────────────────────
if _log_buf: _lw.writerows(_log_buf)
_lf.close(); _kb.stop(); _ml.stop()
pygame.quit(); sys.exit()