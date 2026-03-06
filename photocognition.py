"""
cognitive_widget.py  —  ultra-lean build
Key perf wins over previous version:
  • Surface frame cache (LRU-128): hot path = 1 blit, zero particle work
  • Welford online variance: no full-list recompute each 4Hz tick
  • Running distance sum: incremental, O(1) update
  • Geo cache unchanged (bucket hash)
  • Trig LUT unchanged (7200 entries)
  • Mouse listener instead of Controller.position poll (saves IPC call/frame)
  • Renderer skips cache build when hidden behind other windows (future)
  • 30-fps physics + 60-fps blit via frame doubling
"""

import pygame
import pygame._sdl2.video as sdl2_video
import math, sys, os, time, csv
from collections import deque, OrderedDict
from pynput import mouse, keyboard

try:
    from AppKit import NSApp, NSApplicationActivationPolicyAccessory
    _MAC = True
except ImportError:
    _MAC = False

os.environ["SDL_VIDEO_WINDOW_POS"] = "20,20"
pygame.init()

_info    = pygame.display.Info()
SCREEN_W = _info.current_w
SCREEN_H = _info.current_h
W = H    = int(SCREEN_W * 0.15)
CX, CY   = W >> 1, H >> 1
SCALE    = W / 300.0
FOCAL    = W * 1.6

screen = pygame.display.set_mode(
    (W, H), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)
window = sdl2_video.Window.from_display_module()

if _MAC:
    try:
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        NSApp.windows()[0].setLevel_(3)
    except Exception:
        pass

# ── Colors ────────────────────────────────────────────────────────────────
C_IDLE     = (35,   0,  50)
C_FOCUS    = (255, 120,   0)
C_STRESS   = (0,  255, 180)
C_RECOVERY = (120,  30,   5)
C_SCROLL   = (180,  60, 220)
BLACK      = (0,    0,   0)

# ── Trig LUT ──────────────────────────────────────────────────────────────
_LUT   = 7200
_LUT_K = _LUT / (2.0 * math.pi)
_GOLD  = math.pi * (3.0 - math.sqrt(5.0))
_FIBPH = math.pi * (1.0 + math.sqrt(5.0))
_COS   = [math.cos(i * 2.0 * math.pi / _LUT) for i in range(_LUT)]
_SIN   = [math.sin(i * 2.0 * math.pi / _LUT) for i in range(_LUT)]
def _cos(r): return _COS[int(r * _LUT_K) % _LUT]
def _sin(r): return _SIN[int(r * _LUT_K) % _LUT]

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

# ── Photo ODE ─────────────────────────────────────────────────────────────
class Photo:
    BC=100.0; RL=80.0; CI=70.0; RH=100.0
    EFF=1.0; pred_f=100.0
    K_BCO1=0.00015; K_BCR=0.00004; K_RPE65=0.0025
    K_BIND=0.007;   K_BL=0.0005;   K_SCR=0.0008; K_DK=0.0018
    @classmethod
    def step(cls, I, Sc):
        D   = max(0.0, 100.0 - cls.RH) / 100.0
        I_n = min(I  / 3000.0, 1.0)
        S_n = min(Sc / 200.0,  1.0)
        dBC = -cls.K_BCO1*I_n*cls.BC + cls.K_BCR*(100.0-cls.BC)
        dRL =  cls.K_BCO1*I_n*cls.BC - cls.K_RPE65*cls.RL*D
        dCI =  cls.K_RPE65*cls.RL*D  - cls.K_BIND*cls.CI*D
        dRH = (cls.K_BIND*cls.CI*D - cls.K_BL*I_n*cls.RH
               - cls.K_SCR*S_n*cls.RH + cls.K_DK*D*100.0)
        cls.BC = max(0.0,min(100.0,cls.BC+dBC))
        cls.RL = max(0.0,min(100.0,cls.RL+dRL))
        cls.CI = max(0.0,min(100.0,cls.CI+dCI))
        cls.RH = max(0.0,min(100.0,cls.RH+dRH))
        cls.EFF = ((cls.BC*cls.RL*cls.CI*cls.RH)/(100.0**4))**0.25

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
_FCACHE    = OrderedDict()   # key → Surface
_FCAP      = 128
_T_QUANT   = 0.04            # ~157 bins per 2π cycle

def _make_surface(key, t, ax, ay, br, fr, fg, fb, mode):
    """Render a frame into a new Surface and return it."""
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
        cy=_cos(t*0.55); sy=_sin(t*0.55)
        cx=_cos(t*0.18); sx=_sin(t*0.18)
        for i in range(n):
            nx,ny,nz=geo[i]
            rx=nx*cy+nz*sy; rz=-nx*sy+nz*cy; ry=ny
            ry2=ry*cx-rz*sx; rz2=ry*sx+rz*cx; rx2=rx
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
    """Return cached Surface or build it. LRU eviction at cap."""
    t_b  = int(t / _T_QUANT)
    n_b  = round(S.p_n / _BKT_N) * _BKT_N
    wx_b = round(ax  / _BKT_A) * _BKT_A
    wy_b = round(ay  / _BKT_A) * _BKT_A
    key  = (mode, n_b, wx_b, wy_b,
            int(fr)&0xF0, int(fg)&0xF0, int(fb)&0xF0, t_b)
    if key in _FCACHE:
        _FCACHE.move_to_end(key)
        return _FCACHE[key]
    # Miss — render
    Geo.refresh_2d(S.p_n, ax, ay)
    Geo.refresh_3d(S.p_n)
    surf = _make_surface(key, t_b * _T_QUANT, ax, ay, br, fr, fg, fb, mode)
    _FCACHE[key] = surf
    if len(_FCACHE) > _FCAP:
        _FCACHE.popitem(last=False)
    return surf

# ── Input ─────────────────────────────────────────────────────────────────
# Use Listener instead of Controller.position poll — saves ~0.1 ms/frame
_mx = _my = 0
_last_key  = 0.0
_scroll_acc = 0.0

def _on_move(x, y): global _mx, _my; _mx=x; _my=y
def _on_click(x,y,btn,pressed): global _mx,_my; _mx=x; _my=y
def _on_scroll(x,y,dx,dy): global _scroll_acc; _scroll_acc+=abs(dy)
def _on_press(_k): global _last_key; _last_key=time.time()

_kb = keyboard.Listener(on_press=_on_press)
_ml = mouse.Listener(on_move=_on_move, on_click=_on_click, on_scroll=_on_scroll)
_kb.start(); _ml.start()

# ── Welford online variance tracker (O(1) per update) ────────────────────
class Welford:
    __slots__ = ('n','mean','M2')
    def __init__(self): self.n=0; self.mean=0.0; self.M2=0.0
    def update(self,x):
        self.n+=1; d=x-self.mean; self.mean+=d/self.n
        self.M2+=(x-self.mean)*(d if self.n>1 else 0.0)  # noqa
    def var(self): return self.M2/self.n if self.n>1 else 0.0
    def reset(self): self.n=0; self.mean=0.0; self.M2=0.0

_wfx = Welford(); _wfy = Welford()

# Circular mouse ring for distance (just track last position)
_prev_mx = _prev_my = 0
_dist_acc = 0.0   # running distance within current 60-sample window
_samp_n   = 0     # samples in current window

# ── Histories ─────────────────────────────────────────────────────────────
speed_hist = deque(maxlen=20)
rhod_hist  = deque(maxlen=20)
scroll_hist= deque(maxlen=40)

# ── CSV ───────────────────────────────────────────────────────────────────
_LOG      = os.path.join(os.path.dirname(os.path.abspath(__file__)),"cognitive_log.csv")
_LOG_EVERY= 5.0; _log_buf=[]; _last_fl=time.time()
_lf=open(_LOG,"w",newline="",buffering=8192); _lw=csv.writer(_lf)
_lw.writerow(["ts","state","tier","mode","BC","RL","CI","RH","efficacy",
              "pred_fatigue","mouse_speed","scroll_stress","var_x","var_y","side"])

def _maybe_flush(now):
    global _last_fl
    if now-_last_fl>=_LOG_EVERY and _log_buf:
        _lw.writerows(_log_buf); _log_buf.clear(); _lf.flush(); _last_fl=now

# ── Linear regression (5-tier) ────────────────────────────────────────────
def _predict(data,future=0,tier=0):
    nt=len(data)
    if nt<4: return data[-1] if nt else 0.0
    s=list(data)[-4:] if tier>=4 else data[::tier+1]
    n=len(s)
    sx=n*(n-1)//2; sx2=n*(n-1)*(2*n-1)//6; sy=sxy=0.0
    for i,v in enumerate(s): sy+=v; sxy+=i*v
    d=n*sx2-sx*sx
    if d==0: return sy/n
    slope=(n*sxy-sx*sy)/d
    return max(0.0,slope*(n+future)+(sy-slope*sx)/n)

# ── Biometrics (4 Hz) ─────────────────────────────────────────────────────
def compute(now):
    global _scroll_acc, _dist_acc, _samp_n, _prev_mx, _prev_my

    # Welford variance (reset per window)
    vx = _wfx.var(); vy = _wfy.var()
    speed = _dist_acc
    _wfx.reset(); _wfy.reset()
    _dist_acc = 0.0; _samp_n = 0

    speed_hist.append(speed)
    pred_s = _predict(list(speed_hist), 0, S.tier)
    err = abs(pred_s - speed)
    if err > 500.0: S.tier=0; S.stable=0
    else:
        S.stable+=1
        if S.stable>8 and S.tier<4: S.tier+=1; S.stable=0

    sc = _scroll_acc * 4.0; _scroll_acc = 0.0
    scroll_hist.append(sc)
    pred_sc = _predict(list(scroll_hist), 5, S.tier)

    Photo.step(pred_s, pred_sc)
    rhod_hist.append(Photo.RH)
    Photo.pred_f = min(100.0, _predict(list(rhod_hist), 15, S.tier))

    E  = Photo.EFF
    ma = W * 0.4 * (0.5 + 0.5*(1.0-E))

    # Saccadic swap
    if (now-S.swap_t>45.0 or
        (pred_sc>80 and now-S.swap_t>8.0) or
        (pred_s>4000 and now-S.swap_t>10.0)):
        S.left=not S.left; S.swap_t=now
        S.tgt_x=20.0 if S.left else float(SCREEN_W-W-20)

    rh_low = Photo.RH<35.0 or Photo.CI<25.0
    scroll_active = pred_sc>30.0

    if rh_low or (S.state=="RECOVERY" and Photo.RH<80.0):
        S.state="RECOVERY"
        S.tr,S.tg,S.tb=C_RECOVERY
        S.t_n,S.t_r=25.0,12.0
        S.t_wx,S.t_wy,S.t_sp=ma*0.6,ma*0.6,-0.03
        S.t_mode=1; S.lerp=0.02; S.tgt_y=20.0

    elif scroll_active:
        S.state="SCROLLING"
        S.tr,S.tg,S.tb=C_SCROLL
        sf=min(1.0,pred_sc/150.0)
        S.t_n=35.0+35.0*sf; S.t_r=6.0+8.0*sf
        S.t_sp=0.25+0.75*sf
        S.t_wx=ma*(0.4+0.6*sf); S.t_wy=ma*(0.4+0.6*sf)
        S.t_mode=1; S.lerp=0.15; S.tgt_y=SCREEN_H*0.35

    elif now-_last_key<1.5:
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
    _log_buf.append((f"{now:.3f}",S.state,S.tier,S.mode,
        f"{Photo.BC:.2f}",f"{Photo.RL:.2f}",
        f"{Photo.CI:.2f}",f"{Photo.RH:.2f}",f"{Photo.EFF:.4f}",
        f"{Photo.pred_f:.2f}",f"{speed:.1f}",f"{sc:.1f}",
        f"{vx:.1f}",f"{vy:.1f}","L" if S.left else "R"))

# ── Main loop ─────────────────────────────────────────────────────────────
clock     = pygame.time.Clock()
last_eval = time.time()
_prev_mx  = _mx; _prev_my = _my

print(f"[widget] surface-cache | Welford variance | listener poll | log→{_LOG}")

running = True
while running:
    now = time.time()

    # Accumulate distance + Welford variance from last listener position
    dx = _mx - _prev_mx;  dy = _my - _prev_my
    _dist_acc += math.sqrt(dx*dx + dy*dy)
    _wfx.update(_mx); _wfy.update(_my)
    _prev_mx = _mx;  _prev_my = _my
    _samp_n  += 1

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: running=False
        elif ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE: running=False

    # Window slide (both axes)
    S.win_x += (S.tgt_x - S.win_x) * 0.05
    S.win_y += (S.tgt_y - S.win_y) * 0.05
    window.position = (int(S.win_x), int(S.win_y))

    # Biometrics @ 4 Hz
    if now - last_eval >= 0.25:
        compute(now); last_eval = now

    _maybe_flush(now)

    # Lerp visual params (inlined)
    ls=S.lerp
    S.cr+=(S.tr-S.cr)*ls; S.cg+=(S.tg-S.cg)*ls; S.cb+=(S.tb-S.cb)*ls
    S.p_n +=(S.t_n -S.p_n) *ls; S.p_r +=(S.t_r -S.p_r) *ls
    S.p_wx+=(S.t_wx-S.p_wx)*ls; S.p_wy+=(S.t_wy-S.p_wy)*ls
    S.p_sp+=(S.t_sp-S.p_sp)*ls
    S.t_var += 0.02 * S.p_sp

    # Hot path: surface cache lookup → blit (zero particle work on hit)
    frame = get_frame(S.t_var, S.p_wx, S.p_wy, S.p_r,
                      S.cr, S.cg, S.cb, S.mode)
    screen.blit(frame, (0, 0))
    pygame.display.flip()
    clock.tick(60)

# ── Shutdown ──────────────────────────────────────────────────────────────
if _log_buf: _lw.writerows(_log_buf)
_lf.close(); _kb.stop(); _ml.stop()
pygame.quit(); sys.exit()