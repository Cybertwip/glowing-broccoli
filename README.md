# Adaptive Peripheral Visual Correction via Real-Time Photochemical Cascade Modelling and Fibonacci-Based Optomotor Stimulation

[cite_start]**Anonymous Author(s)** [cite: 3]
[cite_start]*Preprint under review, 2025* [cite: 4, 5]

---

## Abstract
[cite_start]This paper introduces a lightweight desktop overlay designed to counteract digital visual fatigue[cite: 8]. [cite_start]By monitoring mouse kinematics, keyboard cadence, and scroll dynamics, the system generates adaptive optomotor stimulation patterns aimed at supporting rhodopsin regeneration[cite: 8]. [cite_start]The biological core utilizes a four-pool Euler ODE model of the visual-cycle cascade (from $\beta$-carotene to rhodopsin) paired with a five-tier linear-regression predictor[cite: 9]. [cite_start]Visual stimuli transition between 2D golden-angle spirals and 3D Fibonacci spheres based on the user's inferred oculomotor state[cite: 10]. [cite_start]Performance is optimized via a hash-cached geometry system, maintaining CPU load below 2%[cite: 11].

---

## 1. Introduction
[cite_start]Digital eye strain (DES) affects approximately 50–90% of individuals engaged in prolonged screen use[cite: 15]. [cite_start]Symptoms—including blurred vision, headaches, and difficulty refocusing—stem from sustained near-vergence and the continuous bleaching of rhodopsin without adequate recovery[cite: 16]. [cite_start]Rhodopsin is a G-protein-coupled receptor formed by opsin and 11-cis-retinal[cite: 17]. [cite_start]High-intensity illumination bleaches this pigment faster than the retinal pigment epithelium (RPE) can recycle it, leading to reduced sensitivity and contrast discrimination[cite: 18]. [cite_start]While optomotor stimulation (using moving patterns to drive oculomotor reflexes) is used in clinical therapies, this project explores a passive desktop widget for real-time, calibrated stimulation[cite: 19, 22].

---

## 2. Biological Background

### 2.1 The Visual Cycle
[cite_start]The visual cycle, or Wald cycle, describes the continuous regeneration of rhodopsin[cite: 27, 28]. [cite_start]When 11-cis-retinal (CI) absorbs a photon, it isomerizes to all-trans-retinal, triggering bleaching[cite: 29, 30]. [cite_start]This is eventually re-isomerized by the enzyme RPE65 back to CI to rebind with opsin[cite: 31].



### 2.2 $\beta$-Carotene and Retinol Stores
[cite_start]$\beta$-carotene (BC) is cleaved by the enzyme BCO1 to yield retinal, which is reduced to retinol (RL) for storage[cite: 33, 34]. [cite_start]Under high visual demand, RL reserves can be mobilized faster than they are replenished, creating a supply bottleneck[cite: 35]. [cite_start]The model accounts for this with a slow replenishment constant $K_{BCR} = 4 \times 10^{-5}s^{-1}$[cite: 36].

### 2.3 Stress and Wavelength Factors
* [cite_start]**Scrolling Stress:** Sustained vertical scrolling suppresses accommodative fluctuations that typically allow ciliary muscle recovery[cite: 38, 39]. [cite_start]This is modeled as an extra rhodopsin bleach term proportional to scroll rate[cite: 40].
* [cite_start]**Wavelength:** Rhodopsin absorption peaks at ~498 nm[cite: 42]. [cite_start]The system uses a warm red-amber (RGB 120, 30, 5) for its RECOVERY state to minimize further pigment bleaching[cite: 43].

---

## 3. System Architecture
[cite_start]The system utilizes four loosely coupled subsystems to maximize efficiency[cite: 46]:

| Subsystem | Rate | Principal Cost | Mitigation |
| :--- | :--- | :--- | :--- |
| **Input Monitor** | Async | Thread context switch | [cite_start]Daemon threads, atomic float [cite: 48] |
| **Biometric Engine** | 4 Hz | $O(60)$ distance sum | [cite_start]Incremental; tier decimation [cite: 48] |
| **Photo ODE** | 4 Hz | 4 multiplies + 4 adds | [cite_start]Euler integration; no allocation [cite: 48] |
| **Renderer** | 60 Hz | Per-circle LUT lookup | [cite_start]Hash-cached geometry surfaces [cite: 48] |

---

## 4. The Four-Pool Photochemical ODE Model
[cite_start]The model tracks four pools ($BC$, $RL$, $CI$, and $RH$) as percentages of physiological capacity[cite: 51]. [cite_start]The Euler system, integrated at $\Delta t = 0.25$ s, is defined as[cite: 52]:

$$\frac{dBC}{dt} = -K_{BCO1} \cdot I_n \cdot BC + K_{BCR} \cdot (100 - BC)$$
$$\frac{dRL}{dt} = K_{BCO1} \cdot I_n \cdot BC - K_{RPE65} \cdot RL \cdot D$$
$$\frac{dCI}{dt} = K_{RPE65} \cdot RL \cdot D - K_{BIND} \cdot CI \cdot D$$
$$\frac{dRH}{dt} = K_{BIND} \cdot CI \cdot D - K_{BL} \cdot I_n \cdot RH - K_{SCR} \cdot S_n \cdot RH + K_{DK} \cdot D \cdot 100$$

* [cite_start]**$I_n$**: Normalized mouse activity[cite: 55].
* **$S_n$**: Normalized scroll stress[cite: 55].
* **$D$**: Rhodopsin deficit fraction ($max(0, 100-RH)/100$)[cite: 55].
* [cite_start]**Correction Efficacy ($E$)**: Calculated as $(BC \cdot RL \cdot CI \cdot RH / 100^4)^{1/4}$[cite: 57].

---

## 5. Stimulus Pattern Design
[cite_start]The system employs two primary geometric strategies to target specific vision goals[cite: 62]:

### 5.1 2D Golden-Angle Spiral
[cite_start]This pattern distributes $N$ particles along a spiral ($\phi \approx 137.508^\circ$) to drive smooth-pursuit eye movements[cite: 65, 66]. [cite_start]In the READING state, an amplitude ratio of $A_x/A_y=6$ is used to exercise horizontal pursuit without interfering with vertical reading saccades[cite: 67].

### 5.2 3D Fibonacci Sphere
[cite_start]Points are projected from a Fibonacci sphere using the golden ratio $\phi_{Fib} = \pi(1+\sqrt{5})$[cite: 69]. [cite_start]The sphere rotates and oscillates in depth to drive accommodative vergence cycling, intended to release ciliary muscle tension[cite: 70, 71, 73].



### 5.3 State Mapping
| State | Pattern | Colour | Target |
| :--- | :--- | :--- | :--- |
| **RECOVERY** | 3D sphere (slow) | Red-amber | [cite_start]Rhodopsin regeneration [cite: 75] |
| **SCROLLING** | 3D sphere (fast) | Violet | [cite_start]Ciliary muscle reset [cite: 75] |
| **READING** | 2D H-Lissajous | Amber-orange | [cite_start]Horizontal CSF fatigue [cite: 75] |
| **CORRECTIVE**| 2D+3D blend | Cyan-green | [cite_start]Saccadic reset [cite: 75] |
| **TYPING** | 2D ambient | Deep indigo | [cite_start]Peripheral stimulation [cite: 75] |

---

## 6. Optimization & Control
### 6.1 Performance Gains
* **Geometry Cache:** Particle positions are bucketed; a 7200-entry sine/cosine Look-Up Table (LUT) eliminates trigonometric calls in the hot path[cite: 81, 82].
* [cite_start]**Surface Frame Cache:** Uses an LRU cache of 128 pre-rendered surfaces to replace complex particle rendering with a single hardware-accelerated blit[cite: 88, 89, 90].
* [cite_start]**Tiered Regression:** A 5-tier predictor for mouse speed decimates input history to save CPU during stable behavior, only escalating to full data on high prediction errors[cite: 92, 93].

### 6.2 Scroll-Aware Correction
[cite_start]Scrolling is tracked via a listener that accumulates tick magnitude[cite: 96]. [cite_start]High scroll rates ($S > 30$ ticks $s^{-1}$) trigger the SCROLLING state, and the widget window slides vertically to 35% screen height to align with typical gaze anchors[cite: 99, 102].

---

## 7. Future Directions & Conclusion
[cite_start]Preliminary informal observations suggest subjective improvement in visual comfort and faster recovery of contrast sensitivity[cite: 107]. [cite_start]A proposed randomized pilot study ($n=30$) will measure contrast sensitivity and blink rate over a 4-week period to validate the photochemical model[cite: 112, 118, 119].

[cite_start]The system offers a mechanistically principled, low-resource approach to visual care for digital workers, grounded in the biochemistry of the visual cycle[cite: 122, 126].
