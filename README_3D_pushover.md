# 🏗️ Nonlinear Pushover Analysis — 3-Story 3D Steel MRF
### OpenSeesPy · Distributed Plasticity · 3D Frame · SI Units · ops-code Visualization

---

## Overview

This repository implements a fully parametric **nonlinear static (pushover) analysis** of a 3-story, 3D steel moment-resisting frame (MRF) using **OpenSeesPy**. The model runs from a single Python script and produces four publication-quality figures for pre- and post-processing.

The model is also compatible with the [**ops-code**](https://github.com/igor-barcelos/ops-code) VS Code extension for interactive 3-D rendering directly inside your editor.

---

## 📐 Structural System

```
Plan view (X–Z):       Front elevation (X–Y):

  Z                     Y (m)
  ↑                     |
  ● ── ● ── ●     11.2 ─ ●──────●──────●   ← Roof   (W21×44 beams)
  │    │    │          |  │      │      │
  ● ── ● ── ●      7.6 ─ ●──────●──────●   ← Floor 2 (W24×55 beams)
  │    │    │          |  │      │      │
  ● ── ● ── ●  →   4.0 ─ ●──────●──────●   ← Floor 1 (W24×55 beams)
                         |  │      │      │
  ←6m→←6m→            0.0─[▲]────[▲]────[▲] ← Fixed bases
  L=6m per bay              ←6m──→←6m──→
```

| Member | Section | Story |
|--------|---------|-------|
| Column | W14×82 | Story 1 |
| Column | W14×48 | Story 2 |
| Column | W14×38 | Story 3 |
| Beam X | W24×55 | Floors 1–2 |
| Beam X | W21×44 | Roof |
| Beam Z | W21×44 | Floors 1–2 |
| Beam Z | W18×35 | Roof |

**Material:** A992 Steel — Fy = 345 MPa, E = 200 GPa

---

## 🔧 Nonlinear Modelling Strategy

| Feature | Implementation |
|---------|---------------|
| Element | `dispBeamColumn` — displacement-based, 5 Gauss-Lobatto pts |
| Section | `Fiber` (strong-axis, Steel02) + `Aggregator` (elastic My + T) |
| Material | `Steel02` — Giuffré–Menegotto–Pinto, b = 0.02 |
| Fibers | 6 across flange thickness + 16 web fibers |
| Geometry | `Corotational` transformation — full 3D P-Δ/P-δ |
| Gravity | Load-controlled, frozen with `loadConst` |
| Pushover | Displacement-controlled at roof node, 2 mm/step |
| Target | 5 % of total height = 560 mm |
| Lateral pattern | Inverted-triangular — proportional to Wᵢ × hᵢ (FEMA 356 §3.3.3.2) |

> **Why `dispBeamColumn` + `Aggregator`, not `forceBeamColumn`?**
> `ForceBeamColumn3d` requires a fully populated 6×6 section flexibility matrix.
> A pure fiber section (`ops.fiber`) only populates the axial + strong-axis rows
> leaving weak-axis My and torsion T as zero rows → singular matrix →
> `"could not invert flexibility"` at step 0.
> The `section('Aggregator')` appends elastic My and T, giving full rank.

> **Why `vecxz = (0,0,1)` for columns, not `(1,0,0)`?**
> Column local-x = global Y. With `vecxz=(1,0,0)`, lateral load bends about
> the **weak axis** (Iy) — 6× too soft. With `vecxz=(0,0,1)`, lateral load
> bends about the **strong axis** (Iz) — physically correct.

---

## 📊 Visualization Outputs

Four figures are produced automatically (white background, black fonts, SI units):

| Figure | Content |
|--------|---------|
| **Fig 1 — Pre-processing** | 3D frame geometry, W-shape labels, gravity arrows, inverted-Δ lateral pattern, legend panel (outside model area) |
| **Fig 2 — Pushover curve** | Base shear V (kN) vs roof drift Δ/H (%), viridis colourmap, IO/LS/CP performance table (FEMA 356) |
| **Fig 3 — Deformed shape** | ×5 amplified 3D isometric + front elevation with story drift callouts (outside frame area) |
| **Fig 4 — Story drift profile** | Horizontal bar chart per story, RdYlGn colourmap, FEMA 356 limit lines |

The **ops-code** VS Code extension provides an additional interactive 3-D viewer with colour-coded sections, force diagrams, and animation.

---

## 🚀 Quick Start

### 1 · Install dependencies

```bash
pip install openseespy numpy matplotlib
```

### 2 · Run the analysis

```bash
python steel_frame_3D_pushover_SI.py
```

### 3 · Interactive 3-D viewer (optional)

1. Install [ops-code](https://marketplace.visualstudio.com/items?itemName=ops-code) from VS Code Marketplace
2. Open `steel_frame_3D_pushover_SI.py` in VS Code
3. Right-click → **"Run in ops-code"**
4. Click **"Run Analysis"** in the viewer panel

---

## 📂 Repository Structure

```
📦 steel-mrf-3d-pushover/
 ├── steel_frame_3D_pushover_SI.py   # Full model + analysis + plots
 └── README.md                        # This file
```

---

## 📏 Units

| Quantity | Unit |
|----------|------|
| Length | m |
| Force | kN |
| Stress | kN/m² (kPa) |
| Moment | kN·m |
| Mass | t (kN·s²/m) |

---

## ⚙️ Key Parameters

```python
nBayX  = 2       # bays in X direction
nBayZ  = 2       # bays in Z direction
Lbay   = 6.0     # bay width (m)
H      = [4.0, 3.6, 3.6]   # story heights (m)
Fy     = 345e3   # yield stress (kN/m²)
E_s    = 200e6   # Young's modulus (kN/m²)
b_sh   = 0.02    # strain-hardening ratio
```

---

## 📖 References

- **AISC 360-22** — Specification for Structural Steel Buildings
- **ASCE 7-22** — Minimum Design Loads for Buildings
- **FEMA 356** — Prestandard for Seismic Rehabilitation (§3.3, §5.4)
- **ATC-72** — Modeling and Acceptance Criteria for Tall Buildings
- Giuffré & Menegotto (1973) — Steel02 uniaxial material model
- McKenna et al. — *OpenSees*, UC Berkeley
- Barcelos, I. — [ops-code VS Code extension](https://github.com/igor-barcelos/ops-code)

---

## ⚠️ Disclaimer

This script is developed for **research and educational purposes only**.
All real-world structural designs must be reviewed and approved by a
**licensed Professional Engineer (PE)** in the applicable jurisdiction.

---

<div align="center">

Built with [OpenSeesPy](https://openseespydoc.readthedocs.io) · [Matplotlib](https://matplotlib.org) · [ops-code](https://github.com/igor-barcelos/ops-code)

</div>
