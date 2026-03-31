# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:40:22 2026

@author: Wahab
"""

"""
================================================================================
  steel_frame_3D_pushover_SI.py
  3-Bay × 3-Bay Plan · 3-Story Steel Moment-Resisting Frame
  Nonlinear Pushover Analysis — OpenSeesPy · SI Units (kN, m, s)

  VERIFIED ELEMENT STRATEGY
  ─────────────────────────
  · dispBeamColumn (displacement-based, 5 Lobatto pts)
  · section('Aggregator'): fiber section (strong-axis, Steel02)
    + elastic My (weak-axis) + elastic T (torsion)
  · Corotational geometric transformation
  · Displacement-controlled pushover in +X to 5 % Htot

  WHY NOT forceBeamColumn IN 3D?
  ─────────────────────────────
  ForceBeamColumn3d requires a fully populated 6×6 section flexibility matrix.
  A pure fiber section (ops.fiber) only populates the axial + strong-axis bending
  rows — the weak-axis and torsion rows remain zero → singular flexibility
  → "could not invert flexibility" at step 0.
  Fix: section('Aggregator') appends elastic My and T stiffness to the fiber
  section, giving the full rank needed by dispBeamColumn.

  Visualization  (all white/light background)
  ─────────────────────────────────────────────
  Fig 1  PRE   — 3D geometry, supports, load pattern, legend panel
  Fig 2  POST  — Pushover curve + IO/LS/CP performance table
  Fig 3  POST  — 3D deformed shape ×5 + front elevation drift callouts
  Fig 4  POST  — Story drift profile (horizontal bar chart)
================================================================================
"""

import openseespy.opensees as ops
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  LIGHT DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
C = dict(
    bg_fig    = "white",
    bg_ax     = "white",
    bg_panel  = "#f0f4f8",
    bg_box    = "white",
    border    = "#64748b",
    border_lt = "#94a3b8",
    grid      = "#e2e8f0",
    fg        = "black",          # ALL text black
    fg_dim    = "#1e293b",        # near-black for secondary text
    col_s1    = "#1d4ed8",
    col_s2    = "#3b82f6",
    col_s3    = "#60a5fa",
    beam_x    = "#047857",
    beam_z    = "#0891b2",
    load_grav = "#b91c1c",
    load_lat  = "#c2410c",
    node_edge = "#1e293b",
    support   = "#1e3a8a",
    dim       = "#475569",
    deformed  = "#92400e",
    ghost     = "#94a3b8",
    io_clr    = "#15803d",
    ls_clr    = "#92400e",
    cp_clr    = "#991b1b",
    peak_clr  = "#6d28d9",
    curve_cmap= "viridis",
)

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#64748b",
    "axes.labelcolor":  "black",
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "xtick.color":      "black",
    "ytick.color":      "black",
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "text.color":       "black",
    "grid.color":       "#e2e8f0",
    "grid.linewidth":   0.7,
    "legend.facecolor": "#f0f4f8",
    "legend.edgecolor": "#64748b",
    "legend.fontsize":  11,
    "font.family":      "DejaVu Sans",
    "figure.dpi":       110,
})


def _style_ax2d(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["bg_ax"])
    for sp in ax.spines.values():
        sp.set_edgecolor(C["border"]); sp.set_linewidth(1.3)
    ax.set_title(title, color="black", fontsize=14,
                 fontweight="bold", pad=16)
    ax.set_xlabel(xlabel, color="black", fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel, color="black", fontsize=12, labelpad=8)
    ax.tick_params(colors="black", labelsize=11, length=4)
    ax.grid(True, color=C["grid"], linewidth=0.7, linestyle="--", alpha=0.9)
    ax.set_axisbelow(True)


def _style_ax3d(ax, title):
    ax.set_facecolor(C["bg_ax"])
    ax.set_title(title, color="black", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("X  (m)", color="black", fontsize=11, labelpad=6)
    ax.set_ylabel("Z  (m)", color="black", fontsize=11, labelpad=6)
    ax.set_zlabel("Y  (m)", color="black", fontsize=11, labelpad=6)
    ax.tick_params(colors="black", labelsize=9)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(C["border_lt"])
    ax.yaxis.pane.set_edgecolor(C["border_lt"])
    ax.zaxis.pane.set_edgecolor(C["border_lt"])
    ax.grid(True, color=C["grid"], linewidth=0.4, alpha=0.6)


def _style_panel(ax):
    ax.set_facecolor(C["bg_panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(C["border"]); sp.set_linewidth(1.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY & SECTION PROPERTIES (SI)
# ══════════════════════════════════════════════════════════════════════════════
nBayX  = 2;   nBayZ  = 2      # 2×2 bays  →  3×3 column grid
Lbay   = 6.0                  # m
H      = [4.0, 3.6, 3.6]      # story heights (m)
nStory = len(H)
Htot   = sum(H)
nColX  = nBayX + 1            # 3
nColZ  = nBayZ + 1            # 3

# W-shapes (SI)
W = {
    #             A(m²)     E·A elev  Iz(m⁴)    Iy(m⁴)    J(m⁴)     d(m)
    "W14x82": dict(A=0.01548, Iz=3.668e-4, Iy=6.162e-5, J=1.463e-6, d=0.363),
    "W14x48": dict(A=0.00910, Iz=2.014e-4, Iy=2.141e-5, J=5.370e-7, d=0.351),
    "W14x38": dict(A=0.00723, Iz=1.624e-4, Iy=1.184e-5, J=2.817e-7, d=0.353),
    "W24x55": dict(A=0.01045, Iz=5.618e-4, Iy=1.212e-5, J=3.698e-7, d=0.599),
    "W21x44": dict(A=0.00839, Iz=3.509e-4, Iy=8.623e-6, J=2.364e-7, d=0.526),
    "W18x35": dict(A=0.00665, Iz=2.167e-4, Iy=5.369e-6, J=1.406e-7, d=0.460),
}
# bf and tf needed for fiber layout
BF = {"W14x82":0.257,"W14x48":0.204,"W14x38":0.172,
      "W24x55":0.178,"W21x44":0.165,"W18x35":0.152}
TF = {"W14x82":0.0217,"W14x48":0.0151,"W14x38":0.0132,
      "W24x55":0.0128,"W21x44":0.0119,"W18x35":0.0109}
TW = {"W14x82":0.0130,"W14x48":0.00864,"W14x38":0.00762,
      "W24x55":0.01003,"W21x44":0.00889,"W18x35":0.00762}

COL_SEC  = {1:"W14x82", 2:"W14x48", 3:"W14x38"}
BM_X_SEC = {1:"W24x55", 2:"W24x55", 3:"W21x44"}
BM_Z_SEC = {1:"W21x44", 2:"W21x44", 3:"W18x35"}

# sec_tag scheme: fiber section = shape_idx*10+1, aggregator = shape_idx*10+2
SHAPES   = ["W14x82","W14x48","W14x38","W24x55","W21x44","W18x35"]
FIBER_TAG = {s: i*10+1 for i, s in enumerate(SHAPES, 1)}
AGG_TAG   = {s: i*10+2 for i, s in enumerate(SHAPES, 1)}
INT_TAG   = {s: i*10+3 for i, s in enumerate(SHAPES, 1)}

# Material constants
E_s = 200.0e6   # kN/m²
G_s =  77.0e6   # kN/m²
Fy  = 345.0e3   # kN/m²
b_sh = 0.02   # strain-hardening ratio (post-yield / elastic)
              # 0.02 = 2% gives a clear visible kink on the pushover curve
R0, cR1, cR2 = 18.0, 0.925, 0.15

# Gravity loads
wD_fl=3.83; wL_fl=2.39; wD_rf=3.83; wL_rf=0.96
trib_area = Lbay * Lbay   # m²

def Pgrav_node(wD, wL, pos):
    fac = {"corner":0.25, "edge":0.50, "interior":1.00}[pos]
    return (1.2*wD + 1.6*wL) * trib_area * fac

W_fl = (wD_fl+0.5*wL_fl)*(nBayX*Lbay)*(nBayZ*Lbay)
W_rf = (wD_rf+0.5*wL_rf)*(nBayX*Lbay)*(nBayZ*Lbay)
W_story = [W_fl, W_fl, W_rf]


# ══════════════════════════════════════════════════════════════════════════════
#  NODE NUMBERING   tag = story*1000 + iz*10 + ix + 1
# ══════════════════════════════════════════════════════════════════════════════
def ntag(story, iz, ix):
    return story * 1000 + iz * 10 + ix + 1

def node_xyz(story, iz, ix):
    return ix*Lbay, sum(H[:story]), iz*Lbay


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_model():
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # Nodes
    for s in range(nStory+1):
        for iz in range(nColZ):
            for ix in range(nColX):
                x, y, z = node_xyz(s, iz, ix)
                ops.node(ntag(s, iz, ix), x, y, z)

    # Fixed bases
    for iz in range(nColZ):
        for ix in range(nColX):
            ops.fix(ntag(0, iz, ix), 1,1,1,1,1,1)

    # Masses
    g = 9.81
    for s in range(1, nStory+1):
        wD = wD_rf if s==nStory else wD_fl
        wL = wL_rf if s==nStory else wL_fl
        for iz in range(nColZ):
            for ix in range(nColX):
                ex = (ix==0 or ix==nColX-1)
                ez = (iz==0 or iz==nColZ-1)
                pos = "corner" if ex and ez else ("edge" if ex or ez else "interior")
                P = Pgrav_node(wD, wL, pos)
                m = P/g
                ops.mass(ntag(s,iz,ix), m,m,m,0,0,0)

    # ── Material ─────────────────────────────────────────────────────────────
    ops.uniaxialMaterial("Steel02", 1, Fy, E_s, b_sh, R0, cR1, cR2)

    # ── Sections: Fiber (strong-axis) + Aggregator (adds My and T) ──────────
    # The fiber section handles axial (N) + strong-axis bending (Mz via Steel02).
    # section('Aggregator') appends elastic My (weak-axis) and T (torsion).
    # Fiber layout: 6 fibers distributed across flange THICKNESS (not all at centroid)
    # + 16 web fibers → captures gradual plastification from flange tips inward.
    mat_tag = 2
    for shape in SHAPES:
        s   = W[shape]
        d   = s["d"];  bf=BF[shape]; tf=TF[shape]; tw=TW[shape]
        dw  = d - 2*tf
        Iy  = s["Iy"];  J = s["J"]
        Af  = bf*tf;    Aw = dw*tw

        my_mat = mat_tag;     ops.uniaxialMaterial("Elastic", my_mat, E_s*Iy)
        t_mat  = mat_tag+1;   ops.uniaxialMaterial("Elastic", t_mat,  G_s*J)
        mat_tag += 2

        ops.section("Fiber", FIBER_TAG[shape], "-GJ", G_s*J)
        nf, nw = 6, 16   # 6 fibers across flange thickness + 16 web fibers
        # Distribute flange fibers ACROSS flange thickness (not all at centroid)
        for k in range(nf):
            yf = (d/2 - tf) + (k + 0.5)*tf/nf   # from flange root to tip
            ops.fiber( yf, 0.0, Af/nf, 1)
            ops.fiber(-yf, 0.0, Af/nf, 1)
        for k in range(nw):
            yf = -dw/2 + (k + 0.5)*dw/nw
            ops.fiber(yf, 0.0, Aw/nw, 1)

        ops.section("Aggregator", AGG_TAG[shape],
                    my_mat, "My",
                    t_mat,  "T",
                    "-section", FIBER_TAG[shape])

        ops.beamIntegration("Lobatto", INT_TAG[shape], AGG_TAG[shape], 5)

    # ── Geometric transformations ─────────────────────────────────────────────
    # Column local-x axis = global Y (0,1,0).
    # vecxz defines local-z, which determines the bending plane:
    #   vecxz = (0,0,1) → local-z = global Z → lateral load +X bends about local Z
    #                      → element uses Iz (STRONG axis) for X-direction pushover ✓
    #   vecxz = (1,0,0) → local-z = global X → lateral load +X bends about local Y
    #                      → element uses Iy (WEAK axis) ✗  6× too soft!
    ops.geomTransf("Corotational", 1,  0, 0, 1)  # columns : vecxz = global Z ✓
    ops.geomTransf("Linear",       2,  0, 0, 1)  # X-beams : local-z = global Z ✓
    ops.geomTransf("Linear",       3,  1, 0, 0)  # Z-beams : vecxz = global X

    # ── Elements: dispBeamColumn ──────────────────────────────────────────────
    ele_tag = 1

    for s in range(1, nStory+1):                  # Columns
        it = INT_TAG[COL_SEC[s]]
        for iz in range(nColZ):
            for ix in range(nColX):
                ops.element("dispBeamColumn", ele_tag,
                            ntag(s-1,iz,ix), ntag(s,iz,ix), 1, it)
                ele_tag += 1

    for s in range(1, nStory+1):                  # X-direction beams
        it = INT_TAG[BM_X_SEC[s]]
        for iz in range(nColZ):
            for ix in range(nColX-1):
                ops.element("dispBeamColumn", ele_tag,
                            ntag(s,iz,ix), ntag(s,iz,ix+1), 2, it)
                ele_tag += 1

    for s in range(1, nStory+1):                  # Z-direction beams
        it = INT_TAG[BM_Z_SEC[s]]
        for ix in range(nColX):
            for iz in range(nColZ-1):
                ops.element("dispBeamColumn", ele_tag,
                            ntag(s,iz,ix), ntag(s,iz+1,ix), 3, it)
                ele_tag += 1

    return ele_tag - 1


# ══════════════════════════════════════════════════════════════════════════════
#  GRAVITY
# ══════════════════════════════════════════════════════════════════════════════
def apply_gravity():
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for s in range(1, nStory+1):
        wD = wD_rf if s==nStory else wD_fl
        wL = wL_rf if s==nStory else wL_fl
        for iz in range(nColZ):
            for ix in range(nColX):
                ex=(ix==0 or ix==nColX-1); ez=(iz==0 or iz==nColZ-1)
                pos="corner" if ex and ez else ("edge" if ex or ez else "interior")
                P = Pgrav_node(wD, wL, pos)
                ops.load(ntag(s,iz,ix), 0.0,-P,0.0,0.0,0.0,0.0)
    ops.system("BandGeneral"); ops.numberer("RCM"); ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Newton"); ops.analysis("Static")
    ok = ops.analyze(10)
    ops.loadConst("-time", 0.0)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  PUSHOVER
# ══════════════════════════════════════════════════════════════════════════════
def run_pushover(target_pct=0.05):
    ctrl_node = ntag(nStory, 0, 0)
    target    = target_pct * Htot
    print(f"        Control node     : {ctrl_node}")
    print(f"        Target disp      : {target*1000:.1f} mm")

    # Inverted-triangular pattern (total reference = 1 kN)
    h_cum   = [sum(H[:s+1]) for s in range(nStory)]
    wh      = np.array([W_story[s]*h_cum[s] for s in range(nStory)], dtype=float)
    Fx_norm = wh / wh.sum()
    n_floor = nColX * nColZ

    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    for s in range(1, nStory+1):
        F = Fx_norm[s-1] / n_floor
        for iz in range(nColZ):
            for ix in range(nColX):
                ops.load(ntag(s,iz,ix), float(F),0.0,0.0,0.0,0.0,0.0)

    ops.system("BandGeneral"); ops.numberer("RCM"); ops.constraints("Plain")
    d_step  = 0.002
    n_steps = int(target / d_step)
    ops.integrator("DisplacementControl", ctrl_node, 1, d_step)
    ops.algorithm("Newton"); ops.analysis("Static")

    disp, shear = [0.0], [0.0]
    for step in range(n_steps):
        ok = ops.analyze(1)

        if ok != 0:   # sub-step fallback
            ops.integrator("DisplacementControl", ctrl_node, 1, d_step*0.1)
            for _ in range(10):
                ok = ops.analyze(1)
                if ok == 0: break
            ops.integrator("DisplacementControl", ctrl_node, 1, d_step)

        if ok != 0:
            print(f"  ✗ Stopped at step {step} — drift = "
                  f"{disp[-1]/Htot*100:.2f}%")
            break

        ux = ops.nodeDisp(ctrl_node, 1)

        # Base shear from reactions
        ops.reactions()
        V = -sum(ops.nodeReaction(ntag(0,iz,ix), 1)
                 for iz in range(nColZ)
                 for ix in range(nColX))
        disp.append(ux); shear.append(V)

    return np.array(disp), np.array(shear), ctrl_node


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT COLLECTOR  (for 3D plots)
# ══════════════════════════════════════════════════════════════════════════════
def get_segs(deformed=False, scale=1.0):
    segs = []
    for ele in ops.getEleTags():
        ni, nj = ops.eleNodes(ele)
        xi,yi,zi = ops.nodeCoord(ni); xj,yj,zj = ops.nodeCoord(nj)
        if deformed:
            xi+=scale*ops.nodeDisp(ni,1); yi+=scale*ops.nodeDisp(ni,2)
            zi+=scale*ops.nodeDisp(ni,3)
            xj+=scale*ops.nodeDisp(nj,1); yj+=scale*ops.nodeDisp(nj,2)
            zj+=scale*ops.nodeDisp(nj,3)
        dx=xj-xi; dy=yj-yi; dz=zj-zi
        if abs(dy)>max(abs(dx),abs(dz)):
            ym=(yi+yj)/2
            clr=C["col_s1"] if ym<H[0]+0.1 else (C["col_s2"] if ym<H[0]+H[1]+0.1 else C["col_s3"])
        elif abs(dz)<0.01: clr=C["beam_x"]
        else:              clr=C["beam_z"]
        segs.append(((xi,zi,yi),(xj,zj,yj),clr))
    return segs


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def fig_pre():
    # Wide canvas: structure 66%, legend 28%, margins
    fig = plt.figure(figsize=(18, 9), facecolor="white")
    ax  = fig.add_axes([0.02, 0.06, 0.64, 0.88], projection="3d")
    _style_ax3d(ax, "PRE-PROCESSING  ·  3-Story 3D Steel MRF  (SI)")
    ax.view_init(elev=22, azim=-50)

    # Frame elements
    for (x1,z1,y1),(x2,z2,y2),clr in get_segs():
        lw = 3.5 if clr in (C["col_s1"],C["col_s2"],C["col_s3"]) else 2.5
        ax.plot([x1,x2],[z1,z2],[y1,y2], color=clr, linewidth=lw,
                solid_capstyle="round", alpha=0.95)

    # Supports
    for iz in range(nColZ):
        for ix in range(nColX):
            x,y,z = node_xyz(0,iz,ix)
            ax.scatter(x,z,y, s=90, color=C["support"],
                       marker="^", zorder=5, depthshade=False)

    # Lateral load arrows (roof)
    for iz in range(nColZ):
        for ix in range(nColX):
            x,y,z = node_xyz(nStory,iz,ix)
            ax.quiver(x-0.7,z,y, 0.65,0,0, color=C["load_lat"],
                      lw=2.0, arrow_length_ratio=0.28, alpha=0.90)

    # Gravity arrows (floor 1)
    for iz in range(nColZ):
        for ix in range(nColX):
            x,y,z = node_xyz(1,iz,ix)
            ax.quiver(x,z,y+0.6, 0,0,-0.55, color=C["load_grav"],
                      lw=1.4, arrow_length_ratio=0.28, alpha=0.80)

    span = nBayX*Lbay
    ax.set_xlim(0,span); ax.set_ylim(0,span); ax.set_zlim(0,Htot+0.5)
    ax.set_box_aspect([nBayX, nBayZ, Htot/(nBayX*Lbay)])

    # ── LEGEND PANEL — right 28% ─────────────────────────────────────────────
    ax_leg = fig.add_axes([0.70, 0.06, 0.28, 0.88])
    _style_panel(ax_leg)

    ax_leg.text(0.50, 0.975, "Legend",
                transform=ax_leg.transAxes,
                fontsize=14, fontweight="bold",
                color="black", ha="center", va="top")
    ax_leg.axhline(y=0.945, xmin=0.04, xmax=0.96,
                   color=C["border_lt"], linewidth=1.2)

    items = [
        (C["col_s1"],   "Column  Story 1\nW14×82",     "line",  5.0),
        (C["col_s2"],   "Column  Story 2\nW14×48",     "line",  5.0),
        (C["col_s3"],   "Column  Story 3\nW14×38",     "line",  5.0),
        (C["beam_x"],   "Beam  X-dir\nW24×55 / W21×44","line",  4.0),
        (C["beam_z"],   "Beam  Z-dir\nW21×44 / W18×35","line",  4.0),
        (C["load_grav"],"Gravity load\n1.2D + 1.6L",   "arrow", None),
        (C["load_lat"], "Lateral load\nInv.-Δ  (+X)",  "arrow", None),
        (C["support"],  "Fixed support",                 "tri",   None),
    ]
    y_positions = np.linspace(0.89, 0.18, len(items))

    for (clr, lbl, kind, lw), yp in zip(items, y_positions):
        if kind == "line":
            ax_leg.plot([0.05,0.28],[yp,yp], color=clr, linewidth=lw,
                        solid_capstyle="round",
                        transform=ax_leg.transAxes, clip_on=False)
        elif kind == "arrow":
            ax_leg.annotate("",
                xy=(0.28,yp), xytext=(0.05,yp),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=clr,
                                lw=2.2, mutation_scale=12))
        elif kind == "tri":
            ax_leg.add_patch(plt.Polygon(
                [[0.10,yp-0.023],[0.23,yp-0.023],[0.165,yp+0.023]],
                transform=ax_leg.transAxes,
                facecolor=clr, alpha=0.25,
                edgecolor=clr, linewidth=1.5, clip_on=False))
        ax_leg.text(0.32, yp, lbl,
                    transform=ax_leg.transAxes,
                    fontsize=11, color="black",
                    va="center", ha="left", linespacing=1.5)

    # Geometry + analysis note
    note = (
        f"Plan: {nBayX}×{nBayZ} bays   L = {Lbay:.0f} m\n"
        f"H₁={H[0]:.1f} m   H₂={H[1]:.1f} m   H₃={H[2]:.1f} m\n"
        f"Htot = {Htot:.1f} m\n\n"
        "Material: A992 Steel\n"
        "Fy = 345 MPa   E = 200 GPa\n\n"
        "Element: dispBeamColumn\n"
        "Section: Fiber + Aggregator\n"
        "Transf.: Corotational"
    )
    ax_leg.text(0.50, 0.015, note,
                transform=ax_leg.transAxes,
                fontsize=10, color="black",
                ha="center", va="bottom", linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.50",
                          facecolor=C["bg_box"],
                          edgecolor=C["border_lt"],
                          linewidth=1.0))

    fig.text(0.50, 0.003,
             "Units: kN · m   |   AISC 360-22 · ASCE 7-22 · FEMA 356",
             ha="center", fontsize=11, color="black", style="italic")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — PUSHOVER CURVE
# ══════════════════════════════════════════════════════════════════════════════
def fig_pushover(disp, shear):
    drift_pct = disp / Htot * 100.0
    idx_pk    = int(np.argmax(shear))

    # Wide layout: curve axes 64%, performance table 30%, gap 6%
    fig = plt.figure(figsize=(16, 8), facecolor="white")
    ax  = fig.add_axes([0.07, 0.11, 0.58, 0.82])
    _style_ax2d(ax,
                "Pushover Curve  ·  Base Shear vs Roof Drift Ratio",
                "Roof Drift Ratio   Δ/H  (%)",
                "Base Shear   V  (kN)")

    # Performance bands — light fills
    x_max = max(drift_pct[-1] * 1.08, 5.5)
    for x0, x1, clr, alpha in [
            (0.0, 1.0, C["io_clr"], 0.08),
            (1.0, 2.0, C["ls_clr"], 0.08),
            (2.0, 4.0, C["cp_clr"], 0.06)]:
        ax.axvspan(x0, min(x1, x_max), color=clr, alpha=alpha, zorder=0)
        if x1 <= x_max:
            ax.axvline(x1, color=clr, linewidth=1.2,
                       linestyle="--", alpha=0.65, zorder=1)

    # Coloured pushover curve
    pts  = np.array([drift_pct, shear]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(drift_pct.min(), drift_pct.max())
    lc   = LineCollection(segs, cmap=C["curve_cmap"], norm=norm,
                          linewidth=3.2, zorder=4, capstyle="round")
    lc.set_array(drift_pct[:-1])
    ax.add_collection(lc)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=C["curve_cmap"])
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.82, aspect=26)
    cb.set_label("Roof Drift  (%)", color="black", fontsize=11)
    cb.ax.yaxis.set_tick_params(color="black", labelsize=10)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black")
    cb.outline.set_edgecolor(C["border"])

    # Peak marker
    ax.scatter(drift_pct[idx_pk], shear[idx_pk],
               color=C["peak_clr"], s=130, zorder=7,
               edgecolors="black", linewidths=1.4)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, max(shear) * 1.22 if max(shear) > 0 else 1)
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))

    # ── Performance level table ───────────────────────────────────────────────
    ax_tbl = fig.add_axes([0.70, 0.11, 0.28, 0.82])
    _style_panel(ax_tbl)

    ax_tbl.text(0.50, 0.975, "Performance\nLevels",
                transform=ax_tbl.transAxes,
                fontsize=13, fontweight="bold",
                color="black", ha="center", va="top")
    ax_tbl.axhline(y=0.920, xmin=0.05, xmax=0.95,
                   color=C["border_lt"], linewidth=1.2)

    rows = [
        (C["io_clr"],   "IO", "Immediate\nOccupancy",  "< 1.0 %"),
        (C["ls_clr"],   "LS", "Life Safety",            "1.0 – 2.0 %"),
        (C["cp_clr"],   "CP", "Collapse\nPrevention",  "2.0 – 4.0 %"),
        (C["peak_clr"], "▲",
         f"Peak shear\n{shear[idx_pk]:.0f} kN",
         f"@ {drift_pct[idx_pk]:.2f} %"),
    ]
    for (clr, code, desc, dr_rng), ry in zip(rows, [0.82, 0.63, 0.44, 0.24]):
        # Badge
        ax_tbl.text(0.17, ry, code,
                    transform=ax_tbl.transAxes,
                    fontsize=12, fontweight="bold", color=clr,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.32",
                              facecolor=C["bg_box"],
                              edgecolor=clr, linewidth=1.5))
        # Description
        ax_tbl.text(0.33, ry + 0.045, desc,
                    transform=ax_tbl.transAxes,
                    fontsize=10.5, color="black",
                    ha="left", va="center")
        # Drift range
        ax_tbl.text(0.33, ry - 0.058, dr_rng,
                    transform=ax_tbl.transAxes,
                    fontsize=10, color=clr,
                    ha="left", va="center", style="italic")
        ax_tbl.axhline(y=ry - 0.115, xmin=0.05, xmax=0.95,
                       color=C["border_lt"], linewidth=0.6, alpha=0.8)

    ax_tbl.text(0.50, 0.020, "Ref: FEMA 356 — Table C1-3",
                transform=ax_tbl.transAxes,
                fontsize=9, color="black",
                ha="center", va="bottom", style="italic")

    fig.text(0.50, 0.006,
             "Units: kN · m   |   FEMA 356 §3.3.3.2",
             ha="center", fontsize=10.5, color="black", style="italic")
    return fig
# ══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — 3D DEFORMED SHAPE
# ══════════════════════════════════════════════════════════════════════════════
def fig_deformed(scale=5.0):
    fig = plt.figure(figsize=(16, 8), facecolor="white")
    fig.suptitle(
        f"POST-PROCESSING  ·  3D Deformed Shape  (×{scale})  |  "
        f"Target Drift = 5% H",
        color="black", fontsize=14, fontweight="bold", y=0.98)

    ux_all = [abs(ops.nodeDisp(t, 1)) for t in ops.getNodeTags()]
    ux_max = max(ux_all) if max(ux_all) > 0 else 1e-6
    cmap_d = plt.cm.viridis
    norm_d = plt.Normalize(0, ux_max)

    # ── LEFT: isometric 3D ────────────────────────────────────────────────────
    ax1 = fig.add_axes([0.02, 0.06, 0.44, 0.88], projection="3d")
    _style_ax3d(ax1, "Isometric View")
    ax1.view_init(elev=22, azim=-50)

    # Ghost undeformed
    for (x1,z1,y1),(x2,z2,y2),_ in get_segs():
        ax1.plot([x1,x2],[z1,z2],[y1,y2], color=C["ghost"],
                 linewidth=1.0, linestyle="--", alpha=0.55)

    # Deformed coloured
    for ele in ops.getEleTags():
        ni, nj = ops.eleNodes(ele)
        xi,yi,zi = ops.nodeCoord(ni); xj,yj,zj = ops.nodeCoord(nj)
        xi += scale*ops.nodeDisp(ni,1); yi += scale*ops.nodeDisp(ni,2)
        zi += scale*ops.nodeDisp(ni,3)
        xj += scale*ops.nodeDisp(nj,1); yj += scale*ops.nodeDisp(nj,2)
        zj += scale*ops.nodeDisp(nj,3)
        um = (abs(ops.nodeDisp(ni,1)) + abs(ops.nodeDisp(nj,1))) / 2
        ax1.plot([xi,xj],[zi,zj],[yi,yj],
                 color=cmap_d(norm_d(um)), linewidth=2.8,
                 solid_capstyle="round", alpha=0.95)

    span = nBayX * Lbay
    ax1.set_xlim(0, span); ax1.set_ylim(0, span)
    ax1.set_zlim(0, Htot + 0.5)
    ax1.set_box_aspect([nBayX, nBayZ, Htot/(nBayX*Lbay)])

    sm = ScalarMappable(norm=norm_d, cmap=cmap_d); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, shrink=0.68, pad=0.04, aspect=22)
    cb.set_label("|Ux|  (m)", color="black", fontsize=11)
    cb.outline.set_edgecolor(C["border"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black", fontsize=9)

    # ── RIGHT: front elevation (Z=0), annotated drifts ────────────────────────
    ax2 = fig.add_axes([0.54, 0.10, 0.38, 0.84])
    _style_ax2d(ax2, "Front Elevation  (Z = 0  frame)", "X  (m)", "Y  (m)")
    ax2.set_aspect("equal")

    # Collect front frame topology
    front = []
    for iz in range(nColZ):
        for s in range(nStory):
            for ix in range(nColX):
                front.append((ntag(s, iz, ix), ntag(s+1, iz, ix)))
        for s in range(1, nStory+1):
            for ix in range(nColX-1):
                front.append((ntag(s, iz, ix), ntag(s, iz, ix+1)))

    # Ghost
    for ni, nj in front:
        xi,yi,_ = ops.nodeCoord(ni); xj,yj,_ = ops.nodeCoord(nj)
        ax2.plot([xi,xj],[yi,yj], color=C["ghost"],
                 linewidth=1.4, linestyle="--", alpha=0.75, zorder=1)

    # Deformed coloured
    for ni, nj in front:
        xi,yi,_ = ops.nodeCoord(ni); xj,yj,_ = ops.nodeCoord(nj)
        xi += scale*ops.nodeDisp(ni,1); yi += scale*ops.nodeDisp(ni,2)
        xj += scale*ops.nodeDisp(nj,1); yj += scale*ops.nodeDisp(nj,2)
        um = (abs(ops.nodeDisp(ni,1)) + abs(ops.nodeDisp(nj,1))) / 2
        ax2.plot([xi,xj],[yi,yj], color=cmap_d(norm_d(um)),
                 linewidth=3.2, solid_capstyle="round", zorder=4)

    # Story drift annotations — RIGHT margin, well clear of structure
    x_struct_max = nBayX * Lbay        # right edge of structure
    x_ann_start  = x_struct_max + 0.8  # start of leader line
    x_ann_text   = x_struct_max + 0.95 # start of text box
    h_c = [0] + list(np.cumsum(H))
    for s in range(nStory):
        dr   = abs(ops.nodeDisp(ntag(s+1,0,0),1) - ops.nodeDisp(ntag(s,0,0),1))
        pct  = dr / H[s] * 100
        y_mid = (h_c[s] + h_c[s+1]) / 2
        # Leader line from structure edge to annotation
        ax2.annotate("",
            xy=(x_struct_max + 0.05, y_mid),
            xytext=(x_ann_start, y_mid),
            arrowprops=dict(arrowstyle="-", color=C["deformed"],
                            lw=0.9, linestyle="dotted"))
        ax2.text(x_ann_text, y_mid,
                 f"Story {s+1}\nΔ = {dr*1000:.1f} mm\nIDR = {pct:.2f} %",
                 fontsize=10, color="black", va="center",
                 bbox=dict(boxstyle="round,pad=0.35",
                           facecolor=C["bg_box"],
                           edgecolor=C["deformed"],
                           linewidth=1.2))

    ax2.set_xlim(-0.5, x_struct_max + 3.2)
    ax2.set_ylim(-0.6, Htot + 0.8)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96], pad=1.6)
    return fig
# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — STORY DRIFT PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def fig_drift_profile():
    fig, ax = plt.subplots(figsize=(7,8), facecolor="white")
    _style_ax2d(ax,"Story Drift Profile  ·  At Target Displacement",
                "Inter-Story Drift Ratio  (%)","Floor Level")

    h_c=[0]+list(np.cumsum(H))
    ux=[np.mean([ops.nodeDisp(ntag(s,iz,ix),1)
                  for iz in range(nColZ) for ix in range(nColX)])
        for s in range(nStory+1)]
    idr=[max(abs(ux[s+1]-ux[s])/H[s]*100,1e-6) for s in range(nStory)]

    ax.axvline(1.0,color=C["io_clr"],lw=1.2,ls="--",alpha=0.7,label="IO 1.0%")
    ax.axvline(2.0,color=C["ls_clr"],lw=1.2,ls="--",alpha=0.7,label="LS 2.0%")
    ax.axvline(4.0,color=C["cp_clr"],lw=1.2,ls="--",alpha=0.7,label="CP 4.0%")

    cmap_dr=plt.cm.RdYlGn_r; norm_dr=plt.Normalize(0,4.0)
    for s in range(nStory):
        ym=(h_c[s]+h_c[s+1])/2
        ax.barh(ym,idr[s],height=H[s]*0.70,
                color=cmap_dr(norm_dr(idr[s])),alpha=0.75,
                edgecolor=C["border"],linewidth=0.8)
        ax.text(idr[s]+0.06,ym,f"{idr[s]:.2f}%",
                fontsize=9,color=C["fg"],va="center",fontweight="bold")
    ax.plot(idr,[(h_c[s]+h_c[s+1])/2 for s in range(nStory)],
            color=C["deformed"],lw=1.8,marker="o",ms=7,
            mfc="white",mec=C["deformed"],mew=1.5,zorder=5)
    ax.set_yticks([(h_c[s]+h_c[s+1])/2 for s in range(nStory)])
    ax.set_yticklabels([f"Story {s+1}" for s in range(nStory)],fontsize=9)
    ax.set_xlim(0,max(max(idr)*1.35,4.5)); ax.set_ylim(-0.3,Htot+0.3)
    ax.legend(loc="lower right",fontsize=8,
              facecolor=C["bg_panel"],edgecolor=C["border"])
    sm=ScalarMappable(norm=norm_dr,cmap=cmap_dr); sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,shrink=0.75,pad=0.03)
    cb.set_label("IDR (%)",color=C["fg_dim"],fontsize=8)
    cb.outline.set_edgecolor(C["border"])
    plt.setp(cb.ax.yaxis.get_ticklabels(),color=C["fg_dim"])
    fig.tight_layout(pad=1.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — STORY DRIFT PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def fig_drift_profile():
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    _style_ax2d(ax,
                "Story Drift Profile  ·  At Target Displacement",
                "Inter-Story Drift Ratio  (%)",
                "Floor Level")

    h_c = [0] + list(np.cumsum(H))
    ux  = [np.mean([ops.nodeDisp(ntag(s, iz, ix), 1)
                    for iz in range(nColZ) for ix in range(nColX)])
           for s in range(nStory + 1)]
    idr = [max(abs(ux[s+1] - ux[s]) / H[s] * 100, 1e-6) for s in range(nStory)]

    # FEMA 356 limit lines
    for limit, label, clr in [
            (1.0, "IO  1.0 %", C["io_clr"]),
            (2.0, "LS  2.0 %", C["ls_clr"]),
            (4.0, "CP  4.0 %", C["cp_clr"])]:
        ax.axvline(limit, color=clr, lw=1.5, ls="--", alpha=0.75, label=label)

    # Horizontal bars coloured by IDR severity
    cmap_dr = plt.cm.RdYlGn_r
    norm_dr = plt.Normalize(0, 4.0)
    for s in range(nStory):
        ym  = (h_c[s] + h_c[s+1]) / 2
        clr = cmap_dr(norm_dr(idr[s]))
        ax.barh(ym, idr[s], height=H[s]*0.68,
                color=clr, alpha=0.78,
                edgecolor=C["border"], linewidth=1.0)
        # IDR value label — placed right of bar, black text
        ax.text(idr[s] + 0.10, ym,
                f"{idr[s]:.2f} %",
                fontsize=12, color="black",
                va="center", fontweight="bold")

    # Connect bar midpoints with a line
    ax.plot(idr, [(h_c[s]+h_c[s+1])/2 for s in range(nStory)],
            color=C["deformed"], lw=2.0,
            marker="o", ms=8,
            mfc="white", mec=C["deformed"], mew=2.0, zorder=5)

    ax.set_yticks([(h_c[s]+h_c[s+1])/2 for s in range(nStory)])
    ax.set_yticklabels([f"Story {s+1}" for s in range(nStory)],
                       fontsize=12, color="black")
    ax.set_xlim(0, max(max(idr)*1.40, 5.0))
    ax.set_ylim(-0.4, Htot + 0.4)

    ax.legend(loc="lower right", fontsize=11,
              facecolor=C["bg_panel"], edgecolor=C["border"])

    sm = ScalarMappable(norm=norm_dr, cmap=cmap_dr); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.76, pad=0.02)
    cb.set_label("IDR  (%)", color="black", fontsize=11)
    cb.outline.set_edgecolor(C["border"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black", fontsize=10)

    fig.tight_layout(pad=1.6)
    return fig
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n"+"═"*65)
    print(f"  3D STEEL MRF — NONLINEAR PUSHOVER  (SI: kN, m)")
    print(f"  Plan: {nBayX}×{nBayZ} bays · {nStory} stories · Htot={Htot:.1f}m")
    print(f"  dispBeamColumn · Aggregator sect · Corotational · FEMA 356")
    print("═"*65)

    n_ele = build_model()
    print(f"\n  Model: {len(ops.getNodeTags())} nodes · {n_ele} elements")
    fig1 = fig_pre()

    print("\n  [1/3] Gravity …")
    ok = apply_gravity()
    print(f"        {'OK ✓' if ok==0 else 'FAILED ✗'}")

    print("  [2/3] Pushover (target 5% Htot) …")
    disp, shear, ctrl = run_pushover(target_pct=0.05)
    print(f"        Steps   : {len(disp)-1}")
    print(f"        Peak V  : {max(shear):.1f} kN  @  "
          f"{disp[int(np.argmax(shear))]/Htot*100:.2f}%")

    print("  [3/3] Plots …")
    fig2 = fig_pushover(disp, shear)
    fig3 = fig_deformed(scale=5.0)
    fig4 = fig_drift_profile()

    print("\n"+"═"*65)
    print("  STORY DRIFT SUMMARY")
    print("─"*65)
    h_c=[0]+list(np.cumsum(H))
    for s in range(nStory):
        dr=abs(ops.nodeDisp(ntag(s+1,0,0),1)-ops.nodeDisp(ntag(s,0,0),1))
        pct=dr/H[s]*100
        print(f"  Story {s+1}: IDR={pct:.2f}%  Δ={dr*1000:.1f}mm  "
              f"{'OK ✓' if pct<2.0 else 'EXCEEDS LS ✗'}")
    print(f"\n  Peak base shear = {max(shear):.1f} kN")
    print("═"*65+"\n")

    global __viewer__
    __viewer__ = {
        "sections":[
            {"tag":FIBER_TAG["W14x82"],"color":"#1d4ed8","label":"W14×82 Col S1"},
            {"tag":FIBER_TAG["W14x48"],"color":"#3b82f6","label":"W14×48 Col S2"},
            {"tag":FIBER_TAG["W14x38"],"color":"#93c5fd","label":"W14×38 Col S3"},
            {"tag":FIBER_TAG["W24x55"],"color":"#047857","label":"W24×55 Bm-X"},
            {"tag":FIBER_TAG["W21x44"],"color":"#0891b2","label":"W21×44 Bm-Z/X"},
            {"tag":FIBER_TAG["W18x35"],"color":"#06b6d4","label":"W18×35 Bm-Z S3"},
        ],
        "precision":3,
    }
    plt.show()


if __name__ == "__main__":
    main()

# DISCLAIMER: Not a licensed PE. All real-world designs require PE review.