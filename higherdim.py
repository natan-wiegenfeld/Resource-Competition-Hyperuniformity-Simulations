import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
# import matplotlib.animation as animation
from numpy import random
from numba import njit, prange, vectorize, float64
import math
import csv
import os
import sys

from numba import get_num_threads, set_num_threads

import time



L = 7 #System length
dx = 0.0125 #Lattice Spacing
p = 1 #regeneration rate 
D = 1 #Diffusion Coefficient
k = 7  #inverse Cell Consumption rate
lam = 1 #Cell Deposition rate
yc = 1 #Critical Concentration
T=1000 #Simulation Run Time
dt = 0.05*dx**2/(D) #Time Step - condition for numerical stability given in terms of lattice spacing
nu0 = -20 #Initial Cell Vitality
threads = 4 #Number of threads for numba - increasing this too much will slow down code due to diminishing returns and parallelization overheads
output_dir = f"2D\\Length_{L}_D_{D}_K_{k}_yc_{yc}_Lambda_{lam}" #Output directory relative to this file directory for output files and graphs


set_num_threads(int(threads)) 

N = int(L/dx) #number of lattice sites


# Calculates Finite Element Laplacian for Diffusion
@njit(parallel=True, fastmath=True)
def Diff2Arr2D(F, dx):
    N = F.shape[0]
    
    #square array check:
    # if F.shape[1] != N:
    #     raise ValueError("Diff2Arr2D expects a square array (N x N).")

    # coefficients (float to avoid integer division)
    c1 = 1.6
    c2 = 0.2
    c3 = 8.0/315.0
    c4 = 1.0/560.0
    c0 = 205.0/72.0
    inv_dx2 = 1.0/(dx*dx)

    out = np.empty_like(F)

    # ---- D_xx pass (operate along axis=1, row by row), write into out ----
    for j in prange(N):
        # left edge block (i = 0..3), use wrap only where needed
        for i in range(4):
            i1 = i + 1
            i2 = i + 2
            i3 = i + 3
            i4 = i + 4
            im1 = (i - 1) % N
            im2 = (i - 2) % N
            im3 = (i - 3) % N
            im4 = (i - 4) % N

            f_next = F[j, i1]*c1 - F[j, i2]*c2 + F[j, i3]*c3 - F[j, i4]*c4
            f_prev = F[j, im1]*c1 - F[j, im2]*c2 + F[j, im3]*c3 - F[j, im4]*c4
            out[j, i] = (f_next + f_prev - F[j, i]*c0) * inv_dx2

        # middle block (i = 4..N-5), no wrap
        for i in range(4, N-4):
            i1 = i + 1
            i2 = i + 2
            i3 = i + 3
            i4 = i + 4
            im1 = i - 1
            im2 = i - 2
            im3 = i - 3
            im4 = i - 4

            f_next = F[j, i1]*c1 - F[j, i2]*c2 + F[j, i3]*c3 - F[j, i4]*c4
            f_prev = F[j, im1]*c1 - F[j, im2]*c2 + F[j, im3]*c3 - F[j, im4]*c4
            out[j, i] = (f_next + f_prev - F[j, i]*c0) * inv_dx2

        # right edge block (i = N-4..N-1), wrap only for the + side
        for i in range(N-4, N):
            i1 = (i + 1) % N
            i2 = (i + 2) % N
            i3 = (i + 3) % N
            i4 = (i + 4) % N
            im1 = i - 1
            im2 = i - 2
            im3 = i - 3
            im4 = i - 4

            f_next = F[j, i1]*c1 - F[j, i2]*c2 + F[j, i3]*c3 - F[j, i4]*c4
            f_prev = F[j, im1]*c1 - F[j, im2]*c2 + F[j, im3]*c3 - F[j, im4]*c4
            out[j, i] = (f_next + f_prev - F[j, i]*c0) * inv_dx2

    # ---- D_yy pass (operate along axis=0, column by column), add into out ----
    for i in prange(N):
        # top edge block (j = 0..3), wrap only where needed
        for j in range(4):
            j1 = j + 1
            j2 = j + 2
            j3 = j + 3
            j4 = j + 4
            jm1 = (j - 1) % N
            jm2 = (j - 2) % N
            jm3 = (j - 3) % N
            jm4 = (j - 4) % N

            f_next = F[j1, i]*c1 - F[j2, i]*c2 + F[j3, i]*c3 - F[j4, i]*c4
            f_prev = F[jm1, i]*c1 - F[jm2, i]*c2 + F[jm3, i]*c3 - F[jm4, i]*c4
            out[j, i] += (f_next + f_prev - F[j, i]*c0) * inv_dx2

        # middle block (j = 4..N-5), no wrap
        for j in range(4, N-4):
            j1 = j + 1
            j2 = j + 2
            j3 = j + 3
            j4 = j + 4
            jm1 = j - 1
            jm2 = j - 2
            jm3 = j - 3
            jm4 = j - 4

            f_next = F[j1, i]*c1 - F[j2, i]*c2 + F[j3, i]*c3 - F[j4, i]*c4
            f_prev = F[jm1, i]*c1 - F[jm2, i]*c2 + F[jm3, i]*c3 - F[jm4, i]*c4
            out[j, i] += (f_next + f_prev - F[j, i]*c0) * inv_dx2

        # bottom edge block (j = N-4..N-1), wrap only for the + side
        for j in range(N-4, N):
            j1 = (j + 1) % N
            j2 = (j + 2) % N
            j3 = (j + 3) % N
            j4 = (j + 4) % N
            jm1 = j - 1
            jm2 = j - 2
            jm3 = j - 3
            jm4 = j - 4

            f_next = F[j1, i]*c1 - F[j2, i]*c2 + F[j3, i]*c3 - F[j4, i]*c4
            f_prev = F[jm1, i]*c1 - F[jm2, i]*c2 + F[jm3, i]*c3 - F[jm4, i]*c4
            out[j, i] += (f_next + f_prev - F[j, i]*c0) * inv_dx2

    return out


#Computes discrete dy/dt for step
@njit(parallel=True, fastmath=True)
def computeYDer2D(y, cells, dx, p, D, k, y_buffer):
    N = y.shape[0]
    d2y = Diff2Arr2D(y, dx)      # Laplacian with ±4 stencil
    k_area = 1 / (k*dx*dx)       #2D delta normalization (dx=dy)

    for j in prange(N):
        for i in range(N):
            active = 1.0 if cells[j, i] < 0.0 else 0.0
            # delta-like sink/source localized where active==1
            reaction = p * (1.0 - y[j, i] * active * k_area)
            y_buffer[j, i] = reaction + D * d2y[j, i]

    return y_buffer


#Euler method step - could be replaced with RK4
@njit
def RK(y, cells, dx, dt, p, D, k, y_buffer):
    computeYDer2D(y, cells, dx, p, D, k, y_buffer)
    y += dt *y_buffer

#Implement deposition and removal of cells, evolution of viability parameter
@njit(parallel=True, fastmath=False)
def update_cells_2d(cells, cell_born, y, dt, dx, yc, lam, nu0, t):
    N  = cells.shape[0]
    NN = N * N

    mu = lam * dx * dx * dt
    p_birth = -np.expm1(-mu)  # stable 1 - exp(-mu)

    #More complicated calculation for numerical accuracy
    if p_birth < 1e-3:
        logq = math.log1p(-p_birth)
        idx = -1
        while True:
            u = np.random.random()
            jump = 1 + int(math.floor(math.log(u) / logq))
            idx += jump
            if idx >= NN:
                break
            j = idx // N
            i = idx - j * N
            cells[j, i] = nu0
            cell_born[j, i] = t
    else:
        for j in prange(N):
            for i in range(N):
                if np.random.random() < p_birth:
                    cells[j, i] = nu0
                    cell_born[j, i] = t

    # viability update
    for j in prange(N):
        for i in range(N):
            c = cells[j, i]
            if c < 0.0:
                val = c + (c*c + yc - y[j, i]) * dt
                cells[j, i] = val if val < 0.0 else 0.0
            else:
                cells[j, i] = 0.0



def update(y, cells, cell_born, dx, dt, p, D, k, yc, lam, nu0, t, y_buffer):
    #Step Resource field according to cell locations
    RK(y, cells, dx, dt, p, D, k, y_buffer)
    #Add new cells, update vitality of each cell, and remove dead ones
    update_cells_2d(cells, cell_born, y, dt, dx, yc, lam, nu0, t)


#Saving Functions
def clear_file(rel_dir: str, filename: str) -> None:
    """
    Empties (or creates) the file at rel_dir/filename, relative to this script.
    """
    # Resolve paths
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    target_dir  = os.path.join(script_dir, rel_dir)
    target_path = os.path.join(target_dir, filename)

    # Make sure the directory exists
    os.makedirs(target_dir, exist_ok=True)
    # Open in write mode and immediately close → truncates file
    open(target_path, 'w').close()


def append_array_with_time(
    array: np.ndarray,
    time_step,
    rel_dir: str,
    filename: str
) -> None:
    """
    Appends each element of `array` to the file at rel_dir/filename,
    formatted as '<element>\\t<time_step>\\n'.

    rel_dir is interpreted as a path relative to this script's location.
    """
    # Determine the folder containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the target directory + file path
    target_dir = os.path.join(script_dir, rel_dir)
    target_path = os.path.join(target_dir, filename)

    # (Optional) make sure the directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Flatten the array (in case it's multi-dimensional)
    flat = np.ravel(array)

    # Append each element + tab + time + newline
    with open(target_path, 'a') as f:
        for elem in flat:
            f.write(f"{elem}\t{time_step}\n")


params = {
    "L": L, "dx": dx, "p": p, "D": D, "k": k, "lam": lam,
    "yc": yc, "T": T, "dt": dt, "nu0": nu0, "threads": threads,
    "N": N
}

#Log Simulation Parameters
def write_sim_params(output_dir: str, filename: str, params: dict, append: bool=False) -> str:
    """
    Write key/value pairs to: <script_dir>/<output_dir>/<rel_dir>/<filename>
    as tab-separated lines: 'key<TAB>value\\n'.

    - output_dir: base directory name/string (e.g., your f"Length_{...}" string)
    - rel_dir:    path *relative to output_dir* (e.g., "meta", "logs/run1", "")
    - filename:   e.g., "params.tsv"
    - params:     dict of parameters to write
    - append:     if True, append; otherwise overwrite
    Returns the absolute path to the written file.
    """
    # Resolve base directory: script location if available; cwd otherwise
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    target_dir = os.path.join(base_dir, output_dir)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    mode = "a" if append else "w"
    with open(target_path, mode) as f:
        if not append:
            f.write(f"# written {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for k, v in params.items():
            f.write(f"{k}\t{v}\n")

    return target_path

path = write_sim_params(output_dir, filename="params.tsv", params=params, append=False)
print("params saved to:", path)

#System data initialization
# 2D state arrays (same names as your 1D version, now shape (Ny, Nx))
cells      = np.zeros((N, N))
cell_born  = np.zeros_like(cells)
y          = np.zeros_like(cells)
y_buffer   = np.zeros_like(cells)

x_nodes = np.arange(N) * dx   # 0, dx, 2dx, ..., L-dx


#Uncomment along with second segment in main loop to enable graphics
# # --- Figure style ---
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'],
#     'font.size': 10,
#     'axes.labelsize': 10,
#     'axes.titlesize': 11,
#     'legend.fontsize': 9,
#     'xtick.labelsize': 9,
#     'ytick.labelsize': 9,
#     'lines.linewidth': 0.5,
#     'axes.linewidth': 1,
#     'text.usetex': False
# })

# fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=200)

# # --- Heatmap of y (red-blue), centered at yc ---
# # Start with a reasonable color range; adjust as you like.
# vmin, vmax = 0.5, 1.5
# norm = TwoSlopeNorm(vcenter=yc, vmin=vmin, vmax=vmax)

# im = ax.imshow(
#     y, origin='lower',
#     extent=[0, L, 0, L],
#     cmap='RdBu_r', norm=norm,
#     interpolation='nearest'
# )

# cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
# cbar.set_label('y(x, y, t)')

# # --- Cells as dots (cells < 0 are "alive") ---
# alive = (cells < 0.0)
# jj, ii = np.nonzero(alive)
# xs = (ii + 0.5) * dx   # center of cell (optional: use ii*dx if nodes)
# ys = (jj + 0.5) * dx
# cell_dots = ax.scatter(xs, ys, s=12, c='k', label='cells', zorder=5)


# # --- Axes cosmetics ---
# ax.set_xlabel('x'); ax.set_ylabel('y')
# ax.set_xlim(0, L); ax.set_ylim(0, L)
# ax.set_aspect('equal', 'box')
# ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
# ax.legend(loc='upper right', frameon=False)

# plt.tight_layout()
# plt.ion()
# plt.show(block=False)




frame_skip = math.floor(0.1/dt)

#Calculate some minor theoretical prediction
print("Mu-star  = ", ((math.pi*yc*lam)/(2*k))**2)
t = 0

def numba_loop(y, cells, cell_born, dx, dt, p, D, k, yc, lam, nu0, t, frame_skip, y_buffer):
    print(f"Numba is using {get_num_threads()} threads")

    step = 0
    total_steps = int(math.ceil(T / dt))
    prog_stride = max(1, total_steps // 10)  # print ~10 times
    clear_file(output_dir, "y_full.tsv")
    clear_file(output_dir,  "cell_full.tsv")
    while step < total_steps:
        # ---- actual simulation step ----
        update(y, cells, cell_born, dx, dt, p, D, k, yc, lam, nu0, t, y_buffer)

        # ---- progress print ----
        if step % prog_stride == 0:
            print(f"t = {t:.3f}")

        # ---- plotting (every frame_skip steps) ----
        if step % frame_skip == 0:
            if step % (frame_skip * 10) == 0:
                append_array_with_time(y, t, output_dir, "y_full.tsv")
                append_array_with_time(cells, t, output_dir,  "cell_full.tsv")
            
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            span = max(abs(ymax - yc), abs(ymin - yc))
            span = max(span, 1e-3)  # avoid zero width
            alive_n = int(np.count_nonzero(cells < 0.0))
            print(f"[dbg] t={t:.2f}, alive={alive_n}, y[min,max]=({ymin:.3f},{ymax:.3f})")


            #Uncomment for live graphics
            # im.set_data(y)
            # new_norm = TwoSlopeNorm(vcenter=yc, vmin=yc - span, vmax=yc + span)
            # im.set_norm(new_norm)
            # cbar.update_normal(im)  # <-- keeps the colorbar in sync
            # alive = (cells < 0.0)
            
            # if alive.any():
            #     jj, ii = np.nonzero(alive)
            #     xs = (ii + 0.5) * dx
            #     ys = (jj + 0.5) * dx
            #     cell_dots.set_offsets(np.c_[xs, ys])
            # else:
            #     cell_dots.set_offsets(np.empty((0, 2)))

            # ax.set_title(f"t = {t:.3f}")
            # plt.pause(0.001)

        # ---- advance time & counter (MUST be inside while) ----
        t += dt
        step += 1    




#Run main loop
numba_loop(y, cells, cell_born, dx, dt, p, D, k, yc, lam, nu0, t, frame_skip, y_buffer)


# plt.ioff()
# plt.show()
