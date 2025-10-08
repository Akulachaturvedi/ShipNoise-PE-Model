import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import os
import re
from scipy.interpolate import RegularGridInterpolator
from geopy.distance import geodesic
from matplotlib.widgets import Button
from peram.PeRAM import PeRAM
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime
from matplotlib.cm import ScalarMappable


# === Load ship data ===
# Add today's date to the output file name (YYYYMMDD format)
today_str = datetime.today().strftime("%Y%m%d")


# Get all matching files
files = glob.glob("Ship_FullSpectrum_Results_*.xlsx")

if not files:
    raise FileNotFoundError("No Ship_FullSpectrum_Results_YYYYMMDD.xlsx files found")

# Extract dates from filenames and sort
def extract_date(fname):
    m = re.search(r"(\d{8})", fname)  # look for YYYYMMDD
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")
    return datetime.min  # fallback if no date

# Pick the file with the latest date in the filename
file_path = max(files, key=extract_date)

print(f"[Loaded] {file_path}")


df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
df = df[df["F_500Hz"].notna()]

# Extract necessary fields
ship_names = df["Name"].astype(str).values  # Ensure string type


selected_freq = 500  # default
freq_col = f"F_{selected_freq}Hz"


# === Frequency spectrum columns ===
frequency_cols = [col for col in df.columns if col.startswith("F_") and col.endswith("Hz")]
frequencies = [int(col.split("_")[1][:-2]) for col in frequency_cols]
#freq_mask = [(10 <= f <= 10000) for f in frequencies]
frequencies = [int(col.split("_")[1][:-2]) for col in frequency_cols]
available_freqs = sorted(frequencies)


# === Select frequency ===
available_freqs = sorted(frequencies)
#print("Available Frequencies (Hz):", available_freqs)

freq_col = f"F_{selected_freq}Hz"
if freq_col not in df.columns:
    raise ValueError(f"[ERROR] Frequency {selected_freq} Hz not found in dataset.")

import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

print("Loading.... Vessel Traffic and Ocean Bathymetry Map ")

# === Load bathymetry ===
bathy = np.load("GlobalOcean_Bathymetry_LogDepth_Coarse.npz")
lon_bathy = bathy["longitude"]
lat_bathy = bathy["latitude"]
depth = bathy["depth"]
depth_masked = np.ma.masked_where(depth >= 0, depth)
lon_grid, lat_grid = np.meshgrid(lon_bathy, lat_bathy)

# === Ship arrays ===
ship_lons = df["Longitude"].values
ship_lats = df["Latitude"].values
ship_sl_selected = df[freq_col].values


bathy_mode = False


# === Marker shape map ===
marker_map = {
    "Containership": "o",
    "Tanker": "s",
    "Bulker": "^",
    "Vehicle Carrier": "d",
    "Other": "p",
    "Passenger": "*"
}


# === Extract date from file name ===
match = re.search(r"_(\d{8})\.xlsx$", file_path)
if match:
    date_str = match.group(1)  # '20250814'
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    date_formatted = date_obj.strftime("%d %b %Y")  # '14 Aug 2025'
else:
    date_formatted = "Unknown Date"


# === Create map ===
fig = plt.figure(figsize=(19, 9))
plt.subplots_adjust(
    top=0.956,
    bottom=0.04,
    left=0.143,
    right=0.992,
    hspace=0.2,
    wspace=0.2
)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.set_title(f"Visualizing Shipping Traffic on {date_formatted} with Ocean Bathymetry", fontsize=16, fontweight="bold")


# ==== Extents (lon_min, lon_max, lat_min, lat_max) ====
EXTENTS = {
    "Pacific West":  [95, 180, -60, 65],   # Asia/Aus side
    "Pacific East":  [-180, -70, -60, 65],  # Americas side
    "Atlantic":      [-70, 20, -60, 80],
    "Indian":        [20, 147, -60, 25],
    "European Waters": [-45, 70, 26, 90]
}

current_view = "Full Map"  # track current view

def set_view(name):
    """Set map to named view, or Full Map."""
    global current_view
    if name == "Full Map":
        ax.set_global()
    else:
        lon_min, lon_max, lat_min, lat_max = EXTENTS[name]
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    current_view = name
    fig.canvas.draw_idle()

def make_toggle_callback(name):
    def _cb(event):
        # Toggle: if already on this view, go back to Full Map
        set_view("Full Map" if current_view == name else name)
    return _cb

# ==== Buttons (stacked above the radio on the left) ====
btn_w, btn_h = 0.08, 0.05
x = 0.01
ys = [0.64, 0.58, 0.52, 0.46, 0.40, 0.34]  # top to bottom

labels = ["European Waters", "Pacific East", "Pacific West", "Atlantic", "Indian", "Full Map"]
callbacks = [
    make_toggle_callback("European Waters"),
    make_toggle_callback("Pacific East"),
    make_toggle_callback("Pacific West"),
    make_toggle_callback("Atlantic"),
    make_toggle_callback("Indian"),
    lambda evt: set_view("Full Map"),
]

buttons = []
for y, label, cb in zip(ys, labels, callbacks):
    bax = plt.axes([x, y, btn_w, btn_h])
    b = Button(bax, label, color='lightgray', hovercolor='0.8')
    b.on_clicked(cb)
    buttons.append(b)

toolbar = fig.canvas.manager.toolbar

def home_override(*args, **kwargs):
    set_view("Full Map")

toolbar.home = home_override


# === Add IRS Logo ===
logo_path = os.path.join("icons", "IRS.jpg")
if os.path.exists(logo_path):
    logo_img = mpimg.imread(logo_path)

    # Add a small inset axes at top-right (adjust numbers as needed)
    logo_ax = fig.add_axes([0.85, 0.87, 0.12, 0.1])  # [left, bottom, width, height] in figure coords
    logo_ax.imshow(logo_img)
    logo_ax.axis("off")  # No ticks/border
else:
    print("[Warning] IRS logo image not found at:", logo_path)

# === Add IIT Bombay Logo (top-left) ===
iitb_logo_path = os.path.join("icons", "IIT_Bombay_Logo.png")
if os.path.exists(iitb_logo_path):
    iitb_logo_img = mpimg.imread(iitb_logo_path)
    iitb_logo_ax = fig.add_axes([0.005, 0.87, 0.12, 0.1])  # adjust placement
    iitb_logo_ax.imshow(iitb_logo_img)
    iitb_logo_ax.axis("off")
else:
    print("[Warning] IIT Bombay logo image not found at:", iitb_logo_path)
    

# === Bathymetry background ===
bathy_plot = ax.pcolormesh(
    lon_grid, lat_grid, depth_masked,
    cmap="Blues_r", shading="auto",
    norm=Normalize(vmin=-6000, vmax=0), zorder=1
)
cb1 = plt.colorbar(bathy_plot, ax=ax, shrink=0.7, pad=0.02)
cb1.set_label("Depth (m)", fontsize=12, fontweight="bold")

# === Plot ships ===
scatter_handles = []
sc = None
vmin, vmax = 50, 200

for vessel_type, marker in marker_map.items():
    subset = df[df["Vessel Class"] == vessel_type]
    if subset.empty:
        continue
    sc_now = ax.scatter(
        subset["Longitude"], subset["Latitude"], c=subset[freq_col],
        cmap="jet", vmin=vmin, vmax=vmax,
        s=60, marker=marker, edgecolors="lightgray", linewidths=1.2, zorder=3
    )
    if sc is None:
        sc = sc_now
    scatter_handles.append(plt.Line2D([0], [0], marker=marker, linestyle='None',
                                      markerfacecolor='k', markeredgecolor='k',
                                      markersize=10, label=vessel_type))

cb2 = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.04)
cb2.set_label(f"Source Level @ {selected_freq} Hz (dB re 1 µPa m)", fontsize=12, fontweight="bold")

legend_title = plt.Line2D([], [], linestyle='None', label="Vessel Type")
ax.legend(handles=scatter_handles, title="Vessel Type", loc='upper right', fontsize=12, title_fontsize=12)


# === Map features ===
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1, zorder=4)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='--', linewidth=0.6, zorder=4)
ax.tick_params(labelsize=14)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')

# === Gridlines ===
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

# === Annotation box ===
annotation = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="w", alpha=0.95),
                         arrowprops=dict(arrowstyle="->"))
annotation.set_visible(False)

# === Global for tracking ===
spectrum_fig = None
bathy_mode = False          # Or True initially
previous_line = None
last_clicked_ship = None
tl_1d_fig = None
spl_fig = None
bathymetry_fig = []
bath_fig = None
tl_2d_fig = None
show_sel3m_flag = False
previous_lines = []  #  Track all plotted red lines
distance_labels = []
distance_texts = []
distance_lines_labels = []


def onclick(event):
    global show_sel3m_flag, spectrum_fig, previous_line, last_clicked_ship, bathymetry_fig, x_click, y_click
    
    # === Disable interaction if SPL/SEL mode is active ===
    if calc_active:
        print("[Info] SPL & SEL Calculation is active. Ignoring ship click.")
        return
    
    if event.inaxes != ax:
        return
    x_click, y_click = event.xdata, event.ydata
    if x_click is None or y_click is None:
        return

    # Find ship distanace
    distances = np.hypot(ship_lons - x_click, ship_lats - y_click)
    nearest_idx = np.argmin(distances)
    min_dist = distances[nearest_idx]
    ship = df.iloc[nearest_idx]


    # Define frequency column and get SL
    freq_col = f"F_{selected_freq}Hz"
    sl_at_freq = ship.get(freq_col, np.nan)
    last_clicked_ship = ship  #  Store nearest ship for use in PE model

    # === Normal Mode ===
    if min_dist < 0.3:
        name = ship["Name"]
        vessel_type = ship["Vessel Class"]
        lat = ship["Latitude"]
        lon = ship["Longitude"]
        speed = ship["Vessel Speed (kn)"]
        length = ship["Vessel Length (m)"]
        overall = ship["Overall Noise (dB re 1 µPa m)"]
        freq_col = f"F_{selected_freq}Hz"
        sl_at_freq = ship[freq_col]  # DEFINE THIS HERE

        text = (
            f"{name}\n"
            f"{vessel_type}\n"
            f"Lat: {lat:.3f}°, Lon: {lon:.3f}°\n"
            f"SL @ {selected_freq} Hz: {sl_at_freq:.1f} dB re 1 µPa m\n"
            f"Overall: {overall:.1f} dB re 1 µPa m"
        )
        annotation.xy = (lon, lat)
        annotation.set_text(text)
        annotation.set_visible(True)
        fig.canvas.draw_idle()

        if spectrum_fig is not None:
            plt.close(spectrum_fig)

        spectrum_vals = [ship.get(f"F_{f}Hz") for f in frequencies]
        spectrum_fig, ax2 = plt.subplots(figsize=(9, 4))
        ax2.semilogx(frequencies, spectrum_vals, '-o', linewidth=1.5, color='blue')
        ax2.grid(True, which="both", linestyle='--', alpha=0.6)
        ax2.set_xlabel("Frequency (Hz)", fontweight='bold')
        ax2.set_ylabel("Source Level (dB re 1 µPa m)", fontweight='bold')
        ax2.set_xlim(10, 10000)
        ax2.set_title(
            f"Source Spectrum for \"{name}\" ({vessel_type})\n"
            f"Speed: {speed} kn | Length: {length:.1f} m | Overall: {overall:.1f} dB re 1 µPa m",
            fontweight='bold', fontsize=10
        )
        ax2.tick_params(labelsize=11)
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontweight('bold')

        icon_path = next((p for p in (os.path.join("icons", f"{vessel_type}{ext}") for ext in (".png", ".jpg")) if os.path.exists(p)), os.path.join("icons", "default.png"))
        if os.path.exists(icon_path):
            try:
                inset_ax = spectrum_fig.add_axes([0.7, 0.7, 0.18, 0.18])
                img = mpimg.imread(icon_path)
                inset_ax.imshow(img)
                inset_ax.axis("off")
                ax2.annotate(f"Speed: {speed:.1f} kn\nLength: {length:.1f} m",
                             xy=(0.7, 0.67), xycoords='axes fraction',
                             fontsize=7, fontweight='bold',
                             bbox=dict(facecolor='white', edgecolor='none'))
            except Exception as e:
                print(f"Could not load icon: {e}")

        plt.tight_layout()
        plt.show()

    # === SEA CLICK (info only) ===
    else:
        # Show location info with annotation and simple line (optional)
        if previous_line is not None:
            try:
                previous_line.remove()
            except ValueError:
                previous_line = None  # line already removed
    
        lon_idx = np.abs(lon_bathy - x_click).argmin()
        lat_idx = np.abs(lat_bathy - y_click).argmin()
        depth_val = depth[lat_idx, lon_idx]
    
        lat_val = lat_bathy[lat_idx]
        lon_val = lon_bathy[lon_idx]
    
        if depth_val < 0:
            msg = f"Ocean Point\nLat: {lat_val:.3f}°, Lon: {lon_val:.3f}°\nDepth: {depth_val:.1f} m"
        else:
            msg = f"Land Point\nLat: {lat_val:.3f}°, Lon: {lon_val:.3f}°"
    
        annotation.xy = (x_click, y_click)
        annotation.set_text(msg)
        annotation.set_visible(True)
        fig.canvas.draw_idle()
        
    
import glob  
   
# === Safe file delete with retry ===
def safe_delete(file_path, retries=3, delay=0.5):
    for i in range(retries):
        try:
            os.remove(file_path)
            print(f"[Delete] Removed file: {file_path}")
            return
        except PermissionError:
            print(f"[Retry] File in use: {file_path}, retrying ({i+1}/{retries})...")

        except Exception as e:
            print(f"[Error] Could not delete file: {file_path} — {e}")
            return

# === Clear Fig and Annotations ===    
def on_clear_map(event):
    global tl_1d_fig, spl_fig, spectrum_fig, bathymetry_fig, ax
    global previous_lines, bath_fig, tl_2d_fig
    global distance_labels, distance_texts, annotation
    global distance_lines_labels  # <-- Add this line

    # === Close all figures except main map ===
    main_fig = ax.figure
    for f in plt.get_fignums():
        fig = plt.figure(f)
        if fig != main_fig:
            plt.close(fig)

    # === Clear all distance lines and labels ===
    if 'distance_lines_labels' in globals():
        for item in distance_lines_labels:
            try:
                item.remove()
            except Exception:
                pass
        distance_lines_labels.clear()


    # === Clear distance labels on map ===
    if 'distance_labels' in globals():
        for label in distance_labels:
            try:
                label.remove()
            except Exception:
                pass
        distance_labels.clear()

    # === Clear any other text annotations ===
    if 'distance_texts' in globals():
        for txt in distance_texts:
            try:
                txt.remove()
            except Exception:
                pass
        distance_texts.clear()

    # === Clear any dynamically drawn distance lines & labels ===
    if 'distance_lines_labels' in globals():
        for item in distance_lines_labels:
            try:
                item.remove()
            except Exception:
                pass
        distance_lines_labels.clear()

    if 'annotation' in globals() and annotation:
        annotation.set_visible(False)

    ax.figure.canvas.draw_idle()

    # === Delete RAM input bathymetry files, Excel, convergence PNG and TL 2D plots ===
    for file in glob.glob("BathyProfile_*_Input_For_RAM.npz"):
        safe_delete(file)
    
    for file in glob.glob("BathyProfile_*_Input_For_RAM.png"):
        safe_delete(file)
    
    for file in glob.glob("TL2D_*Hz.png"):
        safe_delete(file)

    for file in glob.glob("SPL_Parameter_Study.xlsx"):
        safe_delete(file) 

    for file in glob.glob("SPL_Convergence_*Hz.png"):
        safe_delete(file)

    print("[Clear] All figures, map elements, and RAM input files removed.")


def on_solve_pe_model(event=None, headless=False):
    global selected_freq, last_clicked_ship, spl_fig, tl_2d_fig, x_click, y_click
    
    import glob
    import numpy as np

    import os
# === Find all input profiles ===
    profile_files = sorted(glob.glob("BathyProfile_*_Input_For_RAM.npz"))
    if not profile_files:
        print("[Error] No bathymetry profiles found.")
        return

    print(f"[Info] Found {len(profile_files)} bathymetry profiles.")

    # === Prepare summary for all runs ===
    summary_data = []
    valid_results = []  # Store (ship_name, zr, TL, SPL) across all files
    for file in profile_files:
    
        try:
            print(f"\n[Running PE Model] Using profile: {file}")
            safe_name = os.path.basename(file).replace("BathyProfile_", "").replace("_Input_For_RAM.npz", "")
    
            # === Load Bathymetry ===
            bathy_data = np.load(file)
            rbzb = bathy_data["rbzb"]
            rbzb[:, 1] = np.abs(rbzb[:, 1])  # ensure positive depth
            # now close it
            bathy_data.close()
            
            # === Interpolate CW and z_ss ===
            z_min, z_max = 0, int(np.max(rbzb[:, 1])) + 100
            z_ss = np.linspace(z_min, z_max, num=100)
            rp_ss = np.array([0, int(np.ceil(np.max(rbzb[:, 0]) / 1000) * 1000)])
            cw_profile = 1480 + (z_ss - z_min) * 0.05
            cw = np.vstack([cw_profile, cw_profile + 10]).T
            
            # === Get SL and Draft for the ship used to create this profile ===
            ship_name_from_file = safe_name
            match_row = df[df["Name"].str.replace(" ", "_") == ship_name_from_file]
            
            if not match_row.empty:
                sl_at_freq = match_row.iloc[0][f"F_{selected_freq}Hz"]
                draft_m = float(match_row.iloc[0]["Draft (m)"])  # Get Draft for this ship
            else:
                print(f"[Warning] Could not find ship info for {ship_name_from_file}. Skipping.")
                continue

            # === Prepare Input Dictionary ===
            inputs = dict(
                freq=selected_freq,
                zs=draft_m,   # Source depth = Draft of the ship
                zr=3,
                z_ss=z_ss,
                rp_ss=rp_ss,
                cw=cw,
                z_sb=np.array([0]),
                rp_sb=np.array([0]),
                cb=np.array([[1700]]),
                rhob=np.array([[1.6]]),
                attn=np.array([[0.8]]),
                rmax=int(np.ceil(np.max(rbzb[:, 0]) / 1000) * 1000),
                dr=20,
                dz=5,
                zmplt=int(np.ceil((np.max(rbzb[:, 1]) + 100) / 100) * 100),
                c0=1500,
                rbzb=rbzb
            )

            # === Run RAM Simulation ===
               
            
            peram = PeRAM(**inputs)
            peram.run()

            if not hasattr(peram, 'tlg'):
                raise AttributeError("No 'tlg' found in PeRAM output.")

            # === Save TL Grid ===
            np.savetxt("tl_output.line", peram.tlg, fmt='%.2f')
            print("Simulation complete. TL saved to 'tl_output.line'")

            # === Extract Output ===
            tlg = peram.tlg
            tll = peram.tll
            cpl = peram.cpl
            phase_1d = np.angle(cpl, deg=True)
            
            # === Depth/Range arrays ===
            n_depths, n_ranges = tlg.shape
            ranges = np.arange(0, inputs['rmax'] + inputs['dr'], inputs['dr'])[:n_ranges]
            depths = np.arange(0, inputs['zmplt'] + inputs['dz'], inputs['dz'])[:n_depths]
            ranges_km = ranges / 1000

            # === 2D TL Plot ===
            tl_2d_fig, tl_ax = plt.subplots(figsize=(12, 5))
            
            # Define extent: [x_min, x_max, y_min, y_max]
            extent = [ranges_km[0], ranges_km[-1], depths[-1], depths[0]]
            
            im = tl_ax.imshow(
                tlg, 
                aspect='auto', 
                extent=extent, 
                cmap='jet', 
                origin='upper', 
                vmin=0, vmax=200
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=tl_ax)
            cbar.set_label('Transmission Loss (dB)')
            
            # Labels and title
            tl_ax.set_title(
                f"2D TL Field from {safe_name} → Clicked Location at {inputs['freq']} Hz using PE RAM",
                fontsize=12, fontweight="bold"
            )
            tl_ax.set_xlabel("Range (km)")
            tl_ax.set_ylabel("Depth (m)")
            tl_ax.grid(True)
            plt.tight_layout()
            
            # Save figure as PNG
            tl_png_filename = f"TL2D_{safe_name}_{inputs['freq']}Hz.png"
            tl_2d_fig.savefig(tl_png_filename, dpi=200)
            plt.close(tl_2d_fig)
            
            print(f"Saved 2D TL plot: '{tl_png_filename}'")


            # === Save 1D TL and Phase ===
            np.savetxt("TL_Phase_1D.txt", np.column_stack((ranges_km, tll, phase_1d)), header="Range_km\tTL_dB\tPhase_deg", fmt="%.4f", delimiter="\t")

            # === Extract TL at zr = uderdefined array and Xmax only ===
            receiver_depths = [3, 10, 50, 100, 500, 1000, 2000]
            xmax_index = -1
            #range_xmax_km = ranges_km[xmax_index]
            ship_results = []
            
            # === Get SL for the ship used to create this profile ===
            #ship_name_from_file = safe_name
            #match_row = df[df["Name"].str.replace(" ", "_") == ship_name_from_file]
            
            if not match_row.empty:
                sl_at_freq = match_row.iloc[0][f"F_{selected_freq}Hz"]
            else:
                print(f"[Warning] Could not find ship info for {ship_name_from_file}. Skipping.")
                continue
            
        
            for zr in receiver_depths:
                if zr > np.max(depths):
                    print(f"[Info] Skipping zr = {zr} m — deeper than bathymetry.")
                    continue
            
                zr_index = np.argmin(np.abs(depths - zr))
                tl_at_xmax_zr = tlg[zr_index, xmax_index]
                spl_final = sl_at_freq - tl_at_xmax_zr
                ship_results.append((zr, tl_at_xmax_zr, spl_final))

            # Store all results
            for zr, tl, spl in ship_results:
                valid_results.append((ship_name_from_file, zr, tl, spl))
        
            # === Save per-ship results ===
            #np.savez(f"SPL_TL_Result_{safe_name}.npz", receiver_depths=[r[0] for r in ship_results], tl_at_xmax=[r[1] for r in ship_results], spl=[r[2] for r in ship_results], range_km=ranges_km[xmax_index], ship=safe_name, freq=selected_freq)


            summary_data.append((safe_name, ship_results))

            print(f"[Done] PE Model + SPL for: {safe_name}")

        except Exception as e:
            print(f"[Error] Failed for {file}: {e}")

    # === Summary Plot or Printout ===
    print("\n=== Summary of SPL Results ===")
    for ship_name, result_list in summary_data:
        print(f"\n[Ship: {ship_name}]")
        for zr, tl, spl in result_list:
            print(f"  Depth {zr:>4} m → TL: {tl:5.2f} dB, SPL: {spl:5.2f} dB re 1 µPa")
    
    
    from collections import defaultdict
    
    # === Aggregate SPL by depth ===
    spl_by_depth = defaultdict(list)
    
    for (_, zr, _, spl) in valid_results:
        if not np.isnan(spl):
            spl_by_depth[zr].append(spl)
    
    # === Energetic sum at each depth ===
    final_spl_by_depth = {}
    for zr, spl_list in spl_by_depth.items():
        linear_sum = np.sum([10**(spl / 10) for spl in spl_list])
        total_spl = 10 * np.log10(linear_sum)
        final_spl_by_depth[zr] = total_spl  
 
    display_spl_and_sel_table(final_spl_by_depth, x_click, y_click, selected_freq)

def display_spl_and_sel_table(spl_by_depth, x_loc, y_loc, freq_hz):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # === Mapping depth to marine mammals ===
    depth_to_mammal = {
        10: "Otter",
        50: "Manatee",
        100: "Dolphin",
        1000: "Blue",
        2000: "Sperm"
    }

    receiver_depths = [3, 10, 50, 100, 500, 1000, 2000]
    rows = []

    for depth in receiver_depths:
        spl = spl_by_depth.get(depth, None)
        if spl is None or spl < 0:
            continue  # Skip missing or invalid data

        sel_1d = spl + 10 * np.log10(86400)
        sel_1m = spl + 10 * np.log10(86400 * 30)
        mammal = depth_to_mammal.get(depth, None)
        rows.append((depth, spl, sel_1d, sel_1m, mammal))

    if not rows:
        print("[Info] No valid SPL values to display.")
        return

    # === Setup figure ===
    fig, ax = plt.subplots(figsize=(13, 1 + 0.6 * len(rows)))
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, len(rows) + 0.5)
    ax.axis('off')

    # === Check overall Safe/Unsafe condition ===
    unsafe_flag = any(sel1d > 168 or sel1m > 168 for _, _, sel1d, sel1m, _ in rows)

    # === Header with safety label ===
    headers = ["   Depth (m)", "SPL dB re 1 μPa", "SEL-1Day dB re 1 μPa²·s", "SEL-1Month dB re 1 μPa²·s"]
    for col, text in enumerate(headers):
        ax.text(col + 0.05, len(rows) + 0.2, text,
                fontsize=12, fontweight='bold', va='bottom', ha='left', color='black')

    # Add safety heading
    safety_text = "Unsafe" if unsafe_flag else "Safe"
    safety_color = "red" if unsafe_flag else "green"
    ax.text(len(headers) + 0.5, len(rows) + 0.2, safety_text,
            fontsize=12, fontweight='bold', color=safety_color, va='bottom', ha='left')

    # === Plot each row ===
    for i, (depth, spl, sel1d, sel1m, mammal) in enumerate(rows):
        y = len(rows) - i - 1

        col_texts = [
            str(depth),
            f"{spl:.1f}",
            f"{sel1d:.1f}",
            f"{sel1m:.1f}"
        ]

        # Display each cell value
        for j, text in enumerate(col_texts):
            color = 'red' if j >= 2 and float(text) > 168 else 'black'
            ax.text(j + 0.5, y + 0.5, text, ha='center', va='center',
                    fontsize=11, weight="bold", color=color)

        # === Add mammal image if present ===
        if mammal:
            base_path = os.path.join("mammal", mammal)
            img_path = None
            for ext in [".png", ".jpg"]:
                trial_path = base_path + ext
                if os.path.exists(trial_path):
                    img_path = trial_path
                    break

            if img_path:
                img = mpimg.imread(img_path)
                imagebox = OffsetImage(img, zoom=0.25)
                ab = AnnotationBbox(imagebox, (len(col_texts) + 0.5, y+0.5),
                                    frameon=False, box_alignment=(0.5, 0.5))
                ax.add_artist(ab)
          
                # === Only draw box/label for Blue and Sperm if unsafe ===
                # Only draw red box and "unsafe" label for Blue and Sperm if any SEL is unsafe
                
                #if unsafe_flag and mammal in ['Blue', 'Sperm']:
                
                # Determine which mammals should be flagged based on selected frequency
                warn_mammals = set()
                if unsafe_flag:
                    if selected_freq > 900:
                        warn_mammals = {"Otter", "Manatee", "Dolphin", "Blue", "Sperm"}
                    elif selected_freq > 400:
                        warn_mammals = {"Manatee", "Dolphin", "Blue", "Sperm"}
                    else:
                        warn_mammals = {"Blue", "Sperm"}
                
                    if mammal in warn_mammals:
                        # Red dotted box around image
                        rect = patches.Rectangle(
                            (4 + 0.2, y + 0.2), 0.4, 0.4,
                            linewidth=1.5,
                            edgecolor='red',
                            linestyle='dotted',
                            facecolor='none',
                            transform=ax.transData,
                            zorder=5
                        )
                        ax.add_patch(rect)
                
                        # "unsafe" label inside the image box
                        ax.text(4.4, y + 0.4, "unsafe",
                                color="red", fontsize=7,
                                ha='center', va='center',
                                fontweight='bold', zorder=6)
                
                        

    # === Title ===
    ax.set_title(
        f"$\\bf{{Sound\\ Pressure\\ Level\\ and\\ Sound\\ Exposure\\ Level}}$\n"
        f"Location: ({x_loc:.2f}°, {y_loc:.2f}°) | Frequency: $\\bf{{{selected_freq}}}$ Hz",
        fontsize=16, pad=20, loc='center')

    plt.tight_layout()
    plt.show()



def compute_optimal_k(x_click, y_click, selected_freq, epsilon=1e-6):
    """
    Compute SPL at click location for top ships and determine optimal_k.
    Start k from 3, stop when ΔSPL < 0.5 dB, and save detailed Excel with per-k sheets.
    """
    import pandas as pd
    from geopy.distance import geodesic

    freq_col = f"F_{selected_freq}Hz"

    # Compute distances to clicked point (km)
    distances_to_click = np.hypot(ship_lons - x_click, ship_lats - y_click)
    distances_to_click_km = distances_to_click * 111

    # Filter ships with valid dB
    valid_mask = ~df[freq_col].isna()
    valid_indices = np.where(valid_mask)[0]
    db_values = df.iloc[valid_indices][freq_col].values
    scores = db_values / np.log10(distances_to_click_km[valid_indices] + epsilon)

    # Sort ships by score descending
    sorted_positions = np.argsort(-scores)
    top_indices = valid_indices[sorted_positions]  # consider all valid ships

    spl_values = []
    optimal_k = None

    # Create Excel writer
    writer = pd.ExcelWriter("SPL_Parameter_Study.xlsx", engine='xlsxwriter')

    # Start k from 3
    for k in range(3, len(top_indices)+1):
        idxs = top_indices[:k]
        ships_included = df.iloc[idxs]["Name"].values
        Li = df.iloc[idxs][freq_col].values
        Ri_m = np.array([geodesic((y_click, x_click), (ship_lats[i], ship_lons[i])).m for i in idxs])

        # Compute linear SPL contributions
        contributions = 10**(0.1*(Li - 20*np.log10(Ri_m)))
        contributions_dB = Li - 20*np.log10(Ri_m)
        SPL_click = 10*np.log10(np.sum(contributions))
        spl_values.append(SPL_click)

        # Save per-k details in Excel sheet
        df_sheet = pd.DataFrame({
            "Ship Name": ships_included,
            "Distance_m": Ri_m,
            f"dB_at_{selected_freq}Hz": Li,
            "Contribution_dB": contributions_dB
        })
        sheet_name = f"k={k}"
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

        # Stop if SPL stabilizes (difference < 0.2 dB)
        if k > 3 and abs(spl_values[-1] - spl_values[-2]) < 0.2:
            optimal_k = k
            break

    # If no convergence, take last k
    if optimal_k is None:
        optimal_k = len(spl_values) + 2  # +2 because we started at k=3
    # Ensure optimal_k is an integer
    optimal_k = int(optimal_k)
    
    print(f"Optimal number of ships (k) where SPL stabilizes: {optimal_k}")
    print(f"SPL at optimal_k: {spl_values[optimal_k-3]:.2f} dB")  # index offset

    # Save overall SPL convergence curve in a separate sheet
    df_curve = pd.DataFrame({
        "k": np.arange(3, 3+len(spl_values)),
        "SPL_dB": spl_values
    })
    df_curve.to_excel(writer, sheet_name="SPL_Convergence", index=False)
    
    writer.close()
    print("Saved SPL_Parameter_Study.xlsx.")
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df_curve["k"], df_curve["SPL_dB"], marker='o', color='blue', linewidth=2)
    ax.set_xlabel("k (number of top ships)", fontweight='bold')
    ax.set_ylabel(f"SPL at {selected_freq} Hz (dB)", fontweight='bold')
    ax.set_title("SPL Convergence Curve", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate optimal_k
    ax.axvline(optimal_k, color='red', linestyle='--', linewidth=1.5)
    ax.text(optimal_k, max(spl_values), f"Optimal k = {optimal_k}", color='red',
            ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    png_filename = f"SPL_Convergence_{selected_freq}Hz.png"
    fig.savefig(png_filename, dpi=200)
    plt.close(fig)
    print(f"Saved SPL convergence curve as '{png_filename}'")
    
    return optimal_k




def generate_bathymetry_profiles_from_nearest_ships(x_click, y_click, selected_freq, optimal_k):
    """
    Generate bathymetry profiles from clicked location to top 'optimal_k' ships,
    plotting lines on the map with distance labels.

    Parameters
    ----------
    x_click, y_click : float
        Longitude and latitude of clicked location
    selected_freq : int
        Frequency in Hz
    ax : matplotlib.axes.Axes
        Axis to plot distance lines
    optimal_k : int
        Number of top ships to generate bathymetry profiles for
    """
    freq_col = f"F_{selected_freq}Hz"

    if freq_col not in df.columns:
        print(f"[Error] Frequency column '{freq_col}' not found.")
        return


    # Compute approximate distances to clicked point
    distances_to_click = np.hypot(ship_lons - x_click, ship_lats - y_click)  # degrees
    distances_to_click_km = distances_to_click * 111  # rough conversion deg → km

    # Filter ships with valid dB
    valid_mask = ~df[freq_col].isna()
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print("[Error] No ships with valid dB at this frequency.")
        return

    # Compute dB / log10(distance) score (add epsilon to avoid log(0))
    epsilon = 1e-6
    db_values = df.iloc[valid_indices][freq_col].values
    scores = db_values / np.log10(distances_to_click_km[valid_indices] + epsilon)

    # Pick top k ships by score
    top_positions = np.argsort(-scores)[:int(optimal_k)]  # ensure integer
    top_indices = valid_indices[top_positions]


    # Save last clicked ship (highest score)
    global last_clicked_ship
    last_clicked_ship = df.iloc[top_indices[0]]

    # Setup bathymetry interpolator
    interp_func = RegularGridInterpolator(
        (lat_bathy, lon_bathy), depth, bounds_error=False, fill_value=np.nan
    )

    # Clear previous distance lines
    global distance_lines_labels
    try:
        for item in distance_lines_labels:
            try:
                item.remove()
            except Exception:
                pass
    except NameError:
        distance_lines_labels = []
    distance_lines_labels.clear()

    # Generate profiles
    for idx, ship_idx in enumerate(top_indices[:int(optimal_k)]):
        lon_ship = ship_lons[ship_idx]
        lat_ship = ship_lats[ship_idx]
        ship_name = df.iloc[ship_idx]["Name"]
        ship_dB = df.iloc[ship_idx][freq_col]
        distance_km = distances_to_click_km[ship_idx]
        score = ship_dB / np.log10(distance_km + epsilon)
        print(f"\n[Ship {idx+1}] {ship_name} at ({lat_ship:.3f}, {lon_ship:.3f}) | "
              f"dB={ship_dB:.1f}, distance={distance_km:.1f} km, score={score:.3f}")

        # Interpolate bathymetry profile
        num_points = 200
        lon_line = np.linspace(lon_ship, x_click, num_points)
        lat_line = np.linspace(lat_ship, y_click, num_points)
        line_coords = np.column_stack((lat_line, lon_line))
        depth_profile = interp_func(line_coords)

        # Skip invalid profiles
        if np.any(depth_profile >= 0) or np.all(np.isnan(depth_profile)):
            print("Skipped: Path crosses land or no bathymetry data.")
            continue

        # Compute geodesic distances along path
        distances_profile_km = [0]
        for i in range(1, num_points):
            p1 = (lat_line[i - 1], lon_line[i - 1])
            p2 = (lat_line[i], lon_line[i])
            distances_profile_km.append(distances_profile_km[-1] + geodesic(p1, p2).km)
        distances_profile_km = np.array(distances_profile_km)

        # Save .npz profile
        rbzb_array = np.column_stack((distances_profile_km * 1000, depth_profile))  # km → m
        safe_name = ship_name.replace(" ", "_").replace("/", "_")
        npz_filename = f"BathyProfile_{safe_name}_Input_For_RAM.npz"
        np.savez(npz_filename, rbzb=rbzb_array)
        print(f" Saved: '{npz_filename}'")

        # Save profile figure as PNG without showing
        fig2, ax3 = plt.subplots(figsize=(10, 4))
        color = 'blue'
        ax3.plot(distances_profile_km, depth_profile, color=color, linewidth=2)
        slope = np.abs(np.gradient(depth_profile, distances_profile_km))
        norm = Normalize(vmin=slope.min(), vmax=slope.max())
        slope_colors = plt.cm.plasma(norm(slope))

        for i in range(len(distances_profile_km) - 1):
            ax3.fill_between(
                [distances_profile_km[i], distances_profile_km[i + 1]],
                [depth_profile[i], depth_profile[i + 1]],
                y2=0,
                color=slope_colors[i],
                edgecolor=None
            )

        sm = ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3, orientation='vertical', pad=0.02)
        cbar.set_label("Seabed Slope (|ΔDepth/ΔDistance|)", fontsize=10, fontweight='bold')

        ax3.set_xlabel("Distance from Ship (km)", fontweight='bold')
        ax3.set_ylabel("Depth (m)", fontweight='bold')
        ax3.set_title(f"Bathymetry Profile: {ship_name} → Clicked Location", fontweight='bold', color=color)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.set_ylim(np.nanmin(depth_profile), 0)
        ax3.tick_params(labelsize=12)
        for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
            label.set_fontweight('bold')
        plt.tight_layout()
        png_filename = f"BathyProfile_{safe_name}_Input_For_RAM.png"
        fig2.savefig(png_filename, dpi=200)
        plt.close(fig2)
        print(f" Saved profile image: '{png_filename}'")

        # Plot line and annotate distance
        # --- Line thickness based on rank (1=thickest, 5=thinnest) ---
        #max_rank = 5
        # Line thickness based on rank
        linewidth = max(0.5, 3 - (idx * 0.4))  # minimum 0.5
        
        # Line color with valid alpha
        alpha_val = max(0.1, 0.8 - idx * 0.1)
        line_color = (0, 0, 0, alpha_val)
        
        line_plot = ax.plot([x_click, lon_ship], [y_click, lat_ship],
                    linestyle='--', color=line_color, linewidth=linewidth, zorder=5)[0]
        #line_plot = ax.plot([x_click, lon_ship], [y_click, lat_ship], 'k--', linewidth=1)[0]
        mid_x = (x_click + lon_ship) / 2
        mid_y = (y_click + lat_ship) / 2
        label = ax.text(mid_x, mid_y, f"{distance_km:.1f} km", fontsize=8, color='red',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='red'))
        distance_lines_labels.extend([line_plot, label])

    

def on_calculate_spl_sel(event):
    global cid_click_handler
    print("[Info] Click anywhere on the map to calculate SPL & SEL.")

    # Enable map click for SPL/SEL calculation
    cid_click_handler = fig.canvas.mpl_connect('button_press_event', on_map_click_for_spl)


def on_map_click_for_spl(event):
    global previous_line, x_click, y_click, cid_click_handler, calc_active, solving_text, selected_freq

    if not calc_active:
        return

    print("[Info] SPL & SEL calculation mode active.")
    x_click, y_click = event.xdata, event.ydata
    if x_click is not None and y_click is not None:
        print(f"[Info] Location selected at Lon={x_click:.2f}, Lat={y_click:.2f}. Generating profile and solving...")

        # Disconnect further clicks to avoid re-entry
        fig.canvas.mpl_disconnect(cid_click_handler)
        
        # === Show location info annotation === 🔧
        if previous_line is not None:
            try:
                previous_line.remove()
            except ValueError:
                previous_line = None
        
        lon_idx = np.abs(lon_bathy - x_click).argmin()
        lat_idx = np.abs(lat_bathy - y_click).argmin()
        depth_val = depth[lat_idx, lon_idx]
        
        lat_val = lat_bathy[lat_idx]
        lon_val = lon_bathy[lon_idx]
        
        if depth_val < 0:
            msg = f"Ocean Point\nLon: {lon_val:.2f}°, Lat: {lat_val:.2f}°\nDepth: {depth_val:.1f} m"
        else:
            msg = f"Land Point\nLat: {lat_val:.2f}°, Lon: {lon_val:.2f}°"
        
        annotation.xy = (x_click, y_click)
        annotation.set_text(msg)
        annotation.set_visible(True)
        fig.canvas.draw_idle()
        plt.pause(0.1)


        # === Show "Solving..." message on map ===
        solving_text = ax.annotate(
            "PE Model is being Solved, Pls wait... for Results",
            xy=(x_click, y_click),
            xycoords='data',
            xytext=(0, -20),             # ↓ shift by 20 points (~visible offset)
            textcoords='offset points',  # interpret xytext as point offset
            ha='center',
            va='top',
            fontsize=12,
            color='red',
            bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.8)
        )
        fig.canvas.draw_idle()
        plt.pause(0.1)  # Let GUI update

        try:
            # === Step 1: Generate bathymetry profile ===
            optimal_k = compute_optimal_k(x_click, y_click, selected_freq)  # SPL study already done
            generate_bathymetry_profiles_from_nearest_ships(x_click, y_click, selected_freq, optimal_k)

            # === Step 2: Check profile files ===
            profile_files = sorted(glob.glob("BathyProfile_*_Input_For_RAM.npz"))
            if not profile_files:
                print("[Error] No bathymetry profiles found.")
                solving_text.remove()
                fig.canvas.draw_idle()
                return

            # === Step 3: Run PE model ===
            on_solve_pe_model(None)

        except Exception as e:
            print(f"[Error] {e}")
        
        # === Remove "Solving..." annotation ===
        solving_text.remove()
        fig.canvas.draw_idle()

        # === Step 4: Deactivate button ===
        calc_active = False
        if cid_click_handler is not None:
            fig.canvas.mpl_disconnect(cid_click_handler)
            cid_click_handler = None
            


        calc_button.label.set_text("SPL & SEL Cal. = Off")
        calc_button.color = 'lightgray'
        calc_button.hovercolor = 'orange'
        fig.canvas.draw_idle()
        print("[Info] SPL & SEL calculation mode deactivated.")

    else:
        print("[Warning] Invalid click location.")


def on_map_click_for_spl_manual(lon_val, lat_val):
    """
    Perform SPL & SEL calculation workflow using manually entered coordinates.
    
    Parameters:
        lon_val (float): Longitude in degrees
        lat_val (float): Latitude in degrees
    """
    global previous_line, x_click, y_click, calc_active, solving_text, distance_lines_labels, selected_freq

    x_click, y_click = lon_val, lat_val
    print(f"[Info] Using manual coordinates: Lon={x_click:.2f}, Lat={y_click:.2f}")

    # === Show location info annotation ===
    if previous_line is not None:
        try:
            previous_line.remove()
        except Exception:
            previous_line = None

    lon_idx = np.abs(lon_bathy - x_click).argmin()
    lat_idx = np.abs(lat_bathy - y_click).argmin()
    depth_val = depth[lat_idx, lon_idx]

    if depth_val < 0:
        msg = f"Ocean Point\nLon: {x_click:.2f}°, Lat: {y_click:.2f}°\nDepth: {depth_val:.1f} m"
    else:
        msg = f"Land Point\nLon: {x_click:.2f}°, Lat: {y_click:.2f}°"

    annotation.xy = (x_click, y_click)
    annotation.set_text(msg)
    annotation.set_visible(True)
    fig.canvas.draw_idle()
    plt.pause(0.1)

    # === Show "Solving..." message on map ===
    solving_text = ax.annotate(
        "PE Model is being Solved, Pls wait... for Results",
        xy=(x_click, y_click),
        xycoords='data',
        xytext=(0, -20),             # ↓ shift by 20 points (~visible offset)
        textcoords='offset points',  # interpret xytext as point offset
        ha='center',
        va='top',
        fontsize=12,
        color='red',
        bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.8)
    )

    fig.canvas.draw_idle()
    plt.pause(0.1)  # Let GUI update

    try:
        
        # === Step 1: Generate bathymetry profile ===
        optimal_k = compute_optimal_k(x_click, y_click, selected_freq)  # SPL study already done
        generate_bathymetry_profiles_from_nearest_ships(x_click, y_click, selected_freq, optimal_k)
        
        # === Step 2: Check profile files ===
        profile_files = sorted(glob.glob("BathyProfile_*_Input_For_RAM.npz"))
        if not profile_files:
            print("[Error] No bathymetry profiles found.")
            solving_text.remove()
            fig.canvas.draw_idle()
            return

        # === Step 3: Run PE model ===
        on_solve_pe_model(None)

    except Exception as e:
        print(f"[Error] {e}")

    # === Remove "Solving..." annotation ===
    solving_text.remove()
    fig.canvas.draw_idle()


    # === Step 5: Deactivate calculation mode & update button ===
    calc_active = False
    calc_button.label.set_text("SPL & SEL Cal. = Off")
    calc_button.color = 'lightgray'
    calc_button.hovercolor = 'orange'
    fig.canvas.draw_idle()

    print("[Info] SPL & SEL calculation mode deactivated.")



# Define global flag
calc_active = False
cid_click_handler = None

# Create button with longer label space
calc_ax = plt.axes([0.82, 0.015, 0.12, 0.05])  # Wider size for longer label 
calc_button = Button(calc_ax, "SPL & SEL Cal. = Off", color='lightgray', hovercolor='orange')

import tkinter as tk
from tkinter import simpledialog

def toggle_calc(event):
    """
    Toggle SPL & SEL calculation mode.
    Offers user choice between clicking on map or entering coordinates manually.
    """
    global calc_active, cid_click_handler, x_click, y_click

    calc_active = not calc_active  # Toggle flag

    if calc_active:
        # Ask user which mode to use via GUI dialog
        root = None
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main Tk window

            choice = simpledialog.askstring(
                "SPL & SEL Calculation",
                "Select mode:\n1 = Click on Map\n2 = Enter Coordinates manually"
            )

        finally:
            if root:
                root.destroy()

        if choice is None:
            # User cancelled
            calc_active = False
            print("[Info] SPL & SEL calculation cancelled by user.")
            return

        choice = choice.strip()

        if choice == "1":
            # Map click mode
            calc_button.label.set_text("SPL & SEL Cal. = On")
            calc_button.color = 'lightgreen'
            calc_button.hovercolor = 'red'
            print("[Info] Click anywhere on the map to calculate SPL & SEL.")

            # Connect the click handler
            if cid_click_handler is None:
                cid_click_handler = fig.canvas.mpl_connect("button_press_event", on_map_click_for_spl)

        elif choice == "2":
            # Manual coordinates mode
            root = None
            try:
                root = tk.Tk()
                root.withdraw()
                lon_val = simpledialog.askfloat("Manual Input", "Enter Longitude (deg):")
                lat_val = simpledialog.askfloat("Manual Input", "Enter Latitude (deg):")

                if lon_val is None or lat_val is None:
                    print("[Info] Manual input cancelled.")
                    calc_active = False
                    return

                x_click, y_click = lon_val, lat_val
                print(f"[Info] Coordinates entered: Lon={x_click:.2f}, Lat={y_click:.2f}. Generating profile and solving...")

                # Run the same workflow as a map click
                on_map_click_for_spl_manual(x_click, y_click)

            except Exception as e:
                print(f"[Error] {e}")
                calc_active = False
                calc_button.label.set_text("SPL & SEL Cal. = Off")
                calc_button.color = 'lightgray'
                calc_button.hovercolor = 'orange'
                fig.canvas.draw_idle()

            finally:
                if root:
                    root.destroy()

        else:
            print("[Warning] Invalid choice. Mode not activated.")
            calc_active = False

    else:
        # Deactivate mode
        calc_button.label.set_text("SPL & SEL Cal. = Off")
        calc_button.color = 'lightgray'
        calc_button.hovercolor = 'orange'
        print("[Info] SPL & SEL calculation mode deactivated.")

        # Disconnect click handler if active
        if cid_click_handler is not None:
            fig.canvas.mpl_disconnect(cid_click_handler)
            cid_click_handler = None

    # Refresh button
    calc_button.ax.figure.canvas.draw_idle()





# === Connect the toggle function ===
calc_button.on_clicked(toggle_calc)


# === Connect the event ===
fig.canvas.mpl_connect("button_press_event", onclick)

#plt.tight_layout()

clear_ax = plt.axes([0.68, 0.015, 0.12, 0.05])
clear_button = Button(clear_ax, "Clear Map/ Result Files", color='mistyrose', hovercolor='salmon')
clear_button.on_clicked(on_clear_map)

from matplotlib.widgets import Button


def launch_dropdown():
    """
    Popup frequency selection using a Tkinter OptionMenu (dropdown).
    Uses a temporary Tk root that auto-destroys after selection.
    """
    global selected_freq, freq_col, sc, cb2

    root = None
    try:
        root = tk.Tk()
        root.title("Select Frequency")
        root.geometry("200x110")  # small popup size
        root.resizable(False, False)

        # Store the chosen frequency
        choice_var = tk.StringVar(root)
        choice_var.set(str(available_freqs[0]))  # default selection

        # Label
        label = tk.Label(root, text="Choose Frequency (Hz):", font=("Arial", 10))
        label.pack(pady=2)

        # Dropdown (OptionMenu)
        dropdown = tk.OptionMenu(root, choice_var, *map(str, available_freqs))
        dropdown.config(width=10, font=("Arial", 8))
        dropdown.pack(pady=2)

        # OK button
        def confirm_choice():
            global selected_freq, freq_col  # <-- added for updating globally
            
            choice = choice_var.get()
            if not choice:
                print("[Info] Frequency selection cancelled.")
                root.destroy()
                return

            try:
                freq_val = int(choice.strip())
            except ValueError:
                print(f"[Warning] Invalid frequency input: {choice}")
                root.destroy()
                return

            freq_col = f"F_{freq_val}Hz"
            if freq_col not in df.columns:
                print(f"[Warning] Frequency {freq_val} Hz not in dataset.")
                root.destroy()
                return

            # Update globals
            selected_freq = freq_val

            # Update scatter colors
            sc.set_array(df[freq_col].values)

            # Update colorbar label
            cb2.set_label(f"Source Level @ {selected_freq} Hz (dB re 1 µPa m)")

            # Update title
            ax.set_title(
                f"Visualizing Source Level Shipping Noise @ {selected_freq} Hz on {date_formatted}",
                fontsize=16, fontweight="bold"
            )

            fig.canvas.draw_idle()
            print(f"[Info] Frequency updated to {selected_freq} Hz.")

            root.destroy()

        ok_btn = tk.Button(root, text="OK", command=confirm_choice, width=10, bg="lightblue")
        ok_btn.pack(pady=5)

        # Run this popup as a mini loop
        root.mainloop()

    finally:
        if root:
            try:
                root.destroy()
            except:
                pass

# === Add button to Matplotlib toolbar area ===
button_ax = plt.axes([0.01, 0.015, 0.12, 0.05])
btn = Button(button_ax, "Select Frequency", color="lightyellow", hovercolor="lightblue")

def on_button_click(event):
    # Run dropdown inline (no extra Tk mainloop blocking Matplotlib)
    launch_dropdown()

btn.on_clicked(on_button_click)



from datetime import datetime
year = datetime.now().year

# --- Add the credits text (initial size = 9) ---
credit_text = fig.text(
    0.42, 0.01,
    f"© {year} Map data: Natural Earth | Bathymetry: GEBCO_2021 Grid\n"
    "Rendered with Cartopy & Matplotlib",
    ha="center", va="bottom",
    fontsize=9, fontstyle="italic"
)

# --- Function to resize text automatically ---
def adjust_credit(event):
    # Get current figure width (in inches)
    w = fig.get_size_inches()[0]
    # Scale font size based on width (min 6, max 10)
    new_size = max(6, min(10, w * 0.5))
    credit_text.set_fontsize(new_size)
    fig.canvas.draw_idle()

# --- Connect resize event ---
fig.canvas.mpl_connect("resize_event", adjust_credit)


# === Show the map ===
plt.show()
  