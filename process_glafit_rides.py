import os
import zipfile
import glob
import pandas as pd
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_DIR = "./sensor_logs"                # Folder containing Sensor Logger .zip files or extracted ride folders
OUTPUT_EXCEL = "glafit_battery_dataset.xlsx"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in kilometers between consecutive GPS points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lon1)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def load_ride_tables(source_path):
    """Load Location.csv, Accelerometer.csv, and Metadata.csv from a .zip file or folder."""
    if source_path.lower().endswith(".zip"):
        with zipfile.ZipFile(source_path, "r") as z:
            names = z.namelist()
            loc = pd.read_csv(z.open("Location.csv")) if "Location.csv" in names else None
            acc = pd.read_csv(z.open("Accelerometer.csv")) if "Accelerometer.csv" in names else None
            meta = pd.read_csv(z.open("Metadata.csv")) if "Metadata.csv" in names else None
    else:
        loc_path = os.path.join(source_path, "Location.csv")
        acc_path = os.path.join(source_path, "Accelerometer.csv")
        meta_path = os.path.join(source_path, "Metadata.csv")
        loc = pd.read_csv(loc_path) if os.path.exists(loc_path) else None
        acc = pd.read_csv(acc_path) if os.path.exists(acc_path) else None
        meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    return loc, acc, meta

def process_ride_sensors(source_path):
    """Extract clean distance, speed, elevation, and acceleration metrics from Sensor Logger CSVs."""
    loc, acc, meta = load_ride_tables(source_path)
    if loc is None or loc.empty:
        raise ValueError(f"Missing or empty Location.csv in {source_path}")

    # Metadata extraction
    rec_time = ""
    device_name = ""
    if meta is not None and not meta.empty:
        rec_time = str(meta.loc[0, "recording time"]) if "recording time" in meta.columns else ""
        device_name = str(meta.loc[0, "device name"]) if "device name" in meta.columns else ""

    # Filter poor GPS fixes (> 25m horizontal error)
    loc_clean = loc[loc["horizontalAccuracy"] <= 25.0].copy().reset_index(drop=True)
    if len(loc_clean) < 2:
        loc_clean = loc.copy().reset_index(drop=True)

    # Distance (km) via Haversine
    step_dist = haversine_km(
        loc_clean["latitude"].shift(),
        loc_clean["longitude"].shift(),
        loc_clean["latitude"],
        loc_clean["longitude"]
    ).fillna(0.0)
    gps_distance_km = float(step_dist.sum())

    # Duration (minutes)
    duration_sec = float(loc["seconds_elapsed"].max() - loc["seconds_elapsed"].min())
    duration_min = duration_sec / 60.0

    # Speed metrics (iOS reports -1.0 when speed is invalid; filter speed >= 0)
    valid_speeds_kmh = loc_clean.loc[loc_clean["speed"] >= 0, "speed"] * 3.6
    avg_speed_all_kmh = float(valid_speeds_kmh.mean()) if not valid_speeds_kmh.empty else 0.0
    moving_speeds = valid_speeds_kmh[valid_speeds_kmh > 1.0]
    avg_moving_speed_kmh = float(moving_speeds.mean()) if not moving_speeds.empty else 0.0
    max_speed_kmh = float(valid_speeds_kmh.max()) if not valid_speeds_kmh.empty else 0.0
    idle_pct = float((1.0 - (len(moving_speeds) / max(len(valid_speeds_kmh), 1))) * 100.0)

    # Net elevation gain (m)
    alt_col = "altitudeAboveMeanSeaLevel" if "altitudeAboveMeanSeaLevel" in loc_clean.columns else "altitude"
    alt_diff = loc_clean[alt_col].diff().fillna(0.0)
    elevation_gain_m = float(alt_diff[alt_diff > 0].sum())

    # Acceleration metrics (m/s^2)
    # 1) GPS-derived longitudinal acceleration (immune to handlebar motor vibration)
    dt = loc_clean["seconds_elapsed"].diff()
    dv = (loc_clean["speed"].clip(lower=0)).diff()
    gps_accel = (dv / dt).replace([np.inf, -np.inf], np.nan).dropna()
    gps_accel = gps_accel[(gps_accel >= -8.0) & (gps_accel <= 8.0)]  # clip GPS glitches
    gps_accel_rms = float(np.sqrt(np.mean(gps_accel ** 2))) if not gps_accel.empty else 0.0
    gps_pos_accel_mean = float(gps_accel[gps_accel > 0.1].mean()) if (gps_accel > 0.1).any() else 0.0

    # 2) IMU 3-axis RMS acceleration (low-pass smoothed to reduce high-frequency handlebar buzz)
    imu_rms_ms2 = 0.0
    if acc is not None and not acc.empty:
        acc_smooth = acc[["x", "y", "z"]].rolling(window=5, min_periods=1, center=True).mean()
        acc_mag = np.sqrt(acc_smooth["x"] ** 2 + acc_smooth["y"] ** 2 + acc_smooth["z"] ** 2)
        imu_rms_ms2 = float(np.sqrt(np.mean(acc_mag ** 2)))

    return {
        "Recording_Timestamp": rec_time,
        "Device": device_name,
        "Duration_min": round(duration_min, 2),
        "GPS_Distance_km": round(gps_distance_km, 3),
        "Avg_Speed_Total_kmh": round(avg_speed_all_kmh, 2),
        "Avg_Moving_Speed_kmh": round(avg_moving_speed_kmh, 2),
        "Max_Speed_kmh": round(max_speed_kmh, 2),
        "Idle_Time_pct": round(idle_pct, 1),
        "Elevation_Gain_m": round(elevation_gain_m, 1),
        "GPS_Accel_RMS_ms2": round(gps_accel_rms, 3),
        "GPS_Pos_Accel_Mean_ms2": round(gps_pos_accel_mean, 3),
        "IMU_Accel_RMS_ms2": round(imu_rms_ms2, 3),
    }

def style_and_save_workbook(df, output_path):
    """Write DataFrame to a formatted Excel file with live Excel formulas for drain & projected range."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Battery_Test_Dataset"

    # Styling definitions
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    manual_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    calc_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    zebra_fill = PatternFill(start_color="F9FAFB", end_color="F9FAFB", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    body_font = Font(name="Calibri", size=10, color="111827")
    bold_font = Font(name="Calibri", size=10, bold=True, color="111827")
    thin_border = Border(
        left=Side(style="thin", color="D1D5DB"),
        right=Side(style="thin", color="D1D5DB"),
        top=Side(style="thin", color="D1D5DB"),
        bottom=Side(style="thin", color="D1D5DB"),
    )

    headers = list(df.columns)
    ws.append(headers)

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border
    ws.row_dimensions[1].height = 30

    manual_cols = {
        "Bike_ID", "Driver_Name", "Rider_Weight_kg", "Cargo_Weight_kg",
        "Riding_Mode", "Start_Temp_C", "End_Temp_C",
        "Start_Odo_km", "End_Odo_km", "Start_SoC_pct", "End_SoC_pct", "Notes"
    }
    formula_cols = {
        "Total_Payload_kg", "Odo_Distance_km", "SoC_Drained_pct",
        "Drain_Rate_pct_per_km_GPS", "Drain_Rate_pct_per_km_Odo", "Est_Full_Range_km_GPS"
    }

    col_map = {name: idx + 1 for idx, name in enumerate(headers)}

    for r_idx, row in enumerate(df.itertuples(index=False), start=2):
        for c_idx, val in enumerate(row, start=1):
            col_name = headers[c_idx - 1]
            cell = ws.cell(row=r_idx, column=c_idx)

            # Inject dynamic Excel formulas for calculated columns
            if col_name == "Total_Payload_kg":
                c_rw = get_column_letter(col_map["Rider_Weight_kg"])
                c_cw = get_column_letter(col_map["Cargo_Weight_kg"])
                cell.value = f"={c_rw}{r_idx}+{c_cw}{r_idx}"
            elif col_name == "Odo_Distance_km":
                c_so = get_column_letter(col_map["Start_Odo_km"])
                c_eo = get_column_letter(col_map["End_Odo_km"])
                cell.value = f"={c_eo}{r_idx}-{c_so}{r_idx}"
            elif col_name == "SoC_Drained_pct":
                c_ss = get_column_letter(col_map["Start_SoC_pct"])
                c_es = get_column_letter(col_map["End_SoC_pct"])
                cell.value = f"={c_ss}{r_idx}-{c_es}{r_idx}"
            elif col_name == "Drain_Rate_pct_per_km_GPS":
                c_sd = get_column_letter(col_map["SoC_Drained_pct"])
                c_gd = get_column_letter(col_map["GPS_Distance_km"])
                cell.value = f'=IF({c_gd}{r_idx}>0, {c_sd}{r_idx}/{c_gd}{r_idx}, "")'
            elif col_name == "Drain_Rate_pct_per_km_Odo":
                c_sd = get_column_letter(col_map["SoC_Drained_pct"])
                c_od = get_column_letter(col_map["Odo_Distance_km"])
                cell.value = f'=IF({c_od}{r_idx}>0, {c_sd}{r_idx}/{c_od}{r_idx}, "")'
            elif col_name == "Est_Full_Range_km_GPS":
                c_dr = get_column_letter(col_map["Drain_Rate_pct_per_km_GPS"])
                cell.value = f'=IF(AND(ISNUMBER({c_dr}{r_idx}), {c_dr}{r_idx}>0), 100/{c_dr}{r_idx}, "")'
            else:
                cell.value = val

            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if col_name in manual_cols:
                cell.fill = manual_fill
                cell.font = body_font
            elif col_name in formula_cols:
                cell.fill = calc_fill
                cell.font = bold_font
            else:
                if r_idx % 2 == 1:
                    cell.fill = zebra_fill
                cell.font = body_font

            # Number formats
            if "Drain_Rate" in col_name:
                cell.number_format = "0.000"
            elif "Distance" in col_name or "Range" in col_name or "Accel" in col_name:
                cell.number_format = "0.00"
            elif "Speed" in col_name or "Weight" in col_name or "Payload" in col_name:
                cell.number_format = "0.0"

    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(df) + 1}"

    for col_idx, col_name in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = max(len(col_name) + 4, 14)

    wb.save(output_path)

# ==============================================================================
# MAIN EXECUTION WORKFLOW
# ==============================================================================
def main():
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Find all .zip files or subdirectories inside INPUT_DIR
    entries = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.zip")) +
        [d for d in glob.glob(os.path.join(INPUT_DIR, "*")) if os.path.isdir(d)]
    )

    if not entries:
        print(f"No .zip files or ride folders found in '{INPUT_DIR}'. Place your Sensor Logger exports there and re-run.")
        return

    # Load existing Excel dataset if present so we don't overwrite past rides
    if os.path.exists(OUTPUT_EXCEL):
        existing_df = pd.read_excel(OUTPUT_EXCEL, sheet_name="Battery_Test_Dataset")
        processed_files = set(existing_df["Source_File"].astype(str).tolist())
    else:
        existing_df = pd.DataFrame()
        processed_files = set()

    new_rows = []
    for path in entries:
        fname = os.path.basename(path)
        if fname in processed_files:
            print(f"[SKIP] Already logged: {fname}")
            continue

        print(f"\n--- Processing Ride: {fname} ---")
        metrics = process_ride_sensors(path)
        print(f"Detected GPS Distance: {metrics['GPS_Distance_km']} km | Avg Moving Speed: {metrics['Avg_Moving_Speed_kmh']} km/h | Duration: {metrics['Duration_min']} min")

        # Prompt for manual dashboard & rider parameters
        bike_id = input("  Bike ID (e.g., 108): ").strip()
        driver = input("  Driver Name: ").strip()
        rider_wt = float(input("  Rider Weight (kg): ").strip() or 0)
        cargo_wt = float(input("  Additional Cargo Weight (kg, 0 if none): ").strip() or 0)
        mode = input("  Riding Mode (1 / 2 / 3): ").strip()
        start_temp = input("  Start Dashboard Temp (°C, optional): ").strip()
        end_temp = input("  End Dashboard Temp (°C, optional): ").strip()
        start_odo = float(input("  Start Odometer (km): ").strip())
        end_odo = float(input("  End Odometer (km): ").strip())
        start_soc = float(input("  Start Battery SoC (%): ").strip())
        end_soc = float(input("  End Battery SoC (%): ").strip())
        notes = input("  Notes (traffic/route remarks, optional): ").strip()

        row = {
            "Ride_ID": len(existing_df) + len(new_rows) + 1,
            "Source_File": fname,
            "Recording_Timestamp": metrics["Recording_Timestamp"],
            "Bike_ID": bike_id,
            "Driver_Name": driver,
            "Rider_Weight_kg": rider_wt,
            "Cargo_Weight_kg": cargo_wt,
            "Total_Payload_kg": rider_wt + cargo_wt,
            "Riding_Mode": mode,
            "Start_Temp_C": float(start_temp) if start_temp else None,
            "End_Temp_C": float(end_temp) if end_temp else None,
            "Start_Odo_km": start_odo,
            "End_Odo_km": end_odo,
            "Odo_Distance_km": end_odo - start_odo,
            "Start_SoC_pct": start_soc,
            "End_SoC_pct": end_soc,
            "SoC_Drained_pct": start_soc - end_soc,
            "GPS_Distance_km": metrics["GPS_Distance_km"],
            "Duration_min": metrics["Duration_min"],
            "Avg_Speed_Total_kmh": metrics["Avg_Speed_Total_kmh"],
            "Avg_Moving_Speed_kmh": metrics["Avg_Moving_Speed_kmh"],
            "Max_Speed_kmh": metrics["Max_Speed_kmh"],
            "Idle_Time_pct": metrics["Idle_Time_pct"],
            "Elevation_Gain_m": metrics["Elevation_Gain_m"],
            "GPS_Accel_RMS_ms2": metrics["GPS_Accel_RMS_ms2"],
            "GPS_Pos_Accel_Mean_ms2": metrics["GPS_Pos_Accel_Mean_ms2"],
            "IMU_Accel_RMS_ms2": metrics["IMU_Accel_RMS_ms2"],
            "Drain_Rate_pct_per_km_GPS": None,
            "Drain_Rate_pct_per_km_Odo": None,
            "Est_Full_Range_km_GPS": None,
            "Notes": notes,
        }
        new_rows.append(row)

    if new_rows:
        combined_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        style_and_save_workbook(combined_df, OUTPUT_EXCEL)
        print(f"\n[DONE] Saved {len(new_rows)} new ride(s) to '{OUTPUT_EXCEL}'.")
    else:
        print("\nNo new rides added.")

if __name__ == "__main__":
    main()
