import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

# Use the current project folder (where this script resides)
folder = os.path.dirname(os.path.abspath(__file__))

# Get all CSV files recursively in the current project folder
files = glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True)

print(f"Found {len(files)} CSV file(s) in '{folder}' (recursive search)")

# Read and combine all CSV files safely
dataframes = []
def read_mixed_csv(file_path: str) -> pd.DataFrame:
    """Read CSVs that contain occasional semicolon-only lines and repeated comma headers.
    Keeps only the comma-delimited header and data blocks.
    """
    header_prefix = 'Date,Time'
    cleaned_lines = []
    header_seen = False
    with open(file_path, 'r', encoding='latin1') as fh:
        for raw in fh:
            line = raw.strip('\n')
            if not line.strip():
                continue  # skip blanks
            if line.strip() == 'Date;Time':
                continue  # skip semicolon lines
            if line.startswith(header_prefix):
                if header_seen:
                    # skip repeated headers
                    continue
                cleaned_lines.append(line + '\n')
                header_seen = True
                continue
            # Keep only comma-delimited data lines after header
            if header_seen and ',' in line:
                cleaned_lines.append(line + '\n')
    COMMON_HEADER = [
        'Date','Time','Snow intensity (mm/h)','Intensity of precipitation (mm/h)',
        'Precipitation since start (mm)','Weather code SYNOP WaWa','Weather code METAR/SPECI',
        'Weather code NWS','Radar reflectivity (dBz)','MOR Visibility (m)','Signal amplitude of Laserband',
        'Number of detected particles','Temperature in sensor (�C)','Heating current (A)','Sensor voltage (V)',
        'Kinetic Energy','Spectrum'
    ]
    if not cleaned_lines:
        # Fallback: many monthly files are headerless but comma-delimited, use common header
        return pd.read_csv(
            file_path,
            encoding='latin1',
            sep=',',
            engine='python',
            on_bad_lines='skip',
            header=None,
            names=COMMON_HEADER
        )
    buffer = io.StringIO(''.join(cleaned_lines))
    return pd.read_csv(buffer, sep=',', engine='python')

for file in files:
    try:
        df = read_mixed_csv(file)
        dataframes.append(df)
        print(f"✅ Loaded: {os.path.basename(file)}")
    except Exception as e:
        print(f"❌ Skipped {os.path.basename(file)} due to error: {e}")

# Combine all data into one DataFrame
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\n✅ Combined {len(dataframes)} files successfully!\n")

    # ===== Data Cleaning =====
    cleaned_df = combined_df.copy()

    # Parse timestamp first (before numeric coercion)
    timestamp = None
    if {'Date', 'Time'}.issubset(cleaned_df.columns):
        timestamp = pd.to_datetime(
            cleaned_df['Date'].astype(str).str.strip() + ' ' + cleaned_df['Time'].astype(str).str.strip(),
            format='%d.%m.%Y %H:%M:%S', errors='coerce'
        )
    elif 'Date;Time' in cleaned_df.columns:
        dt_split = cleaned_df['Date;Time'].astype(str).str.split(';', n=1, expand=True)
        if dt_split.shape[1] == 2:
            timestamp = pd.to_datetime(
                dt_split[0].str.strip() + ' ' + dt_split[1].str.strip(),
                format='%d.%m.%Y %H:%M:%S', errors='coerce'
            )
    if timestamp is not None:
        cleaned_df['timestamp'] = timestamp
        date_min = cleaned_df['timestamp'].min()
        date_max = cleaned_df['timestamp'].max()
        if pd.isna(date_min) or pd.isna(date_max):
            date_min, date_max = None, None
    else:
        date_min, date_max = None, None

    # Identify numeric columns: exclude timestamp and textual date/time
    exclude_cols = {'Date', 'Time', 'Date;Time', 'Spectrum', 'timestamp'}
    candidate_cols = [c for c in cleaned_df.columns if c not in exclude_cols]
    # Coerce only candidate columns to numeric where appropriate
    for col in candidate_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

    # Replace sentinel values like -9.999 with NaN
    cleaned_df = cleaned_df.replace({'-9.999': pd.NA, -9.999: pd.NA})

    # Handle nulls: fill numeric NaNs with median, drop rows still all-null in numeric
    numeric_cols = cleaned_df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    else:
        print("⚠️ No numeric columns detected after coercion; stats/outliers may be limited.")

    print(f"✅ Cleaned data: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")

    # (Timestamp already parsed above)

    # ===== Statistics: mean, median, mode =====
    if numeric_cols:
        stats_mean = cleaned_df[numeric_cols].mean()
        stats_median = cleaned_df[numeric_cols].median()
        # Pandas mode can return multiple; take first mode per column if exists
        stats_mode = cleaned_df[numeric_cols].mode().iloc[0] if not cleaned_df[numeric_cols].mode().empty else pd.Series(index=numeric_cols, dtype=float)

        print("\n📊 Mean (numeric columns):")
        print(stats_mean)
        print("\n📊 Median (numeric columns):")
        print(stats_median)
        print("\n📊 Mode (numeric columns):")
        print(stats_mode)
    else:
        print("ℹ️ Skipping mean/median/mode: no numeric columns available.")

    # ===== Outlier Detection (IQR) =====
    outlier_summary = []
    if numeric_cols:
        for col in numeric_cols:
            q1 = cleaned_df[col].quantile(0.25)
            q3 = cleaned_df[col].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                outliers_count = 0
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (cleaned_df[col] < lower) | (cleaned_df[col] > upper)
                outliers_count = int(mask.sum())
            outlier_summary.append({"column": col, "outliers": outliers_count})

        outlier_df = pd.DataFrame(outlier_summary).sort_values(by="outliers", ascending=False)
        print("\n🚩 Outliers per column (IQR method):")
        print(outlier_df)
    else:
        print("ℹ️ Skipping outlier detection: no numeric columns available.")

    # ===== Plots (Matplotlib) =====
    plt.style.use('ggplot')
    plots_dir = os.path.join(folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if numeric_cols:
        # Filter to meaningful numeric columns (avoid blanks/constant)
        candidates = []
        for col in numeric_cols:
            series = cleaned_df[col]
            valid_count = int(series.count())
            unique_count = int(series.nunique(dropna=True))
            std_val = float(series.std()) if valid_count > 1 else 0.0
            if valid_count >= 50 and unique_count >= 2 and std_val > 0:
                candidates.append({'col': col, 'std': std_val, 'valid': valid_count})

        # Sort by variability and cap the number of plots
        candidates.sort(key=lambda x: x['std'], reverse=True)
        selected_cols = [c['col'] for c in candidates[:10]]
        print(f"\n🗂️ Plotting {len(selected_cols)} column(s) out of {len(numeric_cols)} numeric columns (filtered by variance and data sufficiency).")
        if not selected_cols:
            print("ℹ️ No columns met plotting criteria; skipping plot generation.")
        
        # Histograms, boxplots, and timeseries (if timestamp is available)
        def _sanitize(name: str) -> str:
            # Remove characters invalid in Windows filenames
            return ''.join(c if c not in '<>:"/\\|?*' else '_' for c in str(name)).replace(':', '_')

        for col in selected_cols:
            try:
                # Histogram
                plt.figure(figsize=(8, 4))
                cleaned_df[col].hist(bins=30)
                if date_min and date_max:
                    plt.title(f"Histogram - {col} ({date_min:%Y-%m-%d} → {date_max:%Y-%m-%d})")
                else:
                    plt.title(f"Histogram - {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                hist_path = os.path.join(plots_dir, f"{_sanitize(col)}_hist.png")
                plt.tight_layout()
                plt.savefig(hist_path)
                plt.close()

                # Boxplot
                plt.figure(figsize=(6, 4))
                cleaned_df.boxplot(column=[col])
                if date_min and date_max:
                    plt.title(f"Boxplot - {col} ({date_min:%Y-%m-%d} → {date_max:%Y-%m-%d})")
                else:
                    plt.title(f"Boxplot - {col}")
                box_path = os.path.join(plots_dir, f"{_sanitize(col)}_box.png")
                plt.tight_layout()
                plt.savefig(box_path)
                plt.close()

                # Timeseries line plot where timestamp exists
                if 'timestamp' in cleaned_df.columns:
                    ts_df = cleaned_df[['timestamp', col]].dropna().sort_values('timestamp')
                    if not ts_df.empty and ts_df[col].nunique() > 1:
                        plt.figure(figsize=(10, 4))
                        plt.plot(ts_df['timestamp'], ts_df[col], linewidth=1)
                        plt.xlabel('Date')
                        plt.ylabel(col)
                        if date_min and date_max:
                            plt.title(f"Timeseries - {col} ({date_min:%Y-%m-%d} → {date_max:%Y-%m-%d})")
                        else:
                            plt.title(f"Timeseries - {col}")
                        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
                        plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
                        plt.tight_layout()
                        ts_path = os.path.join(plots_dir, f"{_sanitize(col)}_timeseries.png")
                        plt.savefig(ts_path)
                        plt.close()
                    else:
                        print(f"ℹ️ Skipped timeseries for '{col}': insufficient variability or no timestamp data.")
            except Exception as e:
                print(f"⚠️ Plot skipped for column '{col}': {e}")

        print(f"\n🖼️ Plots saved to: {plots_dir}")
    else:
        print("ℹ️ Skipping plots: no numeric columns available.")
else:
    print("⚠️ No valid CSV files found or all failed to load.")
