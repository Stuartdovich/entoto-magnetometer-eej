import os
import pandas as pd
import numpy as np
import chaosmagpy as cp
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import importlib
import calcChaos
importlib.reload(calcChaos)
from calcChaos import chaos, chaos_ext, datetime_to_decimal_year
import re
from datetime import datetime
from tqdm import tqdm
import matplotlib.dates as mdates

def load_entoto_data(directory):
    all_data = []

    def extract_date(filename):
        match = re.search(r'ent(\d{8})pmin\.min', filename)
        if match:
            return datetime.strptime(match.group(1), '%Y%m%d')
        return datetime.min  # fallback

    # List and sort .min files by date in filename
    min_files = sorted(
        [f for f in os.listdir(directory) if f.endswith('.min')],
        key=extract_date
    )

    # Progress bar over files
    for file in tqdm(min_files, desc="Loading Entoto .min files"):
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(
                file_path,
                sep=r'\s+',
                comment='#',
                header=None,
                skiprows=16,
                names=["DATE", "TIME", "DOY", "ENTX", "ENTY", "ENTZ", "ENTF"],
                engine='python',
                on_bad_lines='skip'
            )
            df.replace(99999.0, np.nan, inplace=True)
            # Combine DATE and TIME into a single DATETIME column
            df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], errors='coerce')

            # Drop the original separate DATE and TIME columns
            df.drop(columns=['DATE', 'TIME'], inplace=True)

            all_data.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else None


# Step 2: Calculate H component from X and Y
def calculate_H_component(df):
    df['H'] = np.sqrt(df['ENTX']**2 + df['ENTY']**2)
    return df

# Step 3: Remove CHAOS internal field to get H_residual
def remove_internal_field(df, station_lat=9.108, station_lon=38.807, station_alt=2450):
    df['H_internal'] = np.nan
    df['H_residual'] = np.nan

    # Group data by date
    df['DATE'] = df['DATETIME'].dt.date
    unique_dates = df['DATE'].unique()

    for date in unique_dates:
        daily_df = df[df['DATE'] == date]
        pkl_filename = f"internal_field_{date}.pkl"

        if os.path.exists(pkl_filename):
            daily_internal = pd.read_pickle(pkl_filename)
        else:
            # Compute internal field for each timestamp
            internal_values = []
            for dt in daily_df['DATETIME']:
               
                Bx, By, Bz = chaos(datetime_to_decimal_year(dt), station_lat, station_lon, station_alt)
                H_internal = np.sqrt(Bx**2 + By**2)
                internal_values.append(H_internal)
            daily_internal = pd.Series(internal_values, index=daily_df.index)
            daily_internal.to_pickle(pkl_filename)

        df.loc[daily_df.index, 'H_internal'] = daily_internal
        df.loc[daily_df.index, 'H_residual'] = df.loc[daily_df.index, 'H'] - daily_internal

    df.drop(columns=['DATE'], inplace=True)
    return df
    
def compute_external_field(df, station_lat=9.108, station_lon=38.807, station_alt=2450):
    df['H_external'] = np.nan

    # Group data by date
    df['DATE'] = df['DATETIME'].dt.date
    unique_dates = df['DATE'].unique()

    for date in unique_dates:
        daily_df = df[df['DATE'] == date]
        pkl_filename = f"external_field_{date}.pkl"

        if os.path.exists(pkl_filename):
            daily_external = pd.read_pickle(pkl_filename)
        else:
            # Compute external field for each timestamp
            external_values = []
            for dt in daily_df['DATETIME']:
                Bx_ext, By_ext, Bz_ext = chaos_ext(datetime_to_decimal_year(dt), station_lat, station_lon, station_alt)
                H_external = np.sqrt(Bx_ext**2 + By_ext**2)
                external_values.append(H_external)
            daily_external = pd.Series(external_values, index=daily_df.index)
            daily_external.to_pickle(pkl_filename)

        df.loc[daily_df.index, 'H_external'] = daily_external

    df.drop(columns=['DATE'], inplace=True)
    return df


# Step 4: Estimate average night-time magnetospheric field from H_residual
def estimate_magnetospheric_component(df):
    night_mask = (df['DATETIME'].dt.hour >= 18) | (df['DATETIME'].dt.hour < 6)
    df['H_magnetospheric'] = df.loc[night_mask, 'H_residual']
    return df

# Step 5: Extract daytime EEJ signal from H_residual (still contains magnetospheric field)
def extract_eej_signal(df):
    day_mask = (df['DATETIME'].dt.hour >= 9) & (df['DATETIME'].dt.hour <= 15)
    df['EEJ'] = df.loc[day_mask, 'H_residual']
    return df


def fetch_dst_index(start_date, end_date):
    dst_records = []
    filepath="/home/amore/Documents/00Data/Dst_oct2024_apr2025.dat"
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 26:
                continue  # Skip malformed lines

            # Parse date from ID like DST2410*01PPX120
            id_str = parts[0]
            year = int("20" + id_str[3:5])
            month = int(id_str[5:7])
            day = int(id_str.split("*")[1][:2])

            try:
                hourly_values = [int(val) for val in parts[2:26]]  # Skip 2nd column (always 0), then 24 values
            except ValueError:
                continue  # Skip lines with invalid integer entries

            for hour, dst in enumerate(hourly_values):
                dt = datetime(year, month, day, hour)
                if start_date <= dt <= end_date:
                    dst_records.append({'DATETIME': dt, 'Dst': dst})

    return pd.DataFrame(dst_records)  

# Step 7: Perform linear regression between Dst and H_residual to estimate magnetospheric field
# Step 8: Subtract modeled magnetospheric contribution to get cleaned EEJ signal (HEEJ)

def perform_linear_regression(df, dst_data):
    # Ensure both are sorted by time
    df = df.sort_values('DATETIME')
    dst_data = dst_data.sort_values('DATETIME')

    # Merge with nearest previous Dst value (i.e., forward fill)
    merged = pd.merge_asof(df, dst_data[['DATETIME', 'Dst']], on='DATETIME', direction='backward')

    # Drop rows with missing data
    merged_clean = merged.dropna(subset=['Dst', 'H_residual'])

    # Prepare regression inputs
    x = merged_clean['Dst'].values.reshape(-1, 1)
    y = merged_clean['H_residual'].values.reshape(-1, 1)

    # Perform linear regression
    reg = LinearRegression().fit(x, y)
    merged_clean['Hmag_model'] = reg.predict(x)

    # Merge the modeled magnetospheric signal back into the full dataset
    merged = pd.merge(merged, merged_clean[['DATETIME', 'Hmag_model']], on='DATETIME', how='left')

    # Compute the cleaned EEJ signal
    merged['HEEJ'] = merged['H_residual'] - merged['Hmag_model']

    return merged, reg.coef_[0][0]


# After regression, keep only daytime values of the cleaned signal:
def extract_daytime_eej(df):
    df = df.copy()  # prevent SettingWithCopyWarning
    df['HEEJ_daytime'] = np.nan  # initialize the column with NaNs
    day_mask = (df['DATETIME'].dt.hour >= 9) & (df['DATETIME'].dt.hour <= 15)
    df.loc[day_mask, 'HEEJ_daytime'] = df.loc[day_mask, 'HEEJ']
    return df


def plot_monthly_magnetospheric_vs_dst(df, dst_data):
    df['YEAR'] = df['DATETIME'].dt.year
    df['MONTH'] = df['DATETIME'].dt.month

    for (year, month), group in df.groupby(['YEAR', 'MONTH']):
        start = group['DATETIME'].min()
        end = group['DATETIME'].max()
        dst_subset = dst_data[(dst_data['DATETIME'] >= start) & (dst_data['DATETIME'] <= end)]

        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        ax1.plot(group['DATETIME'], group['H_magnetospheric'], label='H_magnetospheric', color='blue')
        ax1.set_ylabel('H_magnetospheric (nT)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(dst_subset['DATETIME'], dst_subset['Dst'], label='Dst Index', color='red')
        ax2.set_ylabel('Dst Index (nT)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title(f'Magnetospheric Signal vs Dst - {year}-{month:02}')
        ax1.set_xlabel('Date')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'Magnetospheric_vs_Dst_{year}_{month:02}.png', dpi=300)
        plt.close()



# Step 10: Plot raw EEJ signal (before Dst correction)
def plot_superposed_epoch_eej(df):
    if 'EEJ' not in df.columns:
        print("EEJ not available. Skipping superposed epoch plots.")
        return

    df['YEAR'] = df['DATETIME'].dt.year
    df['MONTH'] = df['DATETIME'].dt.month
    df['HOUR_MIN'] = df['DATETIME'].dt.strftime('%H:%M')

    # Keep only daytime
    df_daytime = df[(df['DATETIME'].dt.hour >= 9) & (df['DATETIME'].dt.hour <= 15)].copy()
    
    # Round time to 30-minute bins
    df_daytime['DATETIME'] = df_daytime['DATETIME'].dt.floor('30T')
    df_daytime['HOUR_MIN'] = df_daytime['DATETIME'].dt.strftime('%H:%M')

    for (year, month), group in df_daytime.groupby(['YEAR', 'MONTH']):
        # Pivot: time of day (rows) × day (columns)
        pivot = group.pivot_table(index='HOUR_MIN', columns=group['DATETIME'].dt.date, values='EEJ')

        # Compute mean and std at each time bin
        mean_series = pivot.mean(axis=1)
        std_series = pivot.std(axis=1)

        # Convert HOUR_MIN back to datetime-like index for proper plotting
        time_labels = [datetime.strptime(t, '%H:%M') for t in mean_series.index]

        plt.figure(figsize=(10, 5))
        plt.plot(time_labels, mean_series, label='Mean EEJ', color='orange')
        plt.fill_between(time_labels, mean_series - std_series, mean_series + std_series,
                         color='orange', alpha=0.3, label='±1 Std Dev')
        
        # Format x-axis to show only time (HH:MM)
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.xlabel('Local Time (24-hour)')
        plt.ylabel('EEJ Magnetic Field (nT)')
        plt.title(f'Superposed Epoch of EEJ - {year}-{month:02}')
        plt.grid()
        plt.legend()
        filename = f'Superposed_EEJ_{year}_{month:02}.png'
        plt.savefig(filename, dpi=300)
        plt.close()



def plot_superposed_epoch_eej_vs_heej(df):
    if 'EEJ' not in df.columns or 'HEEJ' not in df.columns:
        print("EEJ or HEEJ not available. Skipping superposed comparison plots.")
        return

    df['YEAR'] = df['DATETIME'].dt.year
    df['MONTH'] = df['DATETIME'].dt.month
    df['HOUR_MIN'] = df['DATETIME'].dt.strftime('%H:%M')

    # Filter for daytime hours
    df_daytime = df[(df['DATETIME'].dt.hour >= 9) & (df['DATETIME'].dt.hour <= 15)].copy()
    df_daytime['DATETIME'] = df_daytime['DATETIME'].dt.floor('30min')
    df_daytime['HOUR_MIN'] = df_daytime['DATETIME'].dt.strftime('%H:%M')

    for (year, month), group in df_daytime.groupby(['YEAR', 'MONTH']):
        # Pivot tables for EEJ and HEEJ
        pivot_eej = group.pivot_table(index='HOUR_MIN', columns=group['DATETIME'].dt.date, values='EEJ')
        pivot_heej = group.pivot_table(index='HOUR_MIN', columns=group['DATETIME'].dt.date, values='HEEJ')

        # Compute mean values
        mean_eej = pivot_eej.mean(axis=1)
        mean_heej = pivot_heej.mean(axis=1)

        # Create time axis
        time_labels = [datetime.strptime(t, '%H:%M') for t in mean_eej.index]

        plt.figure(figsize=(10, 5))
        plt.plot(time_labels, mean_eej, label='Raw EEJ', color='orange')
        plt.plot(time_labels, mean_heej, label='Dst-corrected EEJ (HEEJ)', color='green')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xlabel('Local Time (24-hour)')
        plt.ylabel('Magnetic Field (nT)')
        plt.title(f'Superposed Epoch of EEJ vs HEEJ - {year}-{month:02}')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        filename = f'Superposed_EEJ_vs_HEEJ_{year}_{month:02}.png'
        plt.savefig(filename, dpi=300)
        plt.close()




def filter_by_month(df, year, month):
    # Ensure DATETIME is datetime type
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    # Filter for the specific year and month
    filtered_df = df[(df['DATETIME'].dt.year == year) & (df['DATETIME'].dt.month == month)]

    # Filter for times between 09:00 and 15:00
    filtered_df = filtered_df[(filtered_df['DATETIME'].dt.hour >= 9) & (filtered_df['DATETIME'].dt.hour < 15)]

    print(filtered_df)
    

# Main execution pipeline
def main():
    directory = '/home/amore/Documents/00Data/ENT0'
    output_file = 'testfile.pkl'

    df = load_entoto_data(directory)
    if df is None:
        print("No data files found.")
        return
   
    df = calculate_H_component(df)
   
    df = remove_internal_field(df)
   
    df = compute_external_field(df)
   
    df = extract_eej_signal(df)   
    
    df = estimate_magnetospheric_component(df)

    plot_superposed_epoch_eej(df)
    
    start_date, end_date = df['DATETIME'].min(), df['DATETIME'].max()
    dst_data = fetch_dst_index(start_date, end_date)
    
    if dst_data is not None and not dst_data.empty:
        df, k = perform_linear_regression(df, dst_data)
        print(f"Estimated scaling factor k: {k:.3f}")       
        df = extract_daytime_eej(df)    
        filter_by_month(df, 2025,3)        
        plot_monthly_magnetospheric_vs_dst(df, dst_data)        
        plot_superposed_epoch_eej_vs_heej(df)
        
    else:
        print("Warning: No Dst data available for the date range. Skipping regression and EEJ correction plot.")
   
    
    df.to_pickle(output_file)
    print(f"Processed data saved to {output_file}")
