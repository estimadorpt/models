import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from src.config import DATA_DIR


def dates_to_idx(timelist, reference_date):
    """Convert datetimes to numbers in reference to reference_date"""
    t = (reference_date - timelist) / np.timedelta64(1, "D")
    return np.asarray(t)


def standardize(series):
    """Standardize a pandas series"""
    return (series - series.mean()) / series.std()


def _consolidate_tracking_polls(df, party_cols, rolling_window_days=3):
    """
    Consolidates consecutive tracking polls from the same pollster.

    Args:
        df (pd.DataFrame): DataFrame containing poll data. Must include 
                           'Stratification', 'Fieldwork End', 'pollster', 
                           'sample_size', 'date', and party columns.
        party_cols (List[str]): List of column names representing party shares.
        rolling_window_days (int): The maximum number of days between the end
                                   of consecutive tracking polls to be grouped.

    Returns:
        pd.DataFrame: DataFrame with tracking polls consolidated.
    """
    # Ensure required columns are present
    required_cols = ['Stratification', 'Fieldwork End', 'pollster', 'sample_size', 'date'] + party_cols
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Warning: Missing required columns for tracking poll consolidation: {missing}. Skipping.")
        return df

    # Convert Fieldwork End to datetime (handle potential errors)
    df['Fieldwork End'] = pd.to_datetime(df['Fieldwork End'], errors='coerce')
    # Also convert Fieldwork Start for later use
    if 'Fieldwork Start' in df.columns:
         df['Fieldwork Start'] = pd.to_datetime(df['Fieldwork Start'], errors='coerce')
    else:
        # If Fieldwork Start doesn't exist, use Fieldwork End as a fallback
         df['Fieldwork Start'] = df['Fieldwork End']


    # Identify tracking polls (case-insensitive)
    # Make sure Stratification is string type before using .str accessor
    is_tracking = df['Stratification'].astype(str).str.contains("Tracking poll", na=False, case=False)
    tracking_polls = df[is_tracking].copy()
    non_tracking_polls = df[~is_tracking].copy()

    if tracking_polls.empty:
        print("No tracking polls identified based on 'Stratification' column.")
        return df # Return original df if no tracking polls

    print(f"Identified {len(tracking_polls)} potential tracking poll entries.")

    # Drop rows where Fieldwork End is NaT after conversion
    tracking_polls.dropna(subset=['Fieldwork End'], inplace=True)
    if tracking_polls.empty:
        print("No valid tracking polls remaining after dropping NaT Fieldwork End dates.")
        return non_tracking_polls # Return only non-tracking polls

    # Sort for grouping
    tracking_polls = tracking_polls.sort_values(by=['pollster', 'Fieldwork End'])

    # Calculate time difference between consecutive polls by the same pollster
    tracking_polls['time_diff'] = tracking_polls.groupby('pollster')['Fieldwork End'].diff()

    # Create group IDs for consecutive tracking polls within the window
    group_starts = (tracking_polls['pollster'] != tracking_polls['pollster'].shift(1)) | \
                   (tracking_polls['time_diff'].isna()) | \
                   (tracking_polls['time_diff'] > pd.Timedelta(days=rolling_window_days))
    tracking_polls['group_id'] = group_starts.cumsum()

    # --- Aggregation ---
    consolidated_polls_list = []

    for group_id, group in tracking_polls.groupby('group_id'):
        if len(group) == 1:
            # If only one poll in the group, keep it as is (remove helper columns)
            consolidated_polls_list.append(group.drop(columns=['time_diff', 'group_id']))
            continue

        # Calculate average and total sample size for the group
        # Ensure sample_size is numeric before summing/averaging
        group['sample_size'] = pd.to_numeric(group['sample_size'], errors='coerce').fillna(0)
        # Calculate the average sample size for the output N
        consolidated_sample_size = group['sample_size'].mean()
        # Calculate the sum of sample sizes (total weight) for weighted averaging party shares
        total_weight = group['sample_size'].sum()

        if consolidated_sample_size <= 0: # Avoid division by zero or nonsensical consolidation
             print(f"Warning: Group {group_id} for pollster {group['pollster'].iloc[0]} has zero or negative average sample size ({consolidated_sample_size}). Skipping consolidation for this group.")
             # Append original polls from the group instead of consolidating
             consolidated_polls_list.append(group.drop(columns=['time_diff', 'group_id']))
             continue

        # Calculate weighted average for party columns
        weighted_avg_parties = {}
        if total_weight <= 0:
            # Fallback: simple average if total weight is zero (e.g., all polls in group had 0 sample size)
            print(f"Warning: Total weight (sum of sample sizes) is zero for group {group_id}. Using simple average for party shares.")
            for party in party_cols:
                 if party in group.columns:
                     group[party] = pd.to_numeric(group[party], errors='coerce').fillna(0)
                     weighted_avg_parties[party] = group[party].mean()
                 else:
                     weighted_avg_parties[party] = 0
        else:
             # Normal weighted average calculation using total_weight
             for party in party_cols:
                 if party in group.columns:
                     # Ensure party shares are numeric
                     group[party] = pd.to_numeric(group[party], errors='coerce').fillna(0)
                     # Weight = sample_size * party_share
                     weighted_sum = (group[party] * group['sample_size']).sum()
                     # Use total_weight (sum of sample sizes) as denominator for weighted average
                     weighted_avg_parties[party] = weighted_sum / total_weight
                 else:
                      weighted_avg_parties[party] = 0 # Or handle missing party appropriately

        # Create consolidated row
        last_poll = group.iloc[-1] # Use data from the last poll in the group as reference
        consolidated_row_data = {
            'date': [last_poll['date']], # Use the original 'date' (publication date) of the last poll
            'pollster': [last_poll['pollster']],
            'sample_size': [int(round(consolidated_sample_size))], # Use the average sample size (rounded)
            'Fieldwork Start': [group['Fieldwork Start'].min()], # Use earliest start date in the group
            'Fieldwork End': [last_poll['Fieldwork End']], # Use latest end date in the group
            'Stratification': [f"Consolidated Tracking Poll ({len(group)} entries, avg N={int(round(consolidated_sample_size))})"], # Mark as consolidated, show avg N
            **weighted_avg_parties # Add the weighted party shares
        }
        # Add any other non-party, non-grouping columns from the last poll if they exist
        other_cols = [col for col in df.columns if col not in required_cols and col not in ['Fieldwork Start']]
        for col in other_cols:
            consolidated_row_data[col] = [last_poll[col]]

        consolidated_row = pd.DataFrame(consolidated_row_data)
        consolidated_polls_list.append(consolidated_row)

    if not consolidated_polls_list:
         print("No tracking polls were consolidated.")
         final_df = non_tracking_polls # Should be df if tracking_polls was empty initially
    else:
        consolidated_df = pd.concat(consolidated_polls_list, ignore_index=True)
        print(f"Consolidated {len(tracking_polls)} tracking polls into {len(consolidated_df)} entries (including {len(consolidated_df[consolidated_df['Stratification'].str.contains('Consolidated')])} consolidated groups).")

        # Combine non-tracking and consolidated tracking polls
        # Ensure columns match before concatenating - align columns based on the original df
        consolidated_df = consolidated_df.reindex(columns=df.columns.drop(['time_diff', 'group_id'], errors='ignore'))
        non_tracking_polls = non_tracking_polls.reindex(columns=df.columns.drop(['time_diff', 'group_id'], errors='ignore'))

        final_df = pd.concat([non_tracking_polls, consolidated_df], ignore_index=True)


    # Re-sort by date
    final_df = final_df.sort_values('date').reset_index(drop=True)

    # Drop the Stratification column now if it's no longer needed downstream
    # final_df = final_df.drop(columns=['Stratification'])

    return final_df


def load_marktest_polls():
    """Load polls from marktest_polls.csv"""
    # Read the CSV file, keeping necessary columns for consolidation
    cols_to_read = [
        "Pollster", "Date", "Sample Size", "Stratification",
        "Fieldwork Start", "Fieldwork End", "PS - Partido Socialista",
        "AD - Aliança Democrática", "BE - Bloco de Esquerda",
        "PCP-PEV - Coligação Democrática Unitária", "PAN - Pessoas-Animais-Natureza",
        "A - Aliança", "L - Livre", "Liberal - Iniciativa Liberal",
        "Chega - Chega", "Outros - Outros/Brancos/Nulos",
        "PSD - Partido Social Democrata", "CDS-PP - Partido Popular",
        "PAF - PSD/CDS - Portugal à Frente (2015)"
    ]
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'marktest_polls.csv'), usecols=cols_to_read)
    except ValueError as e:
         print(f"Error reading CSV, potentially missing columns: {e}")
         print("Attempting to read without specifying columns...")
         df = pd.read_csv(os.path.join(DATA_DIR, 'marktest_polls.csv'))
         # Check if essential cols are present after reading all
         if not all(c in df.columns for c in ["Pollster", "Date", "Sample Size"]):
              raise ValueError("Essential columns missing even after reading the full CSV.") from e


    # Convert 'Date' and fieldwork dates to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    # Handle fieldwork date conversion safely
    df['Fieldwork Start'] = pd.to_datetime(df['Fieldwork Start'], errors='coerce', format='%d/%m/%Y')
    df['Fieldwork End'] = pd.to_datetime(df['Fieldwork End'], errors='coerce', format='%d/%m/%Y')

    # Filter out the erroneous Eurosondagem poll from March 10, 2015
    df = df[~((df['Pollster'] == 'Eurosondagem') & (df['Date'] == pd.to_datetime('2015-03-10')))]

    # Rename columns to match the desired format
    rename_map = {
        'Date': 'date',
        'Pollster': 'pollster',
        'Sample Size': 'sample_size',
        'Stratification': 'Stratification', # Keep Stratification
        'Fieldwork Start': 'Fieldwork Start', # Keep Fieldwork Start
        'Fieldwork End': 'Fieldwork End', # Keep Fieldwork End
        'PS - Partido Socialista': 'PS',
        'AD - Aliança Democrática': 'AD',
        'BE - Bloco de Esquerda': 'BE',
        'PCP-PEV - Coligação Democrática Unitária': 'CDU',
        'PAN - Pessoas-Animais-Natureza': 'PAN',
        'L - Livre': 'L',
        'Liberal - Iniciativa Liberal': 'IL',
        'Chega - Chega': 'CH',
        'Outros - Outros/Brancos/Nulos': 'Others', # Keep for potential use later?
        'PSD - Partido Social Democrata': 'PSD',
        'CDS-PP - Partido Popular': 'CDS',
        "PAF - PSD/CDS - Portugal à Frente (2015)": 'PAF',
    }
    df = df.rename(columns=rename_map)

    # Define party columns AFTER renaming
    party_cols = ['PS', 'PSD', 'CH', 'IL', 'BE', 'CDU', 'CDS', 'PAN', 'L', 'AD', 'PAF']
    # Only keep party cols that actually exist in the dataframe after reading/renaming
    party_cols = [p for p in party_cols if p in df.columns]

    # Handle AD coalition (using existing party_cols)
    ad_components = [p for p in ['PSD', 'CDS', 'PAF'] if p in df.columns]
    if 'AD' not in df.columns:
         df['AD'] = 0 # Initialize AD if it doesn't exist
    # Ensure components are numeric before summing, fillna with 0
    df['AD'] = df['AD'].fillna(df[ad_components].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1))
    # Drop components only if they exist
    df.drop(columns=[col for col in ['PAF', 'PSD', 'CDS'] if col in df.columns], inplace=True)
    
    # Update party_cols list after AD consolidation
    final_party_cols = ['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']
    final_party_cols = [p for p in final_party_cols if p in df.columns] # Ensure they exist


    # Convert percentage values to floats (0-1 range) for relevant parties
    for col in final_party_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100
        # Fill NaNs resulting from conversion or originally present with 0
        df[col] = df[col].fillna(0)

    # Ensure sample_size is numeric and handle NaNs before consolidation
    df['sample_size'] = pd.to_numeric(df['sample_size'], errors='coerce')
    mean_sample = df['sample_size'].mean()
    if pd.isna(mean_sample) or mean_sample == 0:
        mean_sample = 1000  # Fallback if mean is NaN or zero
    df['sample_size'] = df['sample_size'].fillna(mean_sample).astype(int)


    # --- Consolidate Tracking Polls ---
    # Pass the final list of party columns present in the DataFrame
    df = _consolidate_tracking_polls(df, final_party_cols)
    # --- End Consolidation ---


    # Select and reorder columns for final output (excluding Stratification if desired)
    # Keep 'Fieldwork End' if needed downstream, otherwise drop
    final_columns = ['date', 'pollster', 'sample_size'] + final_party_cols # Add 'Fieldwork End' if needed
    # Ensure only existing columns are selected
    final_columns = [col for col in final_columns if col in df.columns]
    df = df[final_columns]

    # Sort by date (this might have been done in consolidation, but do it again for safety)
    df = df.sort_values('date').reset_index(drop=True)

    return df


def load_popstar_polls():
    """Load polls from popstar_sondagens_data.csv"""
    df = pd.read_csv(os.path.join(DATA_DIR, "popstar_sondagens_data.csv"), encoding='latin1', na_values=[' '])
    columns_to_convert = [col for col in df.columns if 'sondagens' in col]
    df[columns_to_convert] = df[columns_to_convert].astype(float)
    df.dropna(subset='PS nas sondagens', inplace=True)

    # Filter only the columns that we want to use, instituto, N, dia, sondagens
    columns = ['Instituto', 'N', 'Dia'] + [col for col in df.columns if 'sondagens' in col and 'PDR' not in col] + ['PSDCDS']
    df = df[columns]    

    df = df.rename(columns={
        'Dia': 'date', 
        'Instituto': 'pollster', 
        'PS nas sondagens': 'PS', 
        'PSD nas sondagens': 'PSD', 
        'BE nas sondagens': 'BE', 
        'CDU nas sondagens': 'CDU', 
        'CDS nas sondagens': 'CDS', 
        'Livre nas sondagens': 'L', 
        'PSDCDS': 'AD', 
        'N': 'sample_size'
    })
    
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df[['IL', 'CH']] = 0
    
    return df


def load_rr_polls():
    """Load polls from polls_renascenca.tsv"""
    polls_df = pd.read_csv(os.path.join(DATA_DIR, 'polls_renascenca.tsv'), sep='\t', na_values='—')

    # Rename columns
    polls_df = polls_df.rename(columns={
        'DATA': 'date', 
        'ORIGEM': 'pollster', 
        'ps': 'PS', 
        'psd': 'PSD', 
        'chega': 'CH', 
        'iniciativa liberal': 'IL', 
        'bloco de esquerda': 'BE', 
        'CDU PCP-PEV': 'CDU', 
        'PAN': 'PAN', 
        'CDS': 'CDS', 
        'livre': 'L', 
        'aliança democrática': 'AD', 
        'AMOSTRA': 'sample_size'
    })

    for col in ['PS', 'PSD', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'CDS', 'L', 'AD']:
        polls_df[col] = polls_df[col].str.replace('%', '').astype(float) / 100
    
    polls_df['date'] = pd.to_datetime(polls_df['date'], format='%Y-%m-%d')
    
    return polls_df


def load_election_results(election_dates, political_families):
    """Load election results from parquet files"""
    dfs = []
    
    # Print diagnostic information
    print("\n=== ELECTION DATES ===")
    print(f"Expected election dates: {election_dates}")
    
    available_files = glob.glob(os.path.join(DATA_DIR, 'legislativas_*.parquet'))
    print(f"Available election result files: {available_files}")
    
    matched_dates = []
    # For each legislativas_* file in the data folder, load the data and append it to the results_df
    for file in available_files:
        # Get the file date
        file_date = file.split('_')[-1].split('.')[0]
        print(f"\nProcessing file: {file} (date: {file_date})")
        
        # Get the date from election_dates which year matches the file date
        matching_dates = [date for date in election_dates if file_date in date]
        
        if not matching_dates:
            print(f"WARNING: No matching election date found for file {file}")
            continue
            
        election_date = matching_dates[0]
        matched_dates.append(election_date)
        print(f"Matched to election date: {election_date}")

        temp_results_df = pd.read_parquet(file)
        temp_results_df = temp_results_df.drop(columns='territoryName').sum().to_frame().T
        # Add sample size column with sum of all numeric columns
        temp_results_df['sample_size'] = temp_results_df.select_dtypes(include='number').sum(axis=1)
        temp_results_df['election_date'] = pd.to_datetime(election_date)
        temp_results_df['date'] = pd.to_datetime(election_date)
        temp_results_df['pollster'] = 'result'

        # Print available columns in temp_results_df
        print(f"Available columns in result file: {temp_results_df.columns.tolist()}")

        L_columns_to_sum = [col for col in temp_results_df if col in ['L', 'L/TDA']]
        temp_results_df['L'] = temp_results_df[L_columns_to_sum].sum(axis=1)
        if 'L/TDA' in temp_results_df.columns:
            temp_results_df = temp_results_df.drop(columns='L/TDA')

        AD_columns_to_sum = list(set(temp_results_df.filter(like='PPD/PSD').columns.tolist() + 
                                    temp_results_df.filter(like='CDS').columns.tolist()))
        print(f"AD columns to sum: {AD_columns_to_sum}")
        temp_results_df['AD'] = temp_results_df[AD_columns_to_sum].sum(axis=1)
        temp_results_df = temp_results_df.drop(columns=AD_columns_to_sum)

        # Keep only the columns we want
        columns_to_keep = ['date', 'election_date', 'pollster', 'sample_size', 'PS', 'AD', 'B.E.', 'PCP-PEV', 
                         'IL', 'PAN', 'CH', 'L']
        # Drop items from columns_to_keep that are not in temp_results_df
        available_columns = [col for col in columns_to_keep if col in temp_results_df.columns]
        missing_columns = [col for col in columns_to_keep if col not in temp_results_df.columns]
        
        print(f"Available columns in needed set: {available_columns}")
        print(f"Missing columns: {missing_columns}")
        
        # Calculate 'other' as the sum of all parties not in columns_to_keep
        party_columns = [col for col in temp_results_df.columns if col not in columns_to_keep and 
                       col not in ['sample_size', 'date', 'election_date', 'pollster', 
                                  'number_voters', 'percentage_voters', 'subscribed_voters']]
        temp_results_df['other'] = temp_results_df[party_columns].sum(axis=1)
        temp_results_df = temp_results_df[available_columns + ['other']]
        temp_results_df = temp_results_df.rename(columns={'B.E.': 'BE', 'PCP-PEV': 'CDU'})

        # Divide all numerical columns by 100 to avoid overflow in Multinomial
        for col in ['PS', 'AD', 'BE', 'CDU', 'IL', 'PAN', 'L', 'CH', 'other']:
            if col in temp_results_df.columns:
                temp_results_df[col] = temp_results_df[col] // 100
            else:
                temp_results_df[col] = 0
        
        # Recalculate sample size to be the sum of party votes
        available_parties = [col for col in ['PS', 'AD', 'BE', 'CDU', 'IL', 'PAN', 'L', 'CH'] if col in temp_results_df.columns]
        temp_results_df['sample_size'] = temp_results_df[available_parties].sum(axis=1)
        
        # Print expected vs actual parties
        print(f"Expected political families: {political_families}")
        print(f"Available political families in this file: {available_parties}")
        
        # Debug: Print election date being added
        print(f"---> Adding results for date: {temp_results_df['election_date'].iloc[0]}")
        dfs.append(temp_results_df)
    
    # Check if all election dates have corresponding result files
    unmatched_dates = [date for date in election_dates if date not in matched_dates]
    if unmatched_dates:
        print(f"\nWARNING: Some election dates have no corresponding result files: {unmatched_dates}")
        print("Attempting to handle missing election data gracefully...")
        
        # For each unmatched date, create a placeholder result with zeros
        for missing_date in unmatched_dates:
            print(f"Creating placeholder for {missing_date}")
            placeholder_df = pd.DataFrame({
                'date': [pd.to_datetime(missing_date)],
                'election_date': [pd.to_datetime(missing_date)],
                'pollster': ['result'],
                'sample_size': [1000]  # Placeholder sample size
            })
            
            # Add zero columns for each political family
            for party in political_families:
                placeholder_df[party] = 0
                
            placeholder_df['other'] = 0
            dfs.append(placeholder_df)
    
    if not dfs:
        print("ERROR: No election results could be loaded!")
        return pd.DataFrame()
        
    # Debug: Print dates just before concatenation
    print("\n--- Election dates in list before concat ---")
    for temp_df in dfs:
        if not temp_df.empty:
            print(f"  - {temp_df['election_date'].iloc[0]}")
    print("---------------------------------------------")
    
    df = pd.concat(dfs)
    print(f"\nFinal combined dataframe shape: {df.shape}")
    print(f"Final combined dataframe columns: {df.columns.tolist()}")
    
    # Check for any NaN values in the final df
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print("WARNING: NaN values in final election results dataframe:")
        print(nan_counts[nan_counts > 0])
    
    df.drop(columns=['other'], inplace=True)
    
    # Add countdown column with difference between election_date and date
    df['countdown'] = (df['election_date'] - df['date']).dt.days
    
    return df


def load_generic_predictor(file: str, name: str, freq: str, skiprows: int, sep: str = ";") -> pd.DataFrame:
    """Load a generic time-series predictor from a CSV file"""
    data = pd.read_csv(
        file,
        sep=sep,
        skiprows=skiprows,
    ).iloc[:, [0, 1]]
    data.columns = ["date", name]
    data = data.sort_values("date")

    # as timestamps variables:
    data.index = pd.period_range(
        start=data.date.iloc[0], periods=len(data), freq=freq
    )

    return data.drop("date", axis=1)


def merge_with_data(predictor: pd.DataFrame, freq: str, polls_train, polls_test, results_mult) -> List[pd.DataFrame]:
    """Merge polls and results with a predictor"""
    polls_train = polls_train.copy()
    polls_test = polls_test.copy()
    results_mult = results_mult.copy()
    dfs = []

    print(f"\n=== MERGING PREDICTOR DATA ===")
    print(f"Predictor data shape: {predictor.shape}")
    print(f"Predictor index range: {predictor.index.min()} to {predictor.index.max()}")
    
    # Check for any future dates outside predictor range
    for df_name, df in [("polls_train", polls_train), ("polls_test", polls_test), ("results_mult", results_mult)]:
        period_range = df["date"].dt.to_period(freq)
        min_period = period_range.min()
        max_period = period_range.max()
        print(f"{df_name} period range: {min_period} to {max_period}")
        
        # Check if any periods are outside the predictor range
        if min_period < predictor.index.min() or max_period > predictor.index.max():
            print(f"WARNING: {df_name} has dates outside predictor range")
            
            # Count rows outside range
            outside_range = (period_range < predictor.index.min()) | (period_range > predictor.index.max())
            outside_count = outside_range.sum()
            print(f"  - {outside_count} rows out of {len(df)} are outside predictor range")
            
            # Show sample of dates outside range
            if outside_count > 0:
                sample_outside = df.loc[outside_range, ["date"]].head(5)
                print(f"  - Sample dates outside range: {sample_outside['date'].tolist()}")

    for data in [polls_train, polls_test, results_mult]:
        # add freq to data
        data.index = data["date"].dt.to_period(freq)
        # merge with data
        before_join = data.copy()
        
        # Instead of plain join, use forward fill for missing values
        joined_data = data.join(predictor, how='left')
        
        # Check for NaN values after joining
        nan_count = joined_data.isna().sum().sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in joined data")
            
            # Forward fill missing predictor values (use last known value)
            # This creates a DataFrame with just the predictor columns filled
            filled_predictor = predictor.reindex(joined_data.index).ffill().bfill()
            
            # Apply the filled values to the joined data
            for col in predictor.columns:
                joined_data[col] = filled_predictor[col]
            
            # Check if we still have NaNs after filling
            remaining_nans = joined_data.isna().sum().sum()
            if remaining_nans > 0:
                print(f"WARNING: Still have {remaining_nans} NaN values after forward fill")
                # As a last resort, fill with median for each column
                for col in predictor.columns:
                    if joined_data[col].isna().any():
                        median_val = predictor[col].median()
                        print(f"Filling remaining NaNs in {col} with median: {median_val}")
                        joined_data[col] = joined_data[col].fillna(median_val)
        
        lost_rows = before_join[~before_join.index.isin(joined_data.index)]
        
        if not lost_rows.empty:
            print(f"Lost {len(lost_rows)} rows during join:")
            print(lost_rows)
        
        dfs.append(joined_data.reset_index(drop=True))

    return dfs


def create_government_status(election_dates, government_parties, political_families):
    """
    Create government status dataframe
    
    With the forward-facing government_parties dictionary, each entry represents
    which parties won that election and stayed in power until the next one.
    """
    print("\n=== Creating Government Status ===")
    print(f"Election dates: {election_dates}")
    print(f"Government parties: {government_parties}")
    
    # Create a dataframe with election dates as index and parties as columns
    government_status = pd.DataFrame(index=election_dates, columns=political_families)
    
    # Fill in government status for each election
    for i, date in enumerate(sorted(election_dates)):
        # Get the government parties for this election
        if date in government_parties:
            parties = government_parties[date]
            print(f"Election {date}: Government parties = {parties}")
            
            # Fill in status for each political family
            for party in political_families:
                # Check if this party is in government (either directly or as part of AD coalition)
                is_governing = (party in parties) or (party == 'AD' and ('PSD' in parties and 'CDS' in parties))
                government_status.loc[date, party] = 1 if is_governing else 0
        else:
            print(f"Warning: No government party data for election {date}")
            # Default to all opposition if we don't have data
            for party in political_families:
                government_status.loc[date, party] = 0
    
    # Print the resulting matrix for verification
    print("\nGovernment status matrix:")
    print(government_status)
    
    return government_status.astype(int)  # Ensure integer type


def cast_as_multinomial(df: pd.DataFrame, political_families: List[str]) -> pd.DataFrame:
    """Convert percentages to counts for multinomial modeling"""
    df = df.copy()
    
    # Detailed diagnostics on data
    print("\n=== DIAGNOSTIC INFORMATION ===")
    print("Sample size data type:", df["sample_size"].dtype)
    print("Sample size non-numeric values:", df[pd.to_numeric(df["sample_size"], errors='coerce').isna()][["date", "pollster", "sample_size"]])
    
    # Check for NaN values in political family columns
    for party in political_families:
        nan_count = df[party].isna().sum()
        if nan_count > 0:
            print(f"NaN values in {party} column: {nan_count}")
            print(df[df[party].isna()][["date", "pollster", party]])
    
    # Check for rows where party percentages sum to > 1
    party_sum = df[political_families].sum(axis=1)
    invalid_rows = df[party_sum > 1]
    if not invalid_rows.empty:
        print(f"Found {len(invalid_rows)} rows where party percentages sum to > 1")
        print(invalid_rows[["date", "pollster"] + political_families])
        print("Sum of percentages:", party_sum[party_sum > 1])
    
    # Original conversion logic
    df[political_families] = (
        (df[political_families])
        .mul(df["sample_size"], axis=0)
        .round()
        .fillna(0)
        .astype(int)
    )
    df["sample_size"] = df[political_families].sum(1)

    return df


def train_test_split(polls: pd.DataFrame, test_cutoff: pd.Timedelta = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split polls into train and test sets"""
    last_election = polls.election_date.unique()[-1]
    polls_train = polls[polls.election_date != last_election]
    polls_test = polls[polls.election_date == last_election]

    if test_cutoff:
        test_cutoff_ = last_election - test_cutoff
    else:
        test_cutoff_ = last_election - pd.Timedelta(30, "D")

    polls_train = pd.concat(
        [polls_train, polls_test[polls_test.date <= test_cutoff_]]
    )
    polls_test = polls_test[polls_test.date > test_cutoff_]

    return polls_train, polls_test 