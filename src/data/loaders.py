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

    # --- DEBUG: Dates Before Consolidation ---
    if not tracking_polls.empty:
        unique_dates_before = pd.to_datetime(tracking_polls['date']).dt.date.unique()
        print(f"DEBUG LOADER: Unique tracking poll dates BEFORE consolidation (count={len(unique_dates_before)}):\n{sorted(unique_dates_before)}")
    else:
        print("DEBUG LOADER: No tracking polls found before consolidation step.")
    # --- END DEBUG ---

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

    # --- DEBUG: Dates After Consolidation ---
    if not final_df.empty:
        # Identify which rows were originally tracking polls (either consolidated or kept as is)
        # This requires checking if the poll was in the original tracking_polls DataFrame index or if it's a new consolidated row
        # A simpler proxy: check if the final_df row came from consolidated_df or the original tracking_polls if group size was 1
        
        # Let's just print unique dates of the tracking poll subset in the final dataframe for simplicity
        final_tracking_mask = final_df['Stratification'].astype(str).str.contains("Tracking poll", na=False, case=False) | \
                              final_df['Stratification'].astype(str).str.contains("Consolidated", na=False, case=False)
        final_tracking_dates = final_df.loc[final_tracking_mask, 'date']
        unique_dates_after = pd.to_datetime(final_tracking_dates).dt.date.unique()
        print(f"DEBUG LOADER: Unique tracking poll dates AFTER consolidation (count={len(unique_dates_after)}):\n{sorted(unique_dates_after)}")
    else:
        print("DEBUG LOADER: Final dataframe is empty after consolidation step.")
    # --- END DEBUG ---

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

    # Clean the 'Pollster' column
    df['Pollster'] = df['Pollster'].str.lstrip('* ')

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


def load_election_results(election_dates, political_families, aggregate_national=True):
    """Load election results from parquet files.

    Args:
        election_dates (List[str]): List of election dates (YYYY-MM-DD).
        political_families (List[str]): List of party names to focus on.
        aggregate_national (bool): If True, aggregate results nationally.
                                    If False, return results per district (Circulo).

    Returns:
        pd.DataFrame: Election results, either aggregated nationally or per district.
    """
    dfs = []
    code_to_district = load_code_to_district_map() # Load map once
    if not code_to_district:
        print("Error: Failed to load territory code map. Cannot proceed with district-level results.")
        # Depending on desired behavior, maybe return empty or only allow national aggregation.
        # For now, let's allow national but fail district-level.
        if not aggregate_national:
            return pd.DataFrame()

    # Print diagnostic information
    print("\n=== ELECTION DATES ===")
    print(f"Expected election dates: {election_dates}")

    available_files = glob.glob(os.path.join(DATA_DIR, 'legislativas_*.parquet'))
    print(f"Available election result files: {available_files}")

    matched_dates = []
    for file in available_files:
        file_date_str = file.split('_')[-1].split('.')[0]
        print(f"\nProcessing file: {file} (date: {file_date_str})")

        matching_dates = [date for date in election_dates if file_date_str in date]
        if not matching_dates:
            print(f"WARNING: No matching election date found for file {file}")
            continue
        election_date = matching_dates[0]
        matched_dates.append(election_date)
        print(f"Matched to election date: {election_date}")

        try:
            raw_results_df = pd.read_parquet(file)
        except Exception as e:
            print(f"Error reading parquet file {file}: {e}")
            continue # Skip this file

        # --- Add Circulo Column from territoryCode --- 
        if 'territoryCode' not in raw_results_df.columns:
             print(f"WARNING: 'territoryCode' column not found in {file}. Cannot determine districts for this file.")
             if not aggregate_national:
                 print("Skipping file for district-level results.")
                 continue
             else: 
                 # If aggregating nationally, we can proceed but won't have district info from this file
                 print("Proceeding with national aggregation despite missing territoryCode.")
                 # We won't have a 'Circulo' column for this file's data
                 pass # Let the aggregation happen without district grouping
        else:
             # Ensure territoryCode is string
             raw_results_df['territoryCode'] = raw_results_df['territoryCode'].astype(str)
             # Extract the 2-digit code, allowing for optional 'LOCAL-' prefix
             extracted_codes = raw_results_df['territoryCode'].str.extract(r'(?:LOCAL-)?(\d{2})', expand=False)
             raw_results_df['district_code'] = extracted_codes.fillna('XX') # Handle non-matches with 'XX'
             
             # Map code to district name
             raw_results_df['Circulo'] = raw_results_df['district_code'].map(code_to_district)
             
             # Check for codes that didn't map (excluding the 'XX' for non-matches)
             unmapped_codes = raw_results_df[raw_results_df['Circulo'].isna() & (raw_results_df['district_code'] != 'XX')]['district_code'].unique()
             if len(unmapped_codes) > 0:
                 print(f"Warning: Found territory codes in {file} with no match in code map: {unmapped_codes.tolist()}")
             # Fill NaNs in Circulo (resulting from unmapped codes or initial XX) with a placeholder
             raw_results_df['Circulo'] = raw_results_df['Circulo'].fillna('Unknown District')
             # Can optionally drop helper columns now if not needed later
             # raw_results_df = raw_results_df.drop(columns=['district_code'])

        # --- Continue with existing processing --- 
        raw_results_df['election_date'] = pd.to_datetime(election_date)
        raw_results_df['date'] = pd.to_datetime(election_date)
        raw_results_df['pollster'] = 'result'

        # Standardize party names early
        raw_results_df = raw_results_df.rename(columns={'B.E.': 'BE', 'PCP-PEV': 'CDU'})

        # --- Party Consolidation (pre-aggregation) ---
        l_cols = [col for col in raw_results_df.columns if col in ['L', 'L/TDA']]
        if l_cols:
            raw_results_df['L'] = raw_results_df[l_cols].sum(axis=1)
            if 'L/TDA' in raw_results_df.columns:
                raw_results_df = raw_results_df.drop(columns='L/TDA')
        elif 'L' not in raw_results_df.columns:
             raw_results_df['L'] = 0 

        ad_components_pattern = ['PPD/PSD', 'CDS-PP']
        ad_cols_to_sum = [col for col in raw_results_df.columns if any(pat in col for pat in ad_components_pattern)]
        if ad_cols_to_sum:
            # Ensure components are numeric, fillna(0) before summing
            raw_results_df['AD'] = raw_results_df[ad_cols_to_sum].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
            raw_results_df = raw_results_df.drop(columns=ad_cols_to_sum)
        elif 'AD' not in raw_results_df.columns:
            raw_results_df['AD'] = 0
            
        # Identify relevant parties present in the data
        present_parties = [p for p in political_families if p in raw_results_df.columns]

        # Define base columns to keep
        base_cols = ['date', 'election_date', 'pollster']
        if not aggregate_national and 'Circulo' in raw_results_df.columns:
            base_cols.append('Circulo')
            
        # Identify 'other' parties 
        other_party_cols = [col for col in raw_results_df.columns if col not in base_cols + present_parties + 
                            ['territoryCode', 'district_code', # Explicitly exclude original code columns
                             'null_votes', 'blank_votes', # Exclude non-party vote types if present
                             'number_voters', 'percentage_voters', 'subscribed_voters'] and 
                             isinstance(raw_results_df[col].iloc[0], (int, float))] # Rough check for numeric type
        if other_party_cols:
            # Ensure 'other' columns are numeric before summing
            raw_results_df['other'] = raw_results_df[other_party_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
        else:
            raw_results_df['other'] = 0

        # Define columns to keep explicitly 
        cols_to_keep = base_cols + present_parties + ['other']
        final_cols = [col for col in cols_to_keep if col in raw_results_df.columns]
        temp_results_df = raw_results_df[final_cols].copy()

        # --- Aggregation or District-Level Preparation ---
        if aggregate_national:
            # Aggregate nationally: sum votes across all rows (districts/subdistricts)
            numeric_cols = present_parties + ['other']
            metadata_cols = ['date', 'election_date', 'pollster']
            
            numeric_cols_present = [col for col in numeric_cols if col in temp_results_df.columns]
            if not numeric_cols_present:
                 aggregated_numeric = pd.DataFrame(columns=numeric_cols)
            else:
                 # Ensure numeric before summing
                 aggregated_numeric = temp_results_df[numeric_cols_present].apply(pd.to_numeric, errors='coerce').fillna(0).sum().to_frame().T
            
            aggregated_metadata = temp_results_df[metadata_cols].iloc[[0]].reset_index(drop=True)
            aggregated_df = pd.concat([aggregated_metadata, aggregated_numeric], axis=1)
            
            present_parties_agg = [p for p in present_parties if p in aggregated_df.columns]
            if present_parties_agg:
                 # Ensure numeric before sum for sample size
                aggregated_df['sample_size'] = aggregated_df[present_parties_agg].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
            else:
                 aggregated_df['sample_size'] = 0
            
            processed_df = aggregated_df

        else: # Keep district level
            if 'Circulo' not in temp_results_df.columns:
                 print(f"ERROR: 'Circulo' column missing for district grouping in file {file}. Cannot produce district results.")
                 processed_df = pd.DataFrame()
            else:
                # Group by Circulo and sum votes 
                grouping_cols = [col for col in base_cols if col in temp_results_df.columns] # Use valid base cols
                numeric_cols = present_parties + ['other']
                numeric_cols_present = [col for col in numeric_cols if col in temp_results_df.columns]
                
                if not numeric_cols_present:
                    processed_df = temp_results_df[grouping_cols].drop_duplicates().reset_index(drop=True)
                    processed_df['sample_size'] = 0
                else:
                    # Ensure numeric before grouping sum
                    for col in numeric_cols_present: 
                         temp_results_df[col] = pd.to_numeric(temp_results_df[col], errors='coerce').fillna(0)
                    processed_df = temp_results_df.groupby(grouping_cols, as_index=False)[numeric_cols_present].sum()
                    
                    present_parties_dist = [p for p in present_parties if p in processed_df.columns]
                    if present_parties_dist:
                        processed_df['sample_size'] = processed_df[present_parties_dist].sum(axis=1)
                    else:
                        processed_df['sample_size'] = 0
            
        # --- Final Processing (common to both) ---
        if processed_df.empty:
             print(f"Skipping final processing for {file} due to empty dataframe.")
             continue 
             
        for party in political_families:
            if party not in processed_df.columns:
                processed_df[party] = 0

        final_ordered_cols = base_cols + political_families + ['sample_size', 'other']
        final_ordered_cols = [col for col in final_ordered_cols if col in processed_df.columns]
        processed_df = processed_df[final_ordered_cols]

        dfs.append(processed_df)

    # --- Handle Unmatched Dates (Placeholder Creation) ---
    unmatched_dates = [date for date in election_dates if date not in matched_dates]
    if unmatched_dates:
        print(f"\nWARNING: Some election dates have no corresponding result files: {unmatched_dates}")
        print("Attempting to handle missing election data gracefully...")
        for missing_date in unmatched_dates:
            print(f"Creating placeholder for {missing_date}")
            placeholder_data = {
                'date': [pd.to_datetime(missing_date)],
                'election_date': [pd.to_datetime(missing_date)],
                'pollster': ['result'],
                'sample_size': [0],
                'other': [0]
            }
            if not aggregate_national:
                 placeholder_data['Circulo'] = ['Unknown District'] 

            for party in political_families:
                placeholder_data[party] = 0

            placeholder_df = pd.DataFrame(placeholder_data)
            dfs.append(placeholder_df)

    if not dfs:
        print("ERROR: No election results could be loaded or created!")
        return pd.DataFrame()

    # --- Concatenate Final Results ---
    try:
        final_df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
         print(f"Error during concatenation: {e}")
         for i, item_df in enumerate(dfs):
             print(f"DF {i} shape: {item_df.shape}, columns: {item_df.columns.tolist()}")
         return pd.DataFrame()

    # Check for NaN values
    nan_counts = final_df.isna().sum()
    if nan_counts.sum() > 0:
        print("WARNING: NaN values found in final election results dataframe:")
        print(nan_counts[nan_counts > 0])
        numeric_cols_final = political_families + ['sample_size', 'other']
        for col in numeric_cols_final:
            if col in final_df.columns and final_df[col].isna().any():
                print(f"Filling NaNs in column '{col}' with 0.")
                final_df[col] = final_df[col].fillna(0)

    if 'other' in final_df.columns:
        final_df = final_df.drop(columns=['other'])

    if aggregate_national:
        final_df['countdown'] = (final_df['election_date'] - final_df['date']).dt.days

    # Convert vote counts to integers
    vote_cols = [p for p in political_families if p in final_df.columns]
    for col in vote_cols + ['sample_size']:
         if col in final_df.columns:
             final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0).astype(int)

    if not aggregate_national and 'Circulo' not in final_df.columns and not final_df.empty:
         print("ERROR: Final district-level DataFrame is missing the 'Circulo' column.")

    print(f"Finished loading election results. Final shape: {final_df.shape}")
    return final_df


def load_district_config() -> Dict[str, int]:
    """
    Loads the mapping between electoral districts (Circulos) and the number of seats (Deputados).

    Reads data from 'data/legislativas_2024_circulos_deputados.csv'.

    Returns:
        Dict[str, int]: A dictionary where keys are district names and values are the number of seats.
    """
    file_path = os.path.join(DATA_DIR, 'legislativas_2024_circulos_deputados.csv')
    try:
        df = pd.read_csv(file_path)
        # Use 'Circulo' for district name and 'Deputados' for seats
        district_seats = df.set_index('Circulo')['Deputados'].to_dict()
        print(f"Loaded district configuration for {len(district_seats)} districts from {file_path}")
        return district_seats
    except FileNotFoundError:
        print(f"Error: District configuration file not found at {file_path}")
        return {}
    except KeyError:
        print(f"Error: Required columns ('Circulo', 'Deputados') not found in {file_path}")
        return {}
    except Exception as e:
        print(f"An error occurred while loading district config: {e}")
        return {}


def load_code_to_district_map() -> Dict[str, str]:
    """
    Loads the mapping between the two-digit territory code and the district name (Circulo).

    Reads data from 'data/legislativas_2024_codigos_circulos.csv'.

    Returns:
        Dict[str, str]: A dictionary where keys are the territory codes (as strings, e.g., '01')
                        and values are the district names (Circulo).
    """
    file_path = os.path.join(DATA_DIR, 'legislativas_2024_codigos_circulos.csv')
    code_map = {}
    try:
        df = pd.read_csv(file_path, dtype={'Codigo': str})
        # Remove duplicates, keeping the first occurrence of each code
        df = df.drop_duplicates(subset='Codigo', keep='first')
        # Ensure 'Codigo' is treated as string, potentially pad with 0 if needed (though dtype=str should handle)
        df['Codigo'] = df['Codigo'].astype(str).str.zfill(2)
        code_map = df.set_index('Codigo')['Circulo'].to_dict()
        print(f"Loaded territory code to district map for {len(code_map)} unique codes from {file_path}")
        return code_map
    except FileNotFoundError:
        print(f"Error: Territory code map file not found at {file_path}")
        return {}
    except KeyError:
        print(f"Error: Required columns ('Codigo', 'Circulo') not found in {file_path}")
        return {}
    except Exception as e:
        print(f"An error occurred while loading territory code map: {e}")
        return {}


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