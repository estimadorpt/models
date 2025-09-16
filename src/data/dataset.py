import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

from src.config import DATA_DIR
from src.data.loaders import (
    load_marktest_polls,
    load_popstar_polls,
    load_rr_polls,
    load_election_results,
    load_generic_predictor,
    merge_with_data,
    create_government_status,
    cast_as_multinomial,
    train_test_split,
    standardize
)
from src.data.flexible_loaders import (
    load_election_results_flexible,
    GeographicLevelManager
)
from typing import Optional, Literal


class ElectionDataset:
    """Class for loading and preparing election data for modeling
    
    Supports multi-level geographic aggregation: parish, municipality, district, national
    """
    
    political_families = [
        'PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD'
    ]

    # Historical elections with actual results
    historical_election_dates = [
        '2024-03-10',
        '2022-01-30',
        '2019-10-06',
        '2015-10-04',
        '2011-06-05',
    ]
    
    # Parties that WON each election and governed AFTER that date until the next election
    government_parties = {
        '2009-09-27': ['PS'],          # PS won in 2009 and governed until 2011
        '2011-06-05': ['PSD', 'CDS'],  # PSD-CDS coalition won in 2011
        '2015-10-04': ['PS'],          # PS formed government after 2015 election
        '2019-10-06': ['PS'],          # PS won in 2019
        '2022-01-30': ['PS'],          # PS won in 2022
        '2024-03-10': ['AD'],          # AD won in 2024 and currently in power
    }

    def __init__(
        self,
        election_date: str,
        baseline_timescales: List[int],  # Annual cycle
        election_timescales: List[int],  # Pre-campaign and official campaign periods
        test_cutoff: pd.Timedelta = None,
        geographic_level: Optional[Literal['parish', 'municipality', 'district', 'national']] = 'district',
        election_type: str = 'parliamentary'
    ):
        self.election_date = election_date
        self.baseline_timescales = baseline_timescales
        self.election_timescales = election_timescales
        self.test_cutoff = test_cutoff
        self.geographic_level = geographic_level or 'district'  # Default to district for backward compatibility
        self.election_type = election_type
        
        # Create a combined list of all election cycle end dates (historical + target)
        self.all_election_dates = sorted(list(set(self.historical_election_dates + [self.election_date]))) 
        # self.election_dates is kept as historical only for compatibility in some places, clarify usage.
        # Primarily, coordinates should use all_election_dates or historical_election_dates explicitly.
        self.election_dates = self.historical_election_dates.copy() # Retain for existing logic if needed, but prefer specific lists.
        
        # Initialize geographic level manager
        self.geo_manager = GeographicLevelManager(default_level=self.geographic_level)
        print(f"\nInitialized geographic level manager for {self.geographic_level} level")
        
        # Check if our target election is in the future (not in historical dates)
        is_future_election = election_date not in self.historical_election_dates
        if is_future_election:
            print(f"\nTarget election date {election_date} is in the future (not in historical data).")
            # We'll still use this date for forecasting, but not for training
        else:
            print(f"\nTarget election date {election_date} is in historical data.")
        
        # Load data
        self.polls = self._load_polls()
        
        # Load results using flexible geographic system
        self.results_national = self._load_results_flexible('national')
        
        if self.geographic_level == 'national':
            # For national level, use same data for both
            self.results_mult_district = self.results_national.copy()
        elif self.geographic_level == 'district':
            # Load district results for backward compatibility
            self.results_mult_district = self._load_results_flexible('district')
        else:
            # For municipality/parish level, load that level
            self.results_mult_district = self._load_results_flexible(self.geographic_level)
            
        # Store the primary results based on geographic level
        self.results_primary = self._load_results_flexible(self.geographic_level)
        
        # Ensure results_national has a unique DatetimeIndex based on election_date (NEW)
        if not self.results_national.empty:
            try:
                self.results_national['election_date'] = pd.to_datetime(self.results_national['election_date'])
                # Keep the first occurrence if duplicates exist based on date
                self.results_national = self.results_national.drop_duplicates(subset=['election_date'], keep='first')
                self.results_national = self.results_national.set_index('election_date', drop=False) # Keep column too
                # Sort index just in case
                self.results_national = self.results_national.sort_index()
                print(f"Processed results_national index: Is unique? {self.results_national.index.is_unique}")
            except KeyError:
                print("Error: 'election_date' column not found in results_national. Cannot set index.")
            except Exception as e:
                print(f"Error processing results_national index: {e}")
        else:
            print("Warning: results_national is empty.")
        
        # Ensure results_mult_district is processed (Existing logic, ensure it's still relevant)
        # This logic might now be redundant if results_mult is no longer used directly, but let's keep for now
        # RENAME: self.results_mult -> self.results_mult_district
        if not self.results_mult_district.empty:
            try:
                self.results_mult_district['election_date'] = pd.to_datetime(self.results_mult_district['election_date'])
                # District results NEED the district key, so we don't drop duplicates here based on date alone
                # Instead, ensure index is set if needed elsewhere (though model uses specific indices now)
                # self.results_mult_district = self.results_mult_district.set_index(['election_date', 'Circulo'], drop=False) # Example if multi-index needed
                # self.results_mult_district = self.results_mult_district.sort_index()
                print(f"Processed results_mult_district index: Is unique? {self.results_mult_district.index.is_unique}") # Check original index
            except KeyError:
                print("Error: 'election_date' column not found in results_mult_district. Cannot set index.")
            except Exception as e:
                print(f"Error processing results_mult_district index: {e}")
        else:
            print("Warning: results_mult_district is empty.")
        
        # --- Add Geographic Coordinates based on level ---
        self.unique_geographic_divisions = []
        self.unique_districts = []  # Keep for backward compatibility
        
        if not self.results_primary.empty:
            if 'geographic_id' in self.results_primary.columns:
                self.unique_geographic_divisions = sorted(self.results_primary['geographic_id'].unique())
                print(f"\nLoaded {len(self.unique_geographic_divisions)} unique {self.geographic_level}-level divisions")
                
                # For backward compatibility with district-based code
                if self.geographic_level == 'district' and 'Circulo' in self.results_mult_district.columns:
                    self.unique_districts = sorted(self.results_mult_district['Circulo'].unique())
                elif 'geographic_id' in self.results_primary.columns:
                    self.unique_districts = self.unique_geographic_divisions.copy()
            else:
                print(f"\nWarning: Could not extract geographic divisions from {self.geographic_level} results.")
        # --- End Add Geographic Coordinates ---
        
        # Convert ALL polls to multinomial format first
        all_polls_mult = self.cast_as_multinomial(self.polls)

        # --- Conditional Train/Test Split based on mode (future forecast vs historical validation) ---
        if test_cutoff is None:
            # Training mode for future election: Use ALL polls for training
            print(f"\nUsing ALL {len(all_polls_mult)} polls for training (test_cutoff is None).")
            self.polls_train = all_polls_mult.copy()
            self.polls_test = pd.DataFrame(columns=all_polls_mult.columns)  # Empty test set
        else:
            # Validation mode for historical election: Split based on test_cutoff
            print(f"\nSplitting polls into train/test sets using test_cutoff: {test_cutoff}")
            # Assuming train_test_split works correctly based on the last historical election
            # It needs to operate on all_polls_mult now
            self.polls_train, self.polls_test = train_test_split(all_polls_mult, test_cutoff)
            print(f"  - polls_train shape: {self.polls_train.shape}")
            print(f"  - polls_test shape: {self.polls_test.shape}")

        # --- Continue with processing based on polls_train ---
        # Process data - Ensure factorize uses polls_train
        if not self.polls_train.empty:
            _, self.unique_elections = self.polls_train["election_date"].factorize()
            _, self.unique_pollsters = self.polls_train["pollster"].factorize()
        else:
            print("Warning: polls_train is empty after split. Unique elections/pollsters will be empty.")
            self.unique_elections = pd.Index([])
            self.unique_pollsters = pd.Index([])
            
        # For inference, results_oos should contain only HISTORICAL results
        self.results_oos = self.results_national.copy()
        
        # Ensure results_oos dates align with historical dates
        self.results_oos = self.results_oos[self.results_oos['election_date'].isin(pd.to_datetime(self.historical_election_dates))]
        
        # Debug info about results
        print(f"\n=== RESULTS DATA ===")
        print(f"All results shape: {self.results_national.shape}, dates: {self.results_national['election_date'].unique()}")
        print(f"Historical elections: {self.historical_election_dates}")
        print(f"All election cycles (for model coords): {self.all_election_dates}")
        print(f"Results_oos shape: {self.results_oos.shape}, dates: {self.results_oos['election_date'].unique()}")
        
        self.government_status = create_government_status(
            self.all_election_dates, # Use all dates for gov status matrix
            self.government_parties, 
            self.political_families
        )

        self._load_predictors()
        (
            self.results_preds,
            self.campaign_preds,
        ) = self._standardize_continuous_predictors()
        
        # Final diagnostic check for NaN values in key dataframes
        print("\n=== FINAL NaN DIAGNOSTICS ===")
        for name, df in [
            ("polls", self.polls), 
            ("polls_mult", all_polls_mult),
            ("results_national", self.results_national),
            ("polls_train", self.polls_train),
            ("polls_test", self.polls_test),
            ("results_oos", self.results_oos),
            ("government_status", self.government_status),
            ("results_preds", self.results_preds),
            ("campaign_preds", self.campaign_preds)
        ]:
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                print(f"NaN values in {name}: {nan_count}")
                for col in df.columns:
                    col_nan = df[col].isna().sum()
                    if col_nan > 0:
                        print(f"  - {col}: {col_nan} NaN values")
                        # Print sample of rows with NaN
                        nan_rows = df[df[col].isna()].head(3)
                        print(f"  - Sample rows with NaN in {col}:\n{nan_rows}\n")

    def _load_polls(self):
        """Load poll data from various sources"""
        polls_df = load_marktest_polls()
        
        # Uncomment below to load from other sources if needed
        # polls_df = pd.concat([load_popstar_polls(), load_rr_polls()])

        # Clean pollster names
        polls_df['pollster'] = polls_df['pollster'].str.replace('Eurosondagem ', 'Eurosondagem')
        polls_df['pollster'] = polls_df['pollster'].str.replace('* Aximage', 'Aximage')
        polls_df['pollster'] = polls_df['pollster'].str.replace('CESOP-UCP', 'UCP').str.replace('CESOP/UCP', 'UCP').str.replace('Catolica', 'UCP')
        polls_df['pollster'] = polls_df['pollster'].str.replace('Pitagórica', 'Pitagorica')

        # Fill missing values for all political families
        for party in self.political_families:
            polls_df[party] = polls_df[party].fillna(0)
            
        # Ensure sample_size is numeric and has no NaN
        polls_df['sample_size'] = pd.to_numeric(polls_df['sample_size'], errors='coerce')
        # Replace any NaN with average
        mean_sample = polls_df['sample_size'].mean()
        if pd.isna(mean_sample):
            mean_sample = 1000  # Fallback if we can't calculate mean
        polls_df['sample_size'] = polls_df['sample_size'].fillna(mean_sample)
        
        # Calculate 'other' after handling NaNs
        polls_df['other'] = 1 - polls_df[['PS', 'AD', 'BE', 'CDU', 'IL', 'PAN', 'L', 'CH']].sum(axis=1)
        
        # Add election date and countdown
        polls_df['election_date'] = pd.to_datetime(polls_df.apply(self.find_closest_election_date, axis=1))
        polls_df = polls_df[polls_df['election_date'].notna()]
        polls_df['countdown'] = (polls_df['election_date'] - polls_df['date']).dt.days
        polls_df.drop(columns=['other'], inplace=True)

        return polls_df

    def find_closest_election_date(self, row):
        """Find the closest upcoming election date for a given poll date"""
        # Include both historical elections and the target election
        # Use the combined list of all election dates
        election_datetime = [pd.to_datetime(date) for date in self.all_election_dates]
        row_date = row['date']
        
        # Get the last historical election date (most recent, which is at index 0)
        last_historical_election = pd.to_datetime(self.historical_election_dates[0])
        
        # If the poll date is after the last historical election, assign the target date
        if row_date > last_historical_election and self.election_date not in self.historical_election_dates:
            return pd.to_datetime(self.election_date)
            
        # Otherwise, find the closest upcoming election date
        filtered_dates = [date for date in election_datetime if date > row_date]
        return min(filtered_dates, default=pd.NaT)

    def _load_results(self, aggregate_national=True):
        """Load election results - legacy method for backward compatibility"""
        return load_election_results(self.election_dates, self.political_families, aggregate_national)
    
    def _load_results_flexible(self, level: str) -> pd.DataFrame:
        """Load election results at specified geographic level using flexible system"""
        print(f"Loading election results at {level} level...")
        
        if level == 'national':
            # For national level, use aggregate_national=True for backward compatibility
            return load_election_results_flexible(
                election_dates=self.election_dates,
                political_families=self.political_families,
                aggregate_national=True,
                election_type=self.election_type
            )
        else:
            # Use the new flexible aggregation system
            return load_election_results_flexible(
                election_dates=self.election_dates,
                political_families=self.political_families,
                aggregation_level=level,
                election_type=self.election_type
            )

    def cast_as_multinomial(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert percentages to counts for multinomial modeling"""
        return cast_as_multinomial(df, self.political_families)

    def _load_predictors(self):
        """Load economic and other predictors"""
        # Comment out GDP loading
        """
        self.gdp_data = load_generic_predictor(
            file=os.path.join(DATA_DIR, "gdp.csv"),
            name="gdp",
            freq="Q",
            skiprows=0,
            sep=","
        )
        self.polls_train, self.polls_test, self.results_mult = merge_with_data(
            self.gdp_data, freq="Q", 
            polls_train=self.polls_train, 
            polls_test=self.polls_test, 
            results_mult=self.results_mult
        )
        """
        # Add empty gdp column to avoid errors
        for df in [self.polls_train, self.polls_test, self.results_national]:
            if 'gdp' not in df.columns:
                df['gdp'] = 0.0  # Use float constant value instead of NaN
        return

    def _standardize_continuous_predictors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Standardize continuous predictors to help with sampling and setting priors."""
        continuous_predictors = ["gdp"]
        
        # Add a 'type' column to distinguish polls from results
        polls_data = self.polls_train[["date"] + continuous_predictors].assign(type='poll')
        results_data = self.results_national[["date"] + continuous_predictors].assign(type='result')
        
        self.continuous_predictors = (
            pd.concat([polls_data, results_data])
             .set_index(["date", "type"])
             .sort_index() # Removed standardization
        )
        
        # Extract predictors, aligning results_preds with all_election_dates
        results_preds = self.continuous_predictors.xs('result', level='type')
        campaign_preds = self.continuous_predictors.xs('poll', level='type')
 
        # Reindex results_preds to include all election dates, forward-filling missing values
        all_election_dates_dt = pd.to_datetime(self.all_election_dates)
        results_preds = results_preds.reindex(all_election_dates_dt, method='ffill').fillna(0.0) # Pad with 0 if no prior data

        return results_preds, campaign_preds
    
    def prepare_observed_data(self):
        """Prepare observed poll results for posterior predictive checks"""
        results_data = self.results_national if self.geographic_level == 'national' else self.results_primary
        
        observed_data = pd.DataFrame({
            'date': results_data['date'] if 'date' in results_data.columns else results_data['election_date'],
            'pollster': results_data.get('pollster', 'Election Result'),
        })
        for party in self.political_families:
            if party in results_data.columns and 'sample_size' in results_data.columns:
                observed_data[party] = results_data[party] / results_data['sample_size']
            else:
                observed_data[party] = 0
        return observed_data

    def generate_oos_data(self, posterior):
        """Generate out-of-sample data for forecasting"""
        # Get target election date
        target_election = pd.to_datetime(self.election_date)
        
        # Generate dates from 1 year before election to election day
        start_date = target_election - pd.Timedelta(days=365)
        new_dates = pd.date_range(start=start_date, end=target_election)
        
        # Generate countdown values - number of days until election
        countdown_values = [(target_election - date).days for date in new_dates]
        
        print(f"Generated dates from {new_dates[0].date()} to {new_dates[-1].date()}")
        print(f"Countdown values from {max(countdown_values)} to {min(countdown_values)}")
        
        # Use most reliable pollster for forecast data
        pollster_counts = self.polls_train['pollster'].value_counts()
        most_common_pollster = pollster_counts.idxmax() if len(pollster_counts) > 0 else self.unique_pollsters[0]
        
        # Use a consistent sample size for all forecast points
        avg_sample_size = int(self.polls_train['sample_size'].mean()) if len(self.polls_train) > 0 else 1000
        
        # Create the forecast data frame - one row per date
        oos_data = pd.DataFrame({
            "date": new_dates,
            "countdown": countdown_values,
            "election_date": target_election,
            "pollster": most_common_pollster,
            "sample_size": avg_sample_size
        })
        
        # Add zero-filled columns for each party
        for party in self.political_families:
            oos_data[party] = 0
                
        return new_dates, oos_data 