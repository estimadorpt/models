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


class ElectionDataset:
    """Class for loading and preparing election data for modeling"""
    
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
        baseline_timescales=[365],
        election_timescales=[60],
        test_cutoff: pd.Timedelta = None,
    ):
        self.election_date = election_date
        self.baseline_timescales = baseline_timescales
        self.election_timescales = election_timescales
        self.test_cutoff = test_cutoff
        
        # Use only historical elections for training, not future elections
        self.election_dates = self.historical_election_dates.copy()
        
        # Check if our target election is in the future (not in historical dates)
        if election_date not in self.historical_election_dates:
            print(f"\nTarget election date {election_date} is in the future (not in historical data).")
            # We'll still use this date for forecasting, but not for training
        else:
            print(f"\nTarget election date {election_date} is in historical data.")
        
        # Load data
        self.polls = self._load_polls()
        
        # Filter polls to only include those for historical elections with known results
        historical_polls = self.polls[self.polls["election_date"].isin(self.historical_election_dates)]
        print(f"\nFiltered polls: {len(self.polls)} total polls -> {len(historical_polls)} historical polls")
        
        # For training we use only historical polls
        self.polls_mult = self.cast_as_multinomial(historical_polls)
        self.results_mult = self._load_results()

        # Split data into train/test
        (
            self.polls_train,
            self.polls_test,
        ) = train_test_split(self.polls_mult, test_cutoff)

        # Process data
        _, self.unique_elections = self.polls_train["election_date"].factorize()
        _, self.unique_pollsters = self.polls_train["pollster"].factorize()
        
        # For inference, use all historical elections
        self.results_oos = self.results_mult.copy()
        
        # Debug info about results
        print(f"\n=== RESULTS DATA ===")
        print(f"All results shape: {self.results_mult.shape}, dates: {self.results_mult['election_date'].unique()}")
        print(f"Historical elections: {self.historical_election_dates}")
        print(f"Results_oos shape: {self.results_oos.shape}, dates: {self.results_oos['election_date'].unique()}")
        
        self.government_status = create_government_status(
            self.election_dates, 
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
            ("polls_mult", self.polls_mult),
            ("results_mult", self.results_mult),
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
        all_elections = self.historical_election_dates.copy()
        if self.election_date not in all_elections:
            all_elections.append(self.election_date)
            
        election_datetime = [pd.to_datetime(date) for date in all_elections]
        row_date = row['date']
        filtered_dates = [date for date in election_datetime if date > row_date]
        return min(filtered_dates, default=pd.NaT)

    def _load_results(self):
        """Load election results"""
        return load_election_results(self.election_dates, self.political_families)

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
        for df in [self.polls_train, self.polls_test, self.results_mult]:
            if 'gdp' not in df.columns:
                df['gdp'] = 0  # Use constant value instead of NaN
        return

    def _standardize_continuous_predictors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Standardize continuous predictors to help with sampling and setting priors."""
        continuous_predictors = ["gdp"]
        
        # Add a 'type' column to distinguish polls from results
        polls_data = self.polls_train[["date"] + continuous_predictors].assign(type='poll')
        results_data = self.results_mult[["date"] + continuous_predictors].assign(type='result')
        
        self.continuous_predictors = (
            pd.concat([polls_data, results_data])
            .set_index(["date", "type"])
            .sort_index()
        )
        
        # Simplified to avoid standardization issues with constant values
        results_preds = self.continuous_predictors.xs('result', level='type')
        campaign_preds = self.continuous_predictors.xs('poll', level='type')

        return results_preds, campaign_preds
    
    def prepare_observed_data(self):
        """Prepare observed poll results for posterior predictive checks"""
        observed_data = pd.DataFrame({
            'date': self.results_mult['date'],
            'pollster': self.results_mult['pollster'],
        })
        for party in self.political_families:
            observed_data[party] = self.results_mult[party] / self.results_mult['sample_size']
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