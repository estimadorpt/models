import json
import sys
import os
from typing import Dict, List, Tuple
from urllib.request import urlopen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

import os

import arviz
import numpy as np
import pandas as pd
import pymc as pm

import pytensor



# compute_test_value is 'off' by default, meaning this feature is inactive
#pytensor.config.compute_test_value = 'off' # Use 'warn' to activate this feature
pytensor.config.exception_verbosity = 'high' # Use 'high' to see the full error stack
pytensor.config.optimizer= 'fast_compile'
#pytensor.config.mode= 'DebugMode'
#pytensor.config.on_unused_input='warn'

import pytensor.tensor as pt
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import linalg
import xarray as xr



def dates_to_idx(timelist, reference_date):
    """Convert datetimes to numbers in reference to reference_date"""
    t = (reference_date - timelist) / np.timedelta64(1, "D")
    return np.asarray(t)


def standardize(series):
    """Standardize a pandas series"""
    return (series - series.mean()) / series.std()

from datetime import datetime
import glob

class ElectionsModel:

    political_families = [
        'PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD'
    ]

    election_dates = [
            '2024-03-10',
            '2022-01-30',
            '2019-10-06',
            '2015-10-04',
            '2011-06-05',

        ]
    
    government_parties = {
        '2011-06-05': ['PSD', 'CDS'],  # PSD-CDS coalition
        '2015-10-04': ['PSD', 'CDS'],  # PSD-CDS coalition (initially)
        '2019-10-06': ['PS'],
        '2022-01-30': ['PS'],
        '2024-03-10': ['PS'],  # Assuming PS was in government before this election
    }
    # political_families = [
    #     'ps', 'chega', 'iniciativa liberal', 'bloco de esquerda', 'CDU PCP-PEV', 'PAN', 'livre', 'aliança democrática'
    # ]


    def __init__(
        self,
        election_date: str,
        baseline_timescales=[180],
        election_timescales=[14],
        weights: List[float] = None,
        test_cutoff: pd.Timedelta = None,
    ):
        
        self.gp_config = {
            "baseline_lengthscale": baseline_timescales,
            "election_lengthscale": election_timescales,            
            "kernel": "matern52",
            "zerosum": True,
            "variance_limit": 0.8,
            "variance_weight": weights,
        }

        self.polls = self._load_polls()
        self.polls_mult = self.cast_as_multinomial(self.polls)
        self.results_mult = self._load_results()

        (
            self.polls_train,
            self.polls_test,
        ) = self._train_split(self.polls_mult, test_cutoff)


        _, self.unique_elections = self.polls_train["election_date"].factorize()
        _, self.unique_pollsters = self.polls_train["pollster"].factorize()
        self.results_oos = self.results_mult[
            self.results_mult.election_date != election_date
        ].copy()

        self.government_parties = self.government_parties
        self.government_status = self._create_government_status()

        self._load_predictors()
        (
            self.results_preds,
            self.campaign_preds,
        ) = self._standardize_continuous_predictors()


    @staticmethod
    def _train_split(
        polls: pd.DataFrame, test_cutoff: pd.Timedelta = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    
    def _standardize_continuous_predictors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Substract mean and divide by std to help with sampling and setting priors."""
        continuous_predictors = ["gdp"]
        
        # Add a 'type' column to distinguish polls from results
        polls_data = self.polls_train[["date"] + continuous_predictors].assign(type='poll')
        results_data = self.results_mult[["date"] + continuous_predictors].assign(type='result')
        
        self.continuous_predictors = (
            pd.concat([polls_data, results_data])
            .set_index(["date", "type"])
            .sort_index()
        )
        
        cont_preds_stdz = self.continuous_predictors.copy()
        for col in continuous_predictors:
            cont_preds_stdz[col] = standardize(self.continuous_predictors[col])

        results_preds = cont_preds_stdz.xs('result', level='type')
        campaign_preds = cont_preds_stdz.xs('poll', level='type')

        return results_preds, campaign_preds
    
    def _create_government_status(self):
        government_status = pd.DataFrame(index=self.election_dates, columns=self.political_families)
        for date, parties in self.government_parties.items():
            for party in self.political_families:
                if party in parties or (party == 'AD' and ('PSD' in parties and 'CDS' in parties)):
                    government_status.loc[date, party] = 1  # Use 1 for government
                else:
                    government_status.loc[date, party] = 0  # Use 0 for opposition
        return government_status.astype(int)  # Ensure integer type

    def find_closest_election_date(self, row):

        #convert election dates to datetime
        election_datetime = [datetime.strptime(date, '%Y-%m-%d') for date in self.election_dates]
        row_date = row['date']
        filtered_dates = [date for date in election_datetime if date > row_date]
        return min(filtered_dates, default=pd.NaT)


    def _load_popstar_polls(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "popstar_sondagens_data.csv"), encoding='latin1', na_values=[' '])
        columns_to_convert = [col for col in df.columns if 'sondagens' in col]
        df[columns_to_convert] = df[columns_to_convert].astype(float)
        df.dropna(subset='PS nas sondagens', inplace=True)

        #convert dia column to datetime

        #filter only the columns that we want to use, instituto, N, dia, sondagens
        columns = ['Instituto', 'N', 'Dia'] + [col for col in df.columns if 'sondagens' in col and 'PDR' not in col] + ['PSDCDS']
        df = df[columns]    

        df = df.rename(columns={'Dia': 'date', 'Instituto': 'pollster', 'PS nas sondagens': 'PS', 'PSD nas sondagens': 'PSD', 'BE nas sondagens': 'BE', 'CDU nas sondagens': 'CDU', 'CDS nas sondagens': 'CDS', 'Livre nas sondagens': 'L', 'PSDCDS': 'AD', 'N': 'sample_size'})
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

        df[['IL', 'CH']] = 0
        return df
    
    def _load_rr_polls(self):
        polls_df = pd.read_csv(os.path.join(DATA_DIR, 'polls_renascenca.tsv'), sep='\t', na_values='—')

        #rename columns
        polls_df = polls_df.rename(columns={'DATA': 'date', 'ORIGEM': 'pollster', 'ps': 'PS', 'psd': 'PSD', 'chega': 'CH', 'iniciativa liberal' : 'IL', 'bloco de esquerda': 'BE', 'CDU PCP-PEV': 'CDU', 'PAN': 'PAN', 'CDS': 'CDS', 'livre': 'L', 'aliança democrática': 'AD', 'AMOSTRA': 'sample_size'})

        for col in ['PS', 'PSD', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'CDS', 'L', 'AD']:
            polls_df[col] = polls_df[col].str.replace('%', '').astype(float) / 100
        
        polls_df['date'] = pd.to_datetime(polls_df['date'], format='%Y-%m-%d')
        return polls_df
    
    def _load_marktest_polls(self):
        # Read the CSV file
        df = pd.read_csv('data/marktest_polls.csv')

        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # Rename columns to match the desired format
        df = df.rename(columns={
            'Date': 'date',
            'Pollster': 'pollster',
            'Sample Size': 'sample_size',
            'PS - Partido Socialista': 'PS',
            'AD - Aliança Democrática': 'AD',
            'BE - Bloco de Esquerda': 'BE',
            'PCP-PEV - Coligação Democrática Unitária': 'CDU',
            'PAN - Pessoas-Animais-Natureza': 'PAN',
            'L - Livre': 'L',
            'Liberal - Iniciativa Liberal': 'IL',
            'Chega - Chega': 'CH',
            'Outros - Outros/Brancos/Nulos': 'Others',
            'PSD - Partido Social Democrata': 'PSD',
            'CDS-PP - Partido Popular': 'CDS'
        })

        # Select and reorder columns
        columns = ['date', 'pollster', 'sample_size', 'PS', 'PSD', 'CH', 'IL', 'BE', 'CDU', 'CDS', 'PAN', 'L', 'AD']
        df = df[columns]

        # Handle AD coalition
        df['AD'] = df['AD'].fillna(df['PSD'] + df['CDS'])
        df['PSD'] = df['PSD'].where(df['AD'].isna(), 0)
        df['CDS'] = df['CDS'].where(df['AD'].isna(), 0)

        # Convert percentage values to floats
        for col in df.columns[3:]:
            df[col] = df[col].astype(float) / 100

        # Sort by date
        df = df.sort_values('date')

        return df

    def _load_polls(self):

        polls_df = self._load_marktest_polls()
        #polls_df = pd.concat([self._load_popstar_polls(), self._load_rr_polls()])

        #replace 'Eurosondagem ' with 'Eurosondagem'
        polls_df['pollster'] = polls_df['pollster'].str.replace('Eurosondagem ', 'Eurosondagem')
        polls_df['pollster'] = polls_df['pollster'].str.replace('* Aximage', 'Aximage')
        #replace CESOP-UCP, CESOP/UCP, Catolica  with UCP
        polls_df['pollster'] = polls_df['pollster'].str.replace('CESOP-UCP', 'UCP').str.replace('CESOP/UCP', 'UCP').str.replace('Catolica', 'UCP')
        #replace 'Pitagórica' with 'Pitagorica'
        polls_df['pollster'] = polls_df['pollster'].str.replace('Pitagórica', 'Pitagorica')


        polls_df['AD'] = polls_df['AD'].fillna(polls_df['PSD'] + polls_df['CDS'])
        #drop psd and cds columns
        polls_df = polls_df.drop(columns=['PSD', 'CDS'])
        #fill na with 0
        polls_df = polls_df.fillna(0)
        polls_df['other'] = 1 - polls_df[['PS', 'AD', 'BE', 'CDU', 'IL', 'PAN', 'L', 'CH']].sum(axis=1)

        polls_df['election_date'] = pd.to_datetime(polls_df.apply(self.find_closest_election_date, axis=1))
        polls_df = polls_df[polls_df['election_date'].notna()]
        polls_df['countdown'] = (polls_df['election_date'] - polls_df['date']).dt.days

        #drop polls without an election date
        return(polls_df)

    def _load_results(self):
        dfs= []
        #for each legislativas_* file in the data folder, load the data and append it to the results_df
        for file in glob.glob(os.path.join(DATA_DIR, 'legislativas_*.parquet')):
            #get the file date
            file_date = file.split('_')[-1].split('.')[0]
            #get the date from election_dates which year matches the file date
            election_date = [date for date in self.election_dates if file_date in date][0]


            temp_results_df = pd.read_parquet(file)
            temp_results_df = temp_results_df.drop(columns='territoryName').sum().to_frame().T
            #add sample size column with sum of all numeric columns
            temp_results_df['sample_size'] = temp_results_df.select_dtypes(include='number').sum(axis=1)
            temp_results_df['election_date'] = pd.to_datetime(election_date)
            temp_results_df['date'] =  pd.to_datetime(election_date)
            temp_results_df['pollster'] = 'result'

            L_columns_to_sum = [col for col in temp_results_df if col in ['L', 'L/TDA']]
            temp_results_df['L'] = temp_results_df[L_columns_to_sum].sum(axis=1)
            if 'L/TDA' in temp_results_df.columns:
                temp_results_df = temp_results_df.drop(columns='L/TDA')

            AD_columns_to_sum = temp_results_df.filter(like='PPD/PSD').columns.tolist() + temp_results_df.filter(like='CDS').columns.tolist()
            temp_results_df['AD'] = temp_results_df[AD_columns_to_sum].sum(axis=1)
            temp_results_df = temp_results_df.drop(columns=AD_columns_to_sum)

            #keep only the columns we want PS, AD, B.E., PCP-PEV, IL, PAN, L, CH, sum all other columns to OTHER
            columns_to_keep = ['date', 'election_date', 'pollster', 'sample_size', 'PS', 'AD', 'B.E.', 'PCP-PEV', 'IL', 'PAN', 'CH','L']
            #drop items from columns_to_keep that are not in temp_results_df
            columns_to_keep = [col for col in columns_to_keep if col in temp_results_df.columns]
            temp_results_df['other'] = temp_results_df.drop(columns=columns_to_keep).sum(axis=1)
            temp_results_df = temp_results_df[columns_to_keep+['other']]
            temp_results_df = temp_results_df.rename(columns={'B.E.': 'BE', 'PCP-PEV': 'CDU'})

            #divide all numerical columns by 100
            for col in ['PS', 'AD', 'BE', 'CDU', 'IL', 'PAN', 'L', 'CH']:
                if col in temp_results_df.columns:
                    temp_results_df[col] = temp_results_df[col] // 100
                else:
                    temp_results_df[col] = 0
            
            #recalculate sample size to be the sum of ['PS', 'AD', 'BE', 'CDU', 'IL', 'PAN', 'L', 'CH', 'other']
            temp_results_df['sample_size'] = temp_results_df[['PS', 'AD', 'BE', 'CDU', 'IL', 'PAN', 'L', 'CH']].sum(axis=1)
            
            dfs.append(temp_results_df)
        df = pd.concat(dfs)
        #add countdown column with difference between election_date and date
        df['countdown'] = (df['election_date'] - df['date']).dt.days
        df = df.fillna(0)
        return df


    def cast_as_multinomial(self, df: pd.DataFrame) -> pd.DataFrame:
            
            df = df.copy()
            df[self.political_families] = (
                (df[self.political_families])
                .mul(df["sample_size"], axis=0)
                .round()
                .fillna(0)
                .astype(int)
            )
            df["sample_size"] = df[self.political_families].sum(1)

            return df

    def _load_generic_predictor(self,
            file: str, name: str, freq: str, skiprows: int, sep: str = ";"
        ) -> pd.DataFrame:

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

    def _merge_with_data(
        self, predictor: pd.DataFrame, freq: str
    ) -> List[pd.DataFrame]:
        polls_train = self.polls_train.copy()
        polls_test = self.polls_test.copy()
        results_mult = self.results_mult.copy()
        dfs = []

        for data in [polls_train, polls_test, results_mult]:
            # add freq to data
            data.index = data["date"].dt.to_period(freq)
            # merge with data
            before_join = data.copy()
            joined_data = data.join(predictor)
            lost_rows = before_join[~before_join.index.isin(joined_data.index)]
            
            if not lost_rows.empty:
                print(f"Lost {len(lost_rows)} rows during join:")
                print(lost_rows)
            
            dfs.append(joined_data.reset_index(drop=True))

        return dfs
    
    def _load_predictors(self):
        self.gdp_data = self._load_generic_predictor(
            file=os.path.join(DATA_DIR, "gdp.csv"),
            name="gdp",
            freq="Q",
            skiprows=0,
            sep=","
        )
        self.polls_train, self.polls_test, self.results_mult = self._merge_with_data(
            self.gdp_data, freq="Q"
        )
        return
        


    def _build_coords(self, polls: pd.DataFrame = None):
        data = polls if polls is not None else self.polls_train

        COORDS = {
            "observations": data.index,
            "parties_complete": self.political_families,
        }
        pollster_id, COORDS["pollsters"] = data["pollster"].factorize(sort=True)
        countdown_id, COORDS["countdown"] = data["countdown"].values, np.arange(
            data["countdown"].max() + 1
        )
        election_id, COORDS["elections"] = data["election_date"].factorize()
        COORDS["elections_observed"] = COORDS["elections"][:-1]

        return pollster_id, countdown_id, election_id, COORDS

    def _build_data_containers(
        self,
        polls: pd.DataFrame = None,
    ) :

        if polls is None:
            polls = self.polls_train
        is_here = polls[self.political_families].astype(bool).astype(int)
        
        # Ensure there are no NaNs in government_status
        if self.government_status.isnull().values.any():
            raise ValueError("government_status contains NaN values. Please check the data.")
        
        data_containers = dict(
            election_idx=pm.Data("election_idx", self.election_id, dims="observations"),
            pollster_idx=pm.Data("pollster_idx", self.pollster_id, dims="observations"),
            countdown_idx=pm.Data(
                "countdown_idx", self.countdown_id, dims="observations"
            ),
            stdz_gdp=pm.Data(
                "stdz_gdp",
                self.campaign_preds["gdp"].to_numpy(),
                dims="observations",
            ),
            election_gdp=pm.Data(
                "election_gdp",
                self.results_preds["gdp"].to_numpy(),
                dims="elections",
            ),
            observed_N=pm.Data(
                "observed_N",
                polls["sample_size"].to_numpy(),
                dims="observations",
            ),
            observed_polls=pm.Data(
                "observed_polls",
                polls[self.political_families].to_numpy(),
                dims=("observations", "parties_complete"),
            ),
            results_N=pm.Data(
                "results_N",
                self.results_oos["sample_size"].to_numpy(),
                dims="elections_observed",
            ),
            
            observed_results=pm.Data(
                "observed_results",
                self.results_oos[self.political_families].to_numpy(),
                dims=("elections_observed", "parties_complete"),
            ),

            non_competing_parties_results=pm.Data(
                "non_competing_parties_results",
                self.results_mult[self.political_families].astype(bool).astype(int).replace(to_replace=0, value=-10).replace(to_replace=1, value=0).to_numpy(),
                dims=("elections", "parties_complete"),
            ),
            
            non_competing_polls_additive = pm.Data(
                "non_competing_polls_additive",
                is_here.replace(to_replace=0, value=-10).replace(to_replace=1, value=0).to_numpy(),
                dims=("observations", "parties_complete")
            ),
            non_competing_polls_multiplicative = pm.Data(
                "non_competing_polls_multiplicative",
                is_here.to_numpy(),
                dims=("observations", "parties_complete")
            ),
            government_status=pm.Data(
                "government_status",
                self.government_status.values.astype(int),
                dims=("elections", "parties_complete"),
            ),
        )

        print("Shape of government_status in _build_data_containers:", data_containers["government_status"].shape.eval())
        return data_containers
    
    def build_model(self, polls: pd.DataFrame = None) -> pm.Model:
        (
            self.pollster_id,
            self.countdown_id,
            self.election_id,
            self.coords,
        ) = self._build_coords(polls)

        with pm.Model(coords=self.coords) as model:
            data_containers = self._build_data_containers(polls)

            # --------------------------------------------------------
            #                   BASELINE COMPONENTS
            # --------------------------------------------------------
            party_baseline_sd = pm.HalfNormal("party_baseline_sd", sigma=0.5)
            party_baseline = pm.ZeroSumNormal(
                "party_baseline", sigma=party_baseline_sd, dims="parties_complete"
            )

            election_party_baseline_sd = pm.HalfNormal("election_party_baseline_sd", sigma=0.05)
            election_party_baseline = pm.ZeroSumNormal(
                "election_party_baseline",
                sigma=election_party_baseline_sd,
                dims=("elections", "parties_complete"),
            )

            # --------------------------------------------------------
            #                   FUNDAMENTAL COMPONENTS
            # --------------------------------------------------------

            # Define independent GDP coefficients for government and opposition
            gdp_coeff_gov = pm.Normal("gdp_coeff_gov", mu=0, sigma=0.2)
            gdp_coeff_opp = pm.Normal("gdp_coeff_opp", mu=0, sigma=0.2)

            # Compute GDP effect for polls using pm.math.switch
            gdp_effect_polls = pm.Deterministic(
                "gdp_effect_polls",
                pm.math.switch(
                    data_containers["government_status"][data_containers["election_idx"]],
                    gdp_coeff_gov,
                    gdp_coeff_opp
                ),
                dims=("observations", "parties_complete"),
            )

            # Compute GDP effect for elections using pm.math.switch
            gdp_effect_elections = pm.Deterministic(
                "gdp_effect_elections",
                pm.math.switch(
                    data_containers["government_status"],
                    gdp_coeff_gov,
                    gdp_coeff_opp
                ),
                dims=("elections", "parties_complete"),
            )

            # --------------------------------------------------------
            #               TIME-VARYING COMPONENTS (Multiple Timescales with Weighting)
            # --------------------------------------------------------

            # Loop over baseline timescales and sum their contributions
            baseline_gp_contributions = []
            for i, baseline_lengthscale in enumerate(self.gp_config["baseline_lengthscale"]):
                cov_func_baseline = pm.gp.cov.Matern52(input_dim=1, ls=baseline_lengthscale)
                gp_baseline = pm.gp.HSGP(cov_func=cov_func_baseline, m=[10], c=1.5)
                phi_baseline, sqrt_psd_baseline = gp_baseline.prior_linearized(X=self.coords["countdown"][:, None])

                coord_name = f"gp_basis_baseline_{i}"
                if coord_name not in model.coords:
                    model.add_coords({coord_name: np.arange(gp_baseline.n_basis_vectors)})

                gp_coef_baseline = pm.Normal(
                    f"gp_coef_baseline_{baseline_lengthscale}",
                    mu=0,
                    sigma=1,
                    dims=(coord_name, "parties_complete")
                )

                baseline_contrib = pm.Deterministic(
                    f"party_time_effect_baseline_{baseline_lengthscale}",
                    pt.dot(phi_baseline, gp_coef_baseline * sqrt_psd_baseline[:, None]),
                    dims=("countdown", "parties_complete")
                )
                baseline_gp_contributions.append(baseline_contrib)

            # Sum the contributions from different baseline timescales
            party_time_effect = pm.Deterministic(
                "party_time_effect",
                sum(baseline_gp_contributions),
                dims=("countdown", "parties_complete")
            )

            # Weights for baseline component
            lsd_baseline = pm.Normal("lsd_baseline", mu=-2, sigma=0.5)
            lsd_party_effect = pm.ZeroSumNormal(
                "lsd_party_effect_party_amplitude",
                sigma=0.2,
                dims="parties_complete"
            )
            party_time_weight = pm.Deterministic(
                "party_time_weight",
                pt.exp(lsd_baseline + lsd_party_effect),
                dims="parties_complete"
            )
            party_time_effect_weighted = pm.Deterministic(
                "party_time_effect_weighted",
                party_time_effect * party_time_weight[None, :],
                dims=("countdown", "parties_complete")
            )

            # Loop over election timescales and sum their contributions
            election_gp_contributions = []
            for i, election_lengthscale in enumerate(self.gp_config["election_lengthscale"]):
                cov_func_election = pm.gp.cov.Matern52(input_dim=1, ls=election_lengthscale)
                gp_election = pm.gp.HSGP(cov_func=cov_func_election, m=[10], c=1.5)
                phi_election, sqrt_psd_election = gp_election.prior_linearized(X=self.coords["countdown"][:, None])

                coord_name = f"gp_basis_election_{i}"
                if coord_name not in model.coords:
                    model.add_coords({coord_name: np.arange(gp_election.n_basis_vectors)})

                gp_coef_election = pm.Normal(
                    f"gp_coef_election_{election_lengthscale}",
                    mu=0,
                    sigma=1,
                    dims=(coord_name, "parties_complete", "elections")
                )

                election_contrib = pm.Deterministic(
                    f"election_party_time_effect_{election_lengthscale}",
                    pt.tensordot(phi_election, gp_coef_election * sqrt_psd_election[:, None, None], axes=(1, 0)),
                    dims=("countdown", "parties_complete", "elections")
                )
                election_gp_contributions.append(election_contrib)

            # Sum the contributions from different election timescales
            election_party_time_effect = pm.Deterministic(
                "election_party_time_effect",
                sum(election_gp_contributions),
                dims=("countdown", "parties_complete", "elections")
            )

            # Weights for election-specific component
            lsd_party_effect_election = pm.ZeroSumNormal(
                "lsd_party_effect_election_party_amplitude",
                sigma=0.2,
                dims="parties_complete"
            )
            lsd_election_effect = pm.ZeroSumNormal(
                "lsd_election_effect",
                sigma=0.2,
                dims="elections"
            )
            lsd_election_party_sd = pm.HalfNormal("lsd_election_party_sd", sigma=0.05)
            lsd_election_party_effect = pm.ZeroSumNormal(
                "lsd_election_party_effect",
                sigma=lsd_election_party_sd,
                dims=("parties_complete", "elections"),
                n_zerosum_axes=2
            )

            election_party_time_weight = pm.Deterministic(
                "election_party_time_weight",
                pt.exp(
                    lsd_party_effect_election[:, None]
                    + lsd_election_effect[None, :]
                    + lsd_election_party_effect
                ),
                dims=("parties_complete", "elections")
            )
            election_party_time_effect_weighted = pm.Deterministic(
                "election_party_time_effect_weighted",
                election_party_time_effect * election_party_time_weight[None, :, :],
                dims=("countdown", "parties_complete", "elections")
            )

            # --------------------------------------------------------
            #                        HOUSE EFFECTS & POLL BIAS
            # --------------------------------------------------------

            poll_bias = pm.ZeroSumNormal(
                "poll_bias",
                sigma=0.05,
                dims="parties_complete",
            )

            house_effects = pm.ZeroSumNormal(
                "house_effects",
                sigma=0.05,
                dims=("pollsters", "parties_complete"),
            )

            house_election_effects_sd = pm.HalfNormal(
                "house_election_effects_sd",
                0.1,
                dims=("pollsters", "parties_complete"),
            )
            house_election_effects_raw = pm.ZeroSumNormal(
                "house_election_effects_raw",
                dims=("pollsters", "parties_complete", "elections"),
            )
            house_election_effects = pm.Deterministic(
                "house_election_effects",
                house_election_effects_sd[..., None] * house_election_effects_raw,
                dims=("pollsters", "parties_complete", "elections"),
            )

            # --------------------------------------------------------
            #                      POLL RESULTS
            # --------------------------------------------------------

            # Compute latent_mu
            latent_mu = (
                party_baseline[None, :]
                + election_party_baseline[data_containers["election_idx"]]
                + party_time_effect_weighted[data_containers["countdown_idx"]]
                + election_party_time_effect_weighted[
                    data_containers["countdown_idx"], :, data_containers["election_idx"]
                ]
                + gdp_effect_polls * pt.expand_dims(data_containers["stdz_gdp"], axis=1)  # Apply GDP effect for polls
            )

            latent_mu = latent_mu + data_containers['non_competing_polls_additive'] 

            # Apply softmax over the correct axis
            pm.Deterministic(
                "latent_popularity",
                pt.special.softmax(latent_mu, axis=1),
                dims=("observations", "parties_complete"),
            )

            noisy_mu = (
                latent_mu
                + poll_bias[None, :]
                + house_effects[data_containers["pollster_idx"]]
                + house_election_effects[
                    data_containers["pollster_idx"], :, data_containers["election_idx"]
                ]
                * data_containers['non_competing_polls_multiplicative']
            )

            # Apply softmax over the correct axis
            noisy_popularity = pm.Deterministic(
                "noisy_popularity",
                pt.special.softmax(noisy_mu, axis=1),
                dims=("observations", "parties_complete"),
            )

            # The concentration parameter of a Dirichlet-Multinomial distribution
            concentration_polls = pm.InverseGamma(
                "concentration_polls", mu=1000, sigma=200
            )

            pm.DirichletMultinomial(
                "N_approve",
                a=concentration_polls * noisy_popularity,
                n=data_containers["observed_N"],
                observed=data_containers["observed_polls"],
                dims=("observations", "parties_complete"),
            )

            # --------------------------------------------------------
            #                    ELECTION RESULTS
            # --------------------------------------------------------

            # Compute latent_mu_t0
            latent_mu_t0 = (
                party_baseline[None, :]
                + election_party_baseline
                + party_time_effect_weighted[0]
                + election_party_time_effect_weighted[0].transpose((1, 0))
                + gdp_effect_elections * data_containers["election_gdp"][:, None]  # Apply GDP effect for elections
            )

            latent_mu_t0 = latent_mu_t0 + data_containers['non_competing_parties_results']

            # Apply softmax over the correct axis
            latent_pop_t0 = pm.Deterministic(
                "latent_pop_t0",
                pt.special.softmax(latent_mu_t0, axis=1),
                dims=("elections", "parties_complete"),
            )

            concentration_results = pm.InverseGamma(
                "concentration_results", mu=1000, sigma=200
            )

            pm.DirichletMultinomial(
                "R",
                a=concentration_results * latent_pop_t0[:-1],
                n=data_containers["results_N"],
                observed=data_containers["observed_results"],
                dims=("elections_observed", "parties_complete"),
            )

        return model

    def sample_all(
        self, *, model: pm.Model = None, var_names: List[str], **sampler_kwargs
    ):
        """

        Sample the model and return the trace.

        TODO: Add 3 distinct functions to sample from the prior predictive,
        posterior and posterior predictive distributions.

        Parameters
        ----------
        model : optional
            A model previously created using `self.build_model()`.
            Build a new model if None (default)
        var_names: List[str]
            Variables names passed to `pm.fast_sample_posterior_predictive`
        **sampler_kwargs : dict
            Additional arguments to `pm.sample`
        """
        if model is None:
            model = self.build_model()
            
            #model = self.build_simplified_model()

        with model:
            prior_checks = pm.sample_prior_predictive()
            trace = pm.sample(draws=5000, tune=3000,nuts_sampler='numpyro',return_inferencedata=True, target_accept = 0.995,  **sampler_kwargs)
            post_checks = pm.sample_posterior_predictive(
                trace, var_names=var_names
            )

        return prior_checks, trace, post_checks

    def forecast_election(self, idata: arviz.InferenceData) -> arviz.InferenceData:
        """
        Generate out-of-sample predictions for ``election_to_predict`` specified in ``__init__``.

        Parameters
        ----------
        idata: arviz.InferenceData
            Posterior trace generated by ``self.sample_all`` on the training dataset.
            The dataset used for predictions is generated automatically: one observation for each
            of the days in ``self.coords["countdown"]``. The corresponding values of predictors are
            handled automatically.
        """
        new_dates, oos_data = self._generate_oos_data(idata)
        #oos_data = self._join_with_continuous_predictors(oos_data)
        forecast_data_index = pd.DataFrame(
            data=0,  # just a placeholder
            index=pd.MultiIndex.from_frame(oos_data),
            columns=self.political_families,
        )
        forecast_data = forecast_data_index.reset_index()

        PREDICTION_COORDS = {"observations": new_dates}
        PREDICTION_DIMS = {
            "latent_popularity": ["observations", "parties_complete"],
            "noisy_popularity": ["observations", "parties_complete"],
            "N_approve": ["observations", "parties_complete"],
        }

        forecast_model = self.build_model(
            polls=forecast_data,
        )
        with forecast_model:
            ppc = pm.sample_posterior_predictive(
                idata,
                var_names=[
                    "party_baseline",
                    "latent_popularity",
                    "noisy_popularity",
                    "N_approve",
                    "latent_pop_t0",
                    "R",
                ],
            )
            ppc.assign_coords(coords=PREDICTION_COORDS)
            
        return ppc,PREDICTION_COORDS,PREDICTION_DIMS

    def _generate_oos_data(
                self, idata: arviz.InferenceData
    ) -> Tuple[pd.Index, pd.DataFrame]:

        countdown = idata.posterior["countdown"]
        elections = idata.posterior["elections"]

        estimated_days = np.tile(countdown[::-1], reps=len(elections))
        N_estimated_days = len(estimated_days)

        new_dates = [
            pd.date_range(
                periods=max(countdown.data) + 1,
                end=date,
                freq="D",
            ).to_series()
            for date in elections.data
        ]
        new_dates = pd.concat(new_dates).index

        oos_data = pd.DataFrame.from_dict(
            {
                "countdown": estimated_days,
                "election_date": np.repeat(
                    self.unique_elections, repeats=len(countdown)
                ),
                "pollster": np.random.choice(
                    self.unique_pollsters, size=N_estimated_days
                ),
                "sample_size": np.random.choice(
                    self.results_oos["sample_size"].values, size=N_estimated_days
                ),
            }
        )
        oos_data["date"] = new_dates

        return new_dates, oos_data.set_index("date")
    
    def prepare_observed_data(self):
        # This method prepares the observed poll results for posterior predictive checks
        observed_data = pd.DataFrame({
            'date': self.results['date'],
            'pollster': self.results['pollster'],
        })
        for party in self.political_families:
            observed_data[party] = self.results[party] / self.results['sample_size']
        return observed_data

    def posterior_predictive_check(self, posterior):
        """
        Perform posterior predictive checks.

        Parameters:
        -----------
        posterior : arviz.InferenceData
            The posterior samples containing posterior predictive data.

        Returns:
        --------
        ppc_results : dict
            A dictionary containing various posterior predictive check results.
        """
        # Build the model
        model = self.build_model(self.polls_train)

        
        ppc = posterior.posterior_predictive

        ppc_results = {}

        # Print available keys in ppc for debugging
        print("Available keys in posterior_predictive:", ppc.data_vars.keys())

        # Compare observed data to posterior predictive distribution
        for i, party in enumerate(self.political_families):
            observed_polls = self.polls_train[party].values / self.polls_train['sample_size'].values
            observed_results = self.results_mult[party].values / self.results_mult['sample_size'].values
            
            # Use 'N_approve' from posterior_predictive
            predicted = ppc['N_approve'].values[:, :, :, i] / self.polls_train['sample_size'].values

            # Calculate mean absolute error for polls
            mae_polls = np.mean(np.abs(observed_polls - predicted.mean(axis=(0, 1))))
            ppc_results[f'{party}_mae_polls'] = mae_polls

            # Calculate coverage of 95% credible interval for polls
            lower, upper = np.percentile(predicted, [2.5, 97.5], axis=(0, 1))
            coverage_polls = np.mean((observed_polls >= lower) & (observed_polls <= upper))
            ppc_results[f'{party}_coverage_polls'] = coverage_polls

            # Calculate mean absolute error for results
            mae_results = np.mean(np.abs(observed_results - predicted.mean(axis=(0, 1))[-len(observed_results):]))
            ppc_results[f'{party}_mae_results'] = mae_results

            # Plot observed vs. predicted for polls
            plt.figure(figsize=(10, 6))
            plt.scatter(observed_polls, predicted.mean(axis=(0, 1)), label='Polls', alpha=0.5)
            plt.scatter(observed_results, predicted.mean(axis=(0, 1))[-len(observed_results):], label='Results', marker='x', s=100)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('Observed')
            plt.ylabel('Predicted')
            plt.title(f'Observed vs. Predicted for {party}')
            plt.legend()
            plt.savefig(f'ppc_plot_{party}.png')
            plt.close()

        return ppc_results


import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax

colors = sns.color_palette(as_cmap=True)


def retrodictive_plot(
    trace: arviz.InferenceData,
    posterior,
    parties_complete: List[str],
    polls_train: pd.DataFrame,
    group: str = "posterior",
):
    if len(parties_complete) % 2 == 0:
        fig, axes = plt.subplots(
            len(parties_complete) // 2, 2, figsize=(12, 15), sharey=True
        )
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(
            len(parties_complete) // 2 + 1, 2, figsize=(12, 15), sharey=True
        )
        axes = axes.ravel()
        axes[-1].remove()

    N = trace.constant_data["observed_N"]
    if group == "posterior":
        pp = posterior.posterior_predictive
        POST_MEDIANS = pp["latent_popularity"].median(("chain", "draw"))
        STACKED_POP = pp["latent_popularity"].stack(sample=("chain", "draw"))

    elif group == "prior":
        prior = trace.prior
        pp = trace.prior_predictive
        POST_MEDIANS = prior["latent_popularity"].median(("chain", "draw"))
        STACKED_POP = prior["latent_popularity"].stack(sample=("chain", "draw"))

    POST_MEDIANS_MULT = (pp["N_approve"] / N).median(("chain", "draw"))
    HDI = arviz.hdi(pp)["N_approve"] / N
    SAMPLES = np.random.choice(range(len(STACKED_POP.sample)), size=1000)

    for i, p in enumerate(parties_complete):
        if group == "posterior":
            axes[i].plot(
                polls_train["date"],
                polls_train[p] / N,
                "o",
                color=colors[i],
                alpha=0.4,
            )
        for sample in SAMPLES:
            axes[i].plot(
                polls_train["date"],
                STACKED_POP.sel(parties_complete=p).isel(sample=sample),
                color=colors[i],
                alpha=0.05,
            )
        axes[i].fill_between(
            polls_train["date"],
            HDI.sel(parties_complete=p, hdi="lower"),
            HDI.sel(parties_complete=p, hdi="higher"),
            color=colors[i],
            alpha=0.4,
        )
        axes[i].plot(
            polls_train["date"],
            POST_MEDIANS_MULT.sel(parties_complete=p),
            color="black",
            ls="--",
            lw=3,
            label="Noisy Popularity",
        )
        axes[i].plot(
            polls_train["date"],
            POST_MEDIANS.sel(parties_complete=p),
            color="grey",
            lw=3,
            label="Latent Popularity",
        )
        axes[i].tick_params(axis="x", labelrotation=45, labelsize=10)
        axes[i].set(title=p.title())
        axes[i].legend(fontsize=9, ncol=2)
    plt.suptitle(f"{group.title()} Predictive Check", fontsize=16, fontweight="bold")

def predictive_plot(
    idata: arviz.InferenceData,
    parties_complete: List[str],
    election_date: str,
    polls_train: pd.DataFrame,
    polls_test: pd.DataFrame,
    results: pd.DataFrame = None,
    # test_cutoff: pd.Timedelta = None,
    hdi: bool = False,
):
    election_date = pd.to_datetime(election_date)
    # results = results[results.dateelection == election_date]
    new_dates = idata.constant_data["observations"].to_index()
    election_yar_dates = new_dates[new_dates.year == int(f"{election_date.year}")]
    #election_yar_dates = [pd.to_datetime(date) for date in election_yar_dates]
    #date_idx = new_dates[new_dates.year == int(f"{election_date.year}")]

    #predictions = idata.posterior_predictive.sel(observations =(f"{election_date.year}"))


    predictions=idata.posterior_predictive.sel(
        observations=idata.posterior_predictive.observations.isin(election_yar_dates)
    )
    
    # constant_data = idata.predictions_constant_data.sel(
    #     observations=new_dates[new_dates.year == int(f"{election_date.year}")]
    # )

    # if test_cutoff is None:
    #     test_cutoff = election_date - pd.Timedelta(2, "D")
    # else:
    #     test_cutoff = election_date - test_cutoff

    if len(parties_complete) % 2 == 0:
        fig, axes = plt.subplots(len(parties_complete) // 2, 2, figsize=(12, 15))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(len(parties_complete) // 2 + 1, 2, figsize=(12, 15))
        axes = axes.ravel()
        axes[-1].remove()

    # post_N = constant_data["observed_N"]
    POST_MEDIANS = predictions["latent_popularity"].median(("chain", "draw"))
    STACKED_POP = predictions["latent_popularity"].stack(sample=("chain", "draw"))
    HDI_POP_83 = arviz.hdi(predictions, hdi_prob=0.83)["latent_popularity"]
    SAMPLES = np.random.choice(range(len(STACKED_POP.sample)), size=1000)
    POST_MEDIANS_MULT = predictions["noisy_popularity"].median(("chain", "draw"))
    # HDI_MULT = arviz.hdi(predictions, hdi_prob=0.83)["N_approve"] / post_N


    for i, p in enumerate(parties_complete):
        # axes[i].fill_between(
        #     predictions["observations"],
        #     HDI_MULT.sel(parties_complete=p, hdi="lower"),
        #     HDI_MULT.sel(parties_complete=p, hdi="higher"),
        #     color=colors[i],
        #     alpha=0.2,
        #     label="5 in 6 chance Polls",
        # )
        if hdi:
            axes[i].fill_between(
                predictions["observations"],
                HDI_POP_83.sel(parties_complete=p, hdi="lower"),
                HDI_POP_83.sel(parties_complete=p, hdi="higher"),
                color=colors[i],
                alpha=0.5,
                label="5 in 6 chance",
            )
        else:
            for sample in SAMPLES:
                axes[i].plot(
                    predictions["observations"],
                    STACKED_POP.sel(parties_complete=p).isel(sample=sample),
                    color=colors[i],
                    alpha=0.05,
                )
        axes[i].plot(
            predictions["observations"],
            POST_MEDIANS.sel(parties_complete=p),
            lw=3,
            color="black",
            label="Latent Popularity",
        )
        axes[i].plot(
            predictions["observations"],
            POST_MEDIANS_MULT.sel(parties_complete=p),
            ls="--",
            color="grey",
            label="Noisy Popularity",
        )
        axes[i].plot(
            polls_train["date"],
            polls_train[p] / polls_train["sample_size"],
            "o",
            color="black",
            alpha=0.4,
            label="Observed polls",
        )
        if polls_test is not None and not polls_test.empty:
            axes[i].plot(
                polls_test["date"],
                polls_test[p] / polls_test["sample_size"],
                "x",
                color="black",
                alpha=0.4,
                label="Unobserved polls",
            )
        # axes[i].axvline(
        #     x=test_cutoff,
        #     ymin=-0.01,
        #     ymax=1.0,
        #     ls="--",
        #     c="k",
        #     alpha=0.6,
        #     label="Test cutoff",
        # )
        axes[i].axvline(
            x=election_date,
            ymin=-0.01,
            ymax=1.0,
            ls=":",
            c="k",
            alpha=0.6,
            label="Election Day",
        )
        # axes[i].axhline(
        #     y=(results[p] / 100).to_numpy(),
        #     xmin=-0.01,
        #     xmax=1.0,
        #     ls="-.",
        #     c="k",
        #     alpha=0.6,
        #     label="Result",
        # )
        axes[i].axhline(
            y=xr.apply_ufunc(softmax, predictions["party_baseline"].mean(("chain", "draw"))).sel(
                parties_complete=p
            ),
            xmin=-0.01,
            xmax=1.0,
            ls="-.",
            c=colors[i],
            label="Historical Average",
        )
        axes[i].tick_params(axis="x", labelrotation=45, labelsize=10)
        axes[i].set(title=p, ylim=(-0.01, 0.4))
        axes[i].legend(fontsize=9, ncol=3)
        