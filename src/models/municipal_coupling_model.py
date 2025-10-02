"""PyMC implementation of the national-municipal coupling model (issue #38)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt

pytensor.config.cxx = "/usr/bin/clang++"

from src.data.municipal_coupling import (
    MUNICIPAL_ELECTION_DATES,
    TARGET_PARTIES,
    MunicipalCouplingDataset,
    build_municipal_coupling_dataset,
)


@dataclass
class CouplingEvaluation:
    """Holds evaluation artefacts for a hold-out election."""

    election_year: int
    winner_accuracy: float
    mean_vote_share_mae: float
    predicted_vote_shares: pd.DataFrame
    observed_vote_shares: pd.DataFrame
    coupling_summary: pd.DataFrame


class MunicipalCouplingModel:
    """PyMC implementation of the municipal coupling model."""

    def __init__(
        self,
        dataset: MunicipalCouplingDataset,
        train_years: Sequence[int],
        test_year: Optional[int] = None,
        concentration_prior_rate: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.train_years = list(train_years)
        self.test_year = test_year
        self.parties: List[str] = list(TARGET_PARTIES)
        self.party_vote_columns = [f"votes_{party}" for party in self.parties]
        self.party_clr_columns = [f"clr_{party}" for party in self.parties]
        self.concentration_prior_rate = concentration_prior_rate
        self.local_inc_index = self.parties.index("LOCAL_INC")
        self.other_index = self.parties.index("OTHER")
        self.available_cols = [f"available_{party}" for party in self.parties]

        self.municipality_codes: List[str] = sorted(dataset.baseline_clr.index.tolist())
        self.municipality_to_idx: Dict[str, int] = {
            code: idx for idx, code in enumerate(self.municipality_codes)
        }
        self.election_to_idx: Dict[int, int] = {
            year: idx for idx, year in enumerate(self.train_years)
        }

        metadata_indexed = dataset.metadata.set_index("municipality_code")
        district_lookup = (
            metadata_indexed.reindex(self.municipality_codes)["district_name"].fillna("UNKNOWN")
        )
        district_offset_rows: List[np.ndarray] = []
        district_offset_df = dataset.district_offsets
        if district_offset_df.empty:
            district_offset_rows = [
                np.zeros(len(self.parties), dtype=float) for _ in range(len(district_lookup))
            ]
        else:
            district_offset_df = district_offset_df.reindex(columns=self.parties).fillna(0.0)
            for district in district_lookup:
                if district in district_offset_df.index:
                    district_offset_rows.append(
                        district_offset_df.loc[district].to_numpy(dtype=float)
                    )
                else:
                    district_offset_rows.append(np.zeros(len(self.parties), dtype=float))
        self.district_offset_matrix = np.vstack(district_offset_rows)

        self.training_data = self._prepare_training_frame()
        self.baseline_matrix = (
            dataset.baseline_clr.loc[self.municipality_codes, self.party_clr_columns].to_numpy()
        )
        self.national_matrix = self._prepare_national_matrix(self.train_years)

        exp_baseline = np.exp(self.baseline_matrix)
        self.baseline_probs = exp_baseline / exp_baseline.sum(axis=1, keepdims=True)

        donor_df = dataset.donor_weights.reindex(self.municipality_codes).fillna(0.0)
        donor_df = donor_df.reindex(columns=self.parties, fill_value=0.0)
        self.donor_matrix = donor_df.to_numpy(dtype=float)

        availability_obs_matrix = self.training_data[self.available_cols].to_numpy(dtype=bool)
        self.historical_conditional_obs = self._build_conditional_history(
            municipality_indices=self.training_data["municipality_idx"].to_numpy(dtype=int),
            availability=availability_obs_matrix,
        )


        self.availability_observations = availability_obs_matrix
        if "new_LOCAL_INC" in self.training_data.columns:
            self.new_local_obs = self.training_data["new_LOCAL_INC"].to_numpy(dtype=float)
        else:
            self.new_local_obs = np.zeros(len(self.training_data), dtype=float)
        if "new_OTHER" in self.training_data.columns:
            self.new_other_obs = self.training_data["new_OTHER"].to_numpy(dtype=float)
        else:
            self.new_other_obs = np.zeros(len(self.training_data), dtype=float)
        if "local_list_flag" in self.training_data.columns:
            self.local_indicator_obs = self.training_data["local_list_flag"].to_numpy(dtype=float)
        else:
            self.local_indicator_obs = np.zeros(len(self.training_data), dtype=float)

        self.incumbent_cols = [f"incumbent_{party}" for party in self.parties]
        if set(self.incumbent_cols).issubset(self.training_data.columns):
            self.incumbent_observations = (
                self.training_data[self.incumbent_cols].to_numpy(dtype=float)
            )
        else:
            self.incumbent_observations = np.zeros(
                (len(self.training_data), len(self.parties)), dtype=float
            )

        if "incumbent_local_flag" in self.training_data.columns:
            self.incumbent_local_obs = (
                self.training_data["incumbent_local_flag"].to_numpy(dtype=float)
            )
        else:
            self.incumbent_local_obs = np.zeros(len(self.training_data), dtype=float)

        if "incumbent_prev_share" in self.training_data.columns:
            self.incumbent_prev_share_obs = (
                self.training_data["incumbent_prev_share"].fillna(0.0).to_numpy(dtype=float)
            )
        else:
            self.incumbent_prev_share_obs = np.zeros(len(self.training_data), dtype=float)

        target_year = max(self.dataset.results["election_year"].unique())
        target_frame = (
            self.dataset.results[self.dataset.results["election_year"] == target_year]
            .set_index("municipality_code")
        )
        self.availability_target = (
            target_frame.reindex(self.municipality_codes)[self.available_cols]
            .fillna(0)
            .to_numpy(dtype=bool)
        )
        if "new_LOCAL_INC" in target_frame.columns:
            self.new_local_target = (
                target_frame.reindex(self.municipality_codes)["new_LOCAL_INC"].fillna(0).to_numpy(dtype=float)
            )
        else:
            self.new_local_target = np.zeros(len(self.municipality_codes), dtype=float)
        if "new_OTHER" in target_frame.columns:
            self.new_other_target = (
                target_frame.reindex(self.municipality_codes)["new_OTHER"].fillna(0).to_numpy(dtype=float)
            )
        else:
            self.new_other_target = np.zeros(len(self.municipality_codes), dtype=float)
        if "local_list_flag" in target_frame.columns:
            self.local_indicator_target = (
                target_frame.reindex(self.municipality_codes)["local_list_flag"].fillna(0).to_numpy(dtype=float)
            )
        else:
            self.local_indicator_target = np.zeros(len(self.municipality_codes), dtype=float)

        if set(self.incumbent_cols).issubset(target_frame.columns):
            self.incumbent_indicator_target = (
                target_frame.reindex(self.municipality_codes)[self.incumbent_cols]
                .fillna(0)
                .to_numpy(dtype=float)
            )
        else:
            self.incumbent_indicator_target = np.zeros(
                (len(self.municipality_codes), len(self.parties)), dtype=float
            )

        if "incumbent_local_flag" in target_frame.columns:
            self.incumbent_local_target = (
                target_frame.reindex(self.municipality_codes)["incumbent_local_flag"]
                .fillna(0)
                .to_numpy(dtype=float)
            )
        else:
            self.incumbent_local_target = np.zeros(len(self.municipality_codes), dtype=float)

        if "incumbent_prev_share" in target_frame.columns:
            self.incumbent_prev_share_target = (
                target_frame.reindex(self.municipality_codes)["incumbent_prev_share"]
                .fillna(0)
                .to_numpy(dtype=float)
            )
        else:
            self.incumbent_prev_share_target = np.zeros(len(self.municipality_codes), dtype=float)

        self.historical_conditional_target = self._conditional_clr(
            self.baseline_probs,
            self.availability_target.astype(bool),
            self.donor_matrix,
        )

        self.coords = {
            "municipalities": self.municipality_codes,
            "parties": self.parties,
            "observations": np.arange(len(self.training_data)),
            "elections": self.train_years,
        }

        self.model: Optional[pm.Model] = None
        self.idata: Optional[az.InferenceData] = None

    def _prepare_training_frame(self) -> pd.DataFrame:
        train_results = self.dataset.results[
            self.dataset.results["election_year"].isin(self.train_years)
        ].copy()
        train_results["municipality_idx"] = train_results["municipality_code"].map(
            self.municipality_to_idx
        )
        train_results["election_idx"] = train_results["election_year"].map(
            self.election_to_idx
        )
        train_results = train_results.sort_values([
            "election_year",
            "municipality_idx",
        ]).reset_index(drop=True)
        return train_results

    def _prepare_national_matrix(self, years: Sequence[int]) -> np.ndarray:
        national = (
            self.dataset.national_clr.set_index("election_year")
            .loc[list(years), self.party_clr_columns]
            .to_numpy()
        )
        return national

    def _condition_clr_row(
        self,
        probs: np.ndarray,
        mask: np.ndarray,
        donors: np.ndarray,
    ) -> np.ndarray:
        """Condition a single municipality-level probability vector."""

        mask = mask.astype(bool)
        if not mask.any():
            mask = np.ones_like(mask, dtype=bool)

        adjusted = probs.astype(float).copy()
        missing = ~mask
        small = 1e-6

        if self.local_inc_index < len(adjusted) and mask[self.local_inc_index]:
            donor_vec = donors.astype(float).copy()
            donor_vec[~mask] = 0.0
            donor_vec[self.local_inc_index] = 0.0
            donor_vec = np.clip(donor_vec, 0.0, None)
            donor_total = donor_vec.sum()
            if donor_total > small:
                donor_vec /= donor_total
                redistribution = adjusted * donor_vec
                adjusted -= redistribution
                adjusted[self.local_inc_index] += redistribution.sum()

        for party_idx in np.where(missing)[0]:
            mass = adjusted[party_idx]
            if mass <= 0.0:
                adjusted[party_idx] = 0.0
                continue
            adjusted[party_idx] = 0.0
            donor_vec = donors.astype(float).copy()
            donor_vec[party_idx] = 0.0
            donor_vec[missing] = 0.0
            donor_vec = np.clip(donor_vec, 0.0, None)
            donor_total = donor_vec.sum()
            if donor_total <= small:
                donor_vec = mask.astype(float)
                donor_vec[party_idx] = 0.0
                donor_total = donor_vec.sum()
            if donor_total > small:
                donor_vec /= donor_total
                adjusted += mass * donor_vec

        subset = adjusted[mask]
        subset = np.clip(subset, small, None)
        subset = subset / subset.sum()
        log_subset = np.log(subset)
        clr_subset = log_subset - log_subset.mean()

        clr_row = np.full_like(probs, fill_value=-50.0)
        clr_row[mask] = clr_subset
        return clr_row

    def _conditional_clr(
        self,
        probs_matrix: np.ndarray,
        availability_mask: np.ndarray,
        donor_matrix: np.ndarray,
    ) -> np.ndarray:
        """Return CLR vectors conditioned on the available parties."""

        conditioned = np.full_like(probs_matrix, fill_value=-50.0)
        for idx in range(probs_matrix.shape[0]):
            conditioned[idx] = self._condition_clr_row(
                probs_matrix[idx], availability_mask[idx], donor_matrix[idx]
            )
        return conditioned

    def _build_conditional_history(
        self,
        municipality_indices: np.ndarray,
        availability: np.ndarray,
    ) -> np.ndarray:
        conditioned = np.full_like(availability, fill_value=-50.0, dtype=float)
        for obs_idx, (mun_idx, mask_row) in enumerate(zip(municipality_indices, availability)):
            conditioned[obs_idx] = self._condition_clr_row(
                self.baseline_probs[mun_idx],
                mask_row,
                self.donor_matrix[mun_idx],
            )
        return conditioned

    def build_model(self) -> pm.Model:
        if self.model is not None:
            return self.model

        obs_counts = self.training_data[self.party_vote_columns].to_numpy(dtype=float)
        total_votes = self.training_data["total_votes"].to_numpy(dtype=float)
        municipality_idx = self.training_data["municipality_idx"].to_numpy(dtype="int64")
        election_idx = self.training_data["election_idx"].to_numpy(dtype="int64")

        with pm.Model(coords=self.coords) as model:
            municipality_idx_data = pm.Data("municipality_idx", municipality_idx, dims="observations")
            election_idx_data = pm.Data("election_idx", election_idx, dims="observations")
            historical = pm.Data(
                "historical_clr",
                self.baseline_matrix,
                dims=("municipalities", "parties"),
            )
            historical_conditional = pm.Data(
                "historical_conditional",
                self.historical_conditional_obs,
                dims=("observations", "parties"),
            )
            district_offsets = pm.Data(
                "district_offsets",
                self.district_offset_matrix,
                dims=("municipalities", "parties"),
            )
            new_local_data = pm.Data(
                "new_local_obs",
                self.new_local_obs,
                dims="observations",
            )
            new_other_data = pm.Data(
                "new_other_obs",
                self.new_other_obs,
                dims="observations",
            )
            national = pm.Data(
                "national_clr",
                self.national_matrix,
                dims=("elections", "parties"),
            )
            counts_data = pm.Data(
                "observed_counts",
                obs_counts,
                dims=("observations", "parties"),
            )
            total_votes_data = pm.Data(
                "total_votes",
                total_votes,
                dims="observations",
            )
            availability_data = pm.Data(
                "availability_obs",
                self.availability_observations,
                dims=("observations", "parties"),
            )
            local_indicator_data = pm.Data(
                "local_indicator_obs",
                self.local_indicator_obs,
                dims="observations",
            )
            incumbent_indicator_data = pm.Data(
                "incumbent_indicator_obs",
                self.incumbent_observations,
                dims=("observations", "parties"),
            )
            incumbent_local_data = pm.Data(
                "incumbent_local_obs",
                self.incumbent_local_obs,
                dims="observations",
            )
            incumbent_prev_share_data = pm.Data(
                "incumbent_prev_share_obs",
                self.incumbent_prev_share_obs,
                dims="observations",
            )
            coupling = pm.Beta("coupling", alpha=2.0, beta=2.0, dims="municipalities")
            new_local_coupling_scale = pm.Beta("new_local_coupling_scale", alpha=2.0, beta=2.0)
            new_other_coupling_scale = pm.Beta("new_other_coupling_scale", alpha=2.0, beta=2.0)
            new_local_concentration = pm.HalfNormal("new_local_concentration", sigma=1.0)
            new_other_concentration = pm.HalfNormal("new_other_concentration", sigma=1.0)
            local_list_effect = pm.Normal("local_list_effect", mu=2.5, sigma=0.8, dims="municipalities")
            incumbent_effect = pm.Normal("incumbent_effect", mu=0.5, sigma=0.5, dims="parties")
            incumbent_local_effect = pm.Normal("incumbent_local_effect", mu=2.0, sigma=0.5)
            incumbent_nonlocal_penalty = pm.HalfNormal("incumbent_nonlocal_penalty", sigma=3.0)
            local_incumbent_prev_weight = pm.LogNormal(
                "local_incumbent_prev_weight", mu=np.log(5.0), sigma=0.7
            )
            concentration = pm.Exponential(
                "concentration", lam=self.concentration_prior_rate
            )

            coupling_raw = coupling[municipality_idx_data]
            reduction = (
                new_local_coupling_scale * new_local_data
                + new_other_coupling_scale * new_other_data
            )
            coupling_obs = pm.math.clip(coupling_raw * (1.0 - reduction), 0.0, 1.0)

            effective_concentration = concentration / (
                1.0
                + new_local_concentration * new_local_data
                + new_other_concentration * new_other_data
            )

            linear_pred = (
                coupling_obs[:, None] * national[election_idx_data]
                + (1.0 - coupling_obs[:, None]) * historical_conditional
            )
            linear_pred = linear_pred + district_offsets[municipality_idx_data]
            linear_pred = pt.switch(availability_data, linear_pred, linear_pred - 50.0)
            linear_pred = pt.set_subtensor(
                linear_pred[:, self.local_inc_index],
                linear_pred[:, self.local_inc_index]
                + local_list_effect[municipality_idx_data] * local_indicator_data,
            )
            linear_pred = pt.set_subtensor(
                linear_pred[:, self.local_inc_index],
                linear_pred[:, self.local_inc_index]
                + incumbent_local_effect * incumbent_local_data,
            )
            prev_share_clipped = pm.math.clip(incumbent_prev_share_data, 1e-6, 1.0 - 1e-6)
            prev_share_logit = pt.log(prev_share_clipped) - pt.log1p(-prev_share_clipped)
            linear_pred = pt.set_subtensor(
                linear_pred[:, self.local_inc_index],
                linear_pred[:, self.local_inc_index]
                + local_incumbent_prev_weight
                * prev_share_logit
                * incumbent_local_data,
            )
            linear_pred = linear_pred + incumbent_indicator_data * incumbent_effect[None, :]
            linear_pred = linear_pred - incumbent_nonlocal_penalty * (
                incumbent_local_data[:, None] * (1.0 - incumbent_indicator_data)
            )
            vote_shares_raw = pm.math.softmax(linear_pred, axis=1)
            vote_shares = pm.Deterministic(
                "vote_shares",
                vote_shares_raw / vote_shares_raw.sum(axis=1, keepdims=True),
            )
            vote_concentration = pm.Deterministic(
                "dirichlet_alpha",
                pm.math.clip(vote_shares * effective_concentration[:, None], 1e-6, np.inf),
            )

            alpha_sum = pm.math.sum(vote_concentration, axis=1)
            logp_terms = (
                pt.gammaln(total_votes_data + 1.0)
                + pt.gammaln(alpha_sum)
                - pt.gammaln(total_votes_data + alpha_sum)
                + pm.math.sum(pt.gammaln(counts_data + vote_concentration), axis=1)
                - pm.math.sum(pt.gammaln(counts_data + 1.0), axis=1)
                - pm.math.sum(pt.gammaln(vote_concentration), axis=1)
            )

            pm.Potential("likelihood", pm.math.sum(logp_terms))

        self.model = model
        return model

    def fit(
        self,
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
    ) -> az.InferenceData:
        model = self.build_model()
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=random_seed,
            )
        self.idata = idata
        return idata

    # ------------------------------------------------------------------
    # Evaluation utilities
    # ------------------------------------------------------------------

    def predict_vote_shares(
        self,
        idata: az.InferenceData,
        election_year: int,
    ) -> np.ndarray:
        """Return posterior samples of vote shares for the requested election."""
        if election_year not in MUNICIPAL_ELECTION_DATES:
            raise ValueError(f"Unknown municipal election date for year {election_year}")

        coupling_samples = (
            idata.posterior["coupling"].stack(sample=("chain", "draw")).transpose("sample", "municipalities")
        )
        coupling_values = coupling_samples.to_numpy()

        new_local_coupling_scale = (
            idata.posterior["new_local_coupling_scale"].stack(sample=("chain", "draw"))
        ).to_numpy()
        new_other_coupling_scale = (
            idata.posterior["new_other_coupling_scale"].stack(sample=("chain", "draw"))
        ).to_numpy()
        new_local_concentration = (
            idata.posterior["new_local_concentration"].stack(sample=("chain", "draw"))
        ).to_numpy()
        new_other_concentration = (
            idata.posterior["new_other_concentration"].stack(sample=("chain", "draw"))
        ).to_numpy()

        local_effect_samples = (
            idata.posterior["local_list_effect"].stack(sample=("chain", "draw")).transpose("sample", "municipalities")
        )
        local_effect_values = local_effect_samples.to_numpy()

        incumbent_effect_samples = (
            idata.posterior["incumbent_effect"].stack(sample=("chain", "draw")).transpose("sample", "parties")
        )
        incumbent_effect_values = incumbent_effect_samples.to_numpy()

        incumbent_local_samples = (
            idata.posterior["incumbent_local_effect"].stack(sample=("chain", "draw"))
        )
        incumbent_local_values = incumbent_local_samples.to_numpy()

        nonlocal_penalty_samples = (
            idata.posterior["incumbent_nonlocal_penalty"].stack(sample=("chain", "draw"))
        )
        nonlocal_penalty_values = nonlocal_penalty_samples.to_numpy()

        local_inc_prev_weight_samples = (
            idata.posterior["local_incumbent_prev_weight"].stack(
                sample=("chain", "draw")
            )
        )
        local_inc_prev_weight_values = local_inc_prev_weight_samples.to_numpy()

        national_vector = (
            self.dataset.national_clr.set_index("election_year")
            .loc[election_year, self.party_clr_columns]
            .to_numpy()
        )
        historical_matrix = self.historical_conditional_target
        district_offsets = self.district_offset_matrix

        reduction = (
            new_local_coupling_scale[:, None] * self.new_local_target[None, :]
            + new_other_coupling_scale[:, None] * self.new_other_target[None, :]
        )
        coupling_adjusted = np.clip(coupling_values * (1.0 - reduction), 0.0, 1.0)

        linear = (
            coupling_adjusted[:, :, None] * national_vector[None, None, :]
            + (1.0 - coupling_adjusted[:, :, None]) * historical_matrix[None, :, :]
        )
        linear = linear + district_offsets[None, :, :]
        availability_target = self.availability_target.astype(float)
        linear -= 50.0 * (1.0 - availability_target[None, :, :])
        local_indicator_target = self.local_indicator_target.astype(float)
        inc_local = self.incumbent_local_target.astype(float)
        linear[:, :, self.local_inc_index] += (
            local_effect_values * local_indicator_target[None, :]
        )
        linear[:, :, self.local_inc_index] += (
            incumbent_local_values[:, None] * inc_local[None, :]
        )
        prev_share_target = np.clip(self.incumbent_prev_share_target, 1e-6, 1.0 - 1e-6)
        prev_share_logit_target = np.log(prev_share_target) - np.log1p(-prev_share_target)
        linear[:, :, self.local_inc_index] += (
            local_inc_prev_weight_values[:, None]
            * prev_share_logit_target[None, :]
            * inc_local[None, :]
        )
        inc_indicator = self.incumbent_indicator_target.astype(float)
        linear += incumbent_effect_values[:, None, :] * inc_indicator[None, :, :]
        linear -= nonlocal_penalty_values[:, None, None] * (
            (1.0 - inc_indicator)[None, :, :] * inc_local[None, :, None]
        )
        linear = linear - linear.max(axis=2, keepdims=True)
        exp_linear = np.exp(linear)
        shares = exp_linear / exp_linear.sum(axis=2, keepdims=True)
        return shares

    def evaluate_holdout(self, idata: az.InferenceData, election_year: int) -> CouplingEvaluation:
        shares = self.predict_vote_shares(idata, election_year)
        mean_shares = shares.mean(axis=0)

        holdout = self.dataset.results[
            self.dataset.results["election_year"] == election_year
        ].copy()
        holdout = holdout.set_index("municipality_code").loc[self.municipality_codes]

        observed_counts = holdout[self.party_vote_columns].to_numpy()
        observed_totals = holdout["total_votes"].to_numpy()[:, None]
        observed_shares = observed_counts / observed_totals

        mae = np.abs(mean_shares - observed_shares).mean()
        predicted_winners = mean_shares.argmax(axis=1)
        observed_winners = observed_shares.argmax(axis=1)
        winner_accuracy = float(np.mean(predicted_winners == observed_winners))

        prediction_df = pd.DataFrame(mean_shares, columns=self.parties)
        prediction_df.insert(0, "municipality_code", self.municipality_codes)
        prediction_df = prediction_df.merge(
            self.dataset.metadata,
            on="municipality_code",
            how="left",
        )
        if {"new_LOCAL_INC", "new_OTHER"}.issubset(holdout.columns):
            new_flags = holdout[["new_LOCAL_INC", "new_OTHER"]].reset_index()[
                ["municipality_code", "new_LOCAL_INC", "new_OTHER"]
            ]
            prediction_df = prediction_df.merge(new_flags, on="municipality_code", how="left")

        observed_df = pd.DataFrame(observed_shares, columns=self.parties)
        observed_df.insert(0, "municipality_code", self.municipality_codes)
        observed_df = observed_df.merge(
            self.dataset.metadata,
            on="municipality_code",
            how="left",
        )
        if {"new_LOCAL_INC", "new_OTHER"}.issubset(holdout.columns):
            observed_df = observed_df.merge(new_flags, on="municipality_code", how="left")

        coupling_summary = self._summarize_coupling(idata)

        return CouplingEvaluation(
            election_year=election_year,
            winner_accuracy=winner_accuracy,
            mean_vote_share_mae=mae,
            predicted_vote_shares=prediction_df,
            observed_vote_shares=observed_df,
            coupling_summary=coupling_summary,
        )

    def _summarize_coupling(self, idata: az.InferenceData) -> pd.DataFrame:
        coupling_da = idata.posterior["coupling"]
        mean_series = coupling_da.mean(dim=("chain", "draw")).to_series()
        summary = mean_series.rename("coupling_mean").reset_index()
        summary = summary.rename(columns={"municipalities": "municipality_code"})

        hdi = az.hdi(coupling_da, hdi_prob=0.9)
        lower_series = hdi.sel(hdi="lower").to_array().to_series().reset_index(level=0, drop=True)
        upper_series = hdi.sel(hdi="higher").to_array().to_series().reset_index(level=0, drop=True)
        summary["hdi_lower"] = lower_series.values
        summary["hdi_upper"] = upper_series.values

        enriched = summary.merge(
            self.dataset.metadata,
            on="municipality_code",
            how="left",
        )
        return enriched.sort_values("coupling_mean", ascending=False)


def train_coupling_model(
    trace_path: str,
    election_years: Sequence[int],
    train_years: Sequence[int],
    output_dir: Path,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: Optional[int] = None,
) -> Tuple[MunicipalCouplingModel, az.InferenceData, CouplingEvaluation]:
    dataset = build_municipal_coupling_dataset(election_years, trace_path, train_years)
    model = MunicipalCouplingModel(dataset, train_years=train_years, test_year=max(election_years))
    idata = model.fit(draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed)

    holdout_year = max(election_years)
    evaluation = model.evaluate_holdout(idata, holdout_year)

    output_dir.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, output_dir / "posterior.nc")
    evaluation.predicted_vote_shares.to_csv(output_dir / "predictions.csv", index=False)
    evaluation.observed_vote_shares.to_csv(output_dir / "observed.csv", index=False)
    evaluation.coupling_summary.to_csv(output_dir / "coupling_summary.csv", index=False)
    new_flags = dataset.results[dataset.results["election_year"] == holdout_year][
        [
            "municipality_code",
            "municipality_name",
            "new_LOCAL_INC",
            "new_OTHER",
            "local_list_flag",
            "incumbent_local_flag",
            "incumbent_prev_share",
        ]
    ]
    new_flags.to_csv(output_dir / "new_party_flags.csv", index=False)

    metrics = {
        "holdout_year": holdout_year,
        "winner_accuracy": evaluation.winner_accuracy,
        "mean_vote_share_mae": evaluation.mean_vote_share_mae,
        "draws": draws,
        "tune": tune,
    }
    pd.Series(metrics).to_json(output_dir / "metrics.json")

    return model, idata, evaluation
