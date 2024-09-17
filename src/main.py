import os

import arviz
os.environ["PYTENSOR_FLAGS"] = "optimizer_excluding=local_IncSubtensor_serialize"

from election_model import ElectionsModel

var_names = [
        "party_baseline",
        "election_party_baseline",
        "poll_bias",
        "house_effects",
        "house_election_effects",
        "party_time_effect_weighted",
        "latent_popularity",
        "noisy_popularity",
        "election_party_time_effect_weighted",

        "N_approve",
        "R"
    ]

if __name__ == "__main__":
    b = ElectionsModel('2024-03-10')
    prior, trace, post = b.sample_all(var_names=var_names)
    arviz.to_netcdf(prior, "prior_simplified.nc")
    arviz.to_netcdf(trace, "trace_simplified.nc")
    arviz.to_netcdf(post, "posterior_simplified.nc")