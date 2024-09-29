import os

import arviz
import requests
os.environ["PYTENSOR_FLAGS"] = "optimizer_excluding=local_IncSubtensor_serialize"

from election_model import ElectionsModel
from analysis import run_diagnostics
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
    arviz.to_zarr(prior, "prior.zarr")
    arviz.to_zarr(trace, "trace.zarr")
    arviz.to_zarr(post, "posterior.zarr")
    requests.post("https://ntfy.sh/bc-estimador",
        data="Finished sampling".encode(encoding='utf-8'))

    model = ElectionsModel('2024-03-10')
    # Extract necessary data from idata
    polls_train = model.polls_train
    polls_test = model.polls_test
    results_mult = model.results_mult
    election_dates = model.election_dates
    government_parties = model.government_parties

    # Note: The model object is not typically stored in zarr files.
    # If you need the model, you may need to recreate it or load it separately.
     # Adjust the date as needed

    try:
        run_diagnostics(trace, prior, post, model, polls_train, polls_test, results_mult, election_dates, government_parties)
        requests.post("https://ntfy.sh/bc-estimador",
            data="Finished analysis".encode(encoding='utf-8'))
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        requests.post("https://ntfy.sh/bc-estimador",
            data=f"Error running diagnostics: {e}".encode(encoding='utf-8'))