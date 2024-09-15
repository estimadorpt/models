import os

import arviz
os.environ["PYTENSOR_FLAGS"] = "optimizer_excluding=local_IncSubtensor_serialize"

from election_model import ElectionsModel

if __name__ == "__main__":
    b = ElectionsModel('2024-03-10')
    prior,trace,post = b.sample_all(
    var_names=[
        "latent_popularity",
        "latent_pop_t0",
       "R",
        "noisy_popularity",
       "N_approve",
    ],)
    arviz.to_netcdf(prior, "prior.nc")
    arviz.to_netcdf(trace, "trace.nc")
    arviz.to_netcdf(post, "posterior.nc")