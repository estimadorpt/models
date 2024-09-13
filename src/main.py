from election_model import ElectionsModel


if __name__ == "__main__":
    b = ElectionsModel('2024-03-10')
    model = b.build_model()

    prior_2,trace_2,post_2 = b.sample_all(model=model,
    var_names=[
   #     "latent_popularity",
  #      "latent_pop_t0",
       "R",
#        ""noisy_popularity",
       #"N_approve",
    ],
)