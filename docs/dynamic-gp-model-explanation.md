# How Our Dynamic Gaussian Process Election Model Works

We've developed a statistical forecasting model for Portuguese elections that captures both long-term political trends and district-level dynamics. It represents an evolution of traditional election models, designed specifically to address the challenges of Portugal's multi-party, district-based electoral system. This document explains the approach, intuition, and technical details behind our methodology.

## The Challenge of Forecasting Portuguese Elections

Portugal's electoral landscape presents unique forecasting challenges. The country has multiple significant political parties ranging from the traditional center-left (PS) and center-right (AD) to newer entrants like the liberal IL and right-wing Chega. Elections are decided through a proportional representation system across multiple districts, with seats allocated using the D'Hondt method.

Traditional forecasting approaches struggle with several aspects of this system:

1. District-level variation in political support that doesn't uniformly follow national trends
2. Different parties having varying sensitivity to national sentiment shifts
3. Campaign dynamics that can shift rapidly during election season
4. Pollster-specific biases that need correction

Our dynamic Gaussian process model addresses these challenges through a principled Bayesian statistical framework.

## The Intuition Behind the Model

Our model works by breaking down party support into several components:

1. **Long-term national trends**: The baseline support for each party over extended periods
2. **Short-term fluctuations**: Changes in support during campaign periods
3. **District-specific patterns**: How each district deviates from national trends
4. **Pollster effects**: Systematic biases in how polling firms measure support
5. **Uncertainty**: The probabilistic range of possible outcomes

The "dynamic" in our model name refers to its ability to capture changing support patterns over time, while "GP" (Gaussian Process) refers to the statistical technique that allows us to model smoothly varying support without imposing rigid assumptions about how it changes.

Unlike simpler approaches that rely on uniform national swing (where all districts are assumed to shift by the same amount), our model allows for differential shifts. When national support for party A increases by 5 percentage points, some districts might shift by 8 points, while others barely move. The model learns these patterns from historical data.

## How the Model Functions

Imagine a party's support level as an invisible line that evolves continuously over time. We never observe this line directly—we only see noisy snapshots from polls or election results. Our model reconstructs the most likely trajectory of this line by combining:

- Prior knowledge about typical political changes
- Information from polls
- Historical election results
- District-specific patterns
- Knowledge about how pollsters tend to measure

When the model forecasts an upcoming election, it projects these components forward in time and aggregates them into probabilistic vote shares for each party in each district. These are then translated into seat allocations using the D'Hondt method.

## Technical Components of the Model

Let's examine each major component of the model:

### 1. Baseline GP over Calendar Time

The foundation of our model is a long-term trajectory for each party's support, modeled using a Gaussian Process. This captures gradual shifts in the political landscape over time.

We use a covariance kernel with a lengthscale parameter of roughly 2.5 years, meaning that support levels separated by this much time have a correlation of approximately 0.2. This captures the intuition that political support evolves gradually, with changes spanning multiple years.

The amplitude parameter determines how much variation we expect to see in this long-term trend. Together, these parameters define how the model understands the smooth, long-term evolution of party support.

### 2. Short-Term GP for Campaign Dynamics

Elections often see rapid shifts during campaign periods. We model this with a separate short-term GP that has a shorter lengthscale.

With a typical lengthscale of 30-45 days, this component captures rapid changes in support that occur during campaign periods, such as responses to debates, scandals, or policy announcements. The amplitude parameter controls the magnitude of these fluctuations.

These two GPs work together to create a flexible model of temporal dynamics: the baseline captures long-term shifts in the electorate, while the short-term component captures the more volatile campaign period.

### 3. District-Level Effects

One of the most important innovations in our model is how it handles district-level variation. Rather than assuming uniform change across all districts, we model each district as having:

1. A **base offset** that represents its persistent deviation from the national average
2. A **sensitivity parameter (beta)** that determines how strongly the district responds to national trends

When beta equals 1, the district follows the national trend exactly. When beta is greater than 1, the district amplifies national changes; when less than 1, it dampens them.

This formulation allows us to capture how, for example, rural districts might respond more strongly to changes in support for conservative parties, while urban districts might be more responsive to shifts among progressive parties.

### 4. House Effects (Pollster Biases)

Different polling firms often show systematic biases. Some consistently overestimate certain parties, while others underestimate them. Our model explicitly accounts for these "house effects."

These effects are constrained to sum to zero across parties for each pollster, meaning they represent relative biases rather than absolute shifts in the total. The magnitude of these effects is controlled by a standard deviation parameter, which can vary by party to reflect that some parties may be harder to poll accurately than others.

### 5. Softmax Transformation and Likelihood

All these components combine to form a latent score for each party. We then apply a softmax transformation to ensure the vote shares sum to 1, creating the national latent popularity.

Finally, we model the observed poll and election result counts using Dirichlet-Multinomial distributions. This likelihood function connects our latent support models to the actual observed data, accounting for the additional variation in poll results beyond simple sampling error through concentration parameters.

## Statistical Methodology

The statistical machinery behind our model relies on several advanced techniques:

### Gaussian Processes and HSGP Approximation

Gaussian Processes provide a flexible way to model functions over time without specifying a particular parametric form. However, exact GPs become computationally infeasible for large datasets. We use the Hilbert Space Gaussian Process (HSGP) approximation, which represents the GP using a finite set of basis functions.

This approach dramatically improves computational efficiency while maintaining excellent approximation quality. The model uses coefficients and basis functions to represent both the long-term and short-term temporal patterns.

### Non-centered Parameterization

To improve sampling efficiency, we use non-centered parameterizations for hierarchical parameters like house effects. Instead of directly sampling the house effects, we sample standardized values and then multiply by a scale parameter. This approach helps the MCMC sampler navigate the posterior distribution more efficiently.

### Concentration Parameters

The Dirichlet-Multinomial likelihood includes concentration parameters that capture extra-multinomial variation. Higher concentration values indicate polls that closely follow the expected multinomial distribution, while lower values suggest additional sources of variation beyond sampling error.

## Making Predictions

Our model generates predictions in several steps:

1. For a given date (election day, last poll date, or current date), we extract the posterior distribution of national latent support.
2. We apply district-specific adjustments based on the base offset and beta sensitivity parameters.
3. We convert these to vote shares using the softmax function.
4. We simulate the D'Hondt seat allocation method thousands of times to generate a distribution of possible seat outcomes.

The result is not a single prediction but a probability distribution over possible outcomes, reflecting all sources of uncertainty in the model.

### District Vote Share Prediction

The core of our prediction methodology combines national trends with district-specific adjustments. We first extract the national trend at the target date and calculate how it deviates from the long-term average. We then apply district-specific adjustments by:

1. Calculating the dynamic adjustment as (beta - 1) multiplied by the national deviation
2. Adding the base offset to get the total district adjustment
3. Applying this adjustment to the national trend
4. Converting to vote shares using softmax

This approach allows districts to respond differently to national trends while maintaining a coherent overall structure.

### Seat Allocation

Once we have vote share predictions, we simulate the election outcome using the D'Hondt method, which allocates seats proportionally based on each party's votes. By running this simulation thousands of times across our posterior samples, we generate a probability distribution over possible seat outcomes for each party.



## Limitations and Future Improvements

Like all models, ours has limitations:

1. It assumes that historical patterns of district behavior will continue into the future.
2. It does not incorporate non-polling data such as economic indicators or government approval ratings.
3. The district effects model could potentially be enhanced with spatial correlation structure.

Future versions may address these limitations by incorporating additional data sources and more sophisticated spatial modeling techniques.
