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

Let's examine each major component of the model from a statistical perspective:

### 1. Baseline GP over Calendar Time

The foundation of the model is a smooth, long-term trend representing the baseline national support for each party. This is modeled as a Gaussian Process evolving over calendar time. The properties of this GP (specifically its covariance structure) are chosen to reflect assumptions about the slow, gradual nature of fundamental political shifts, capturing dependencies over multiple years.

### 2. Medium-Term GP over Calendar Time

Superimposed on the baseline trend is a second Gaussian Process, also evolving over calendar time. This component is designed with a moderate correlation length (e.g., centered around a year), allowing it to capture deviations from the long-term baseline over medium timescales. This could reflect evolving responses to specific events or other medium-term dynamics.

### 3. Very Short-Term GP over Calendar Time

A third Gaussian Process, again over calendar time, is added to capture even more rapid fluctuations. This GP has a very short correlation length (e.g., centered around a few weeks). It is designed to model fast-moving campaign dynamics, late shifts in public opinion, or reactions to breaking news closer to an election.

The sum of these three processes (baseline, medium-term, very short-term) forms the latent (unobserved) national support trajectory for each party.

### 4. District-Level Effects

To account for Portugal's district-based system, the model incorporates district-specific deviations from the national trend. In the current implementation, this is achieved solely through estimated **static base offset** parameters for each district and party. These parameters represent the persistent, time-invariant tendency for a district's support for a given party to be higher or lower than the national average, relative to the average trend. These offsets are learned primarily from historical election results at the district level. Unlike previous experimental versions, this model *currently uses only these static offsets*. It assumes that district deviations from the national trend do not dynamically change based on the magnitude of national swings within a single election cycle (i.e., the sensitivity or 'beta' component is not currently active).

### 5. House Effects (Pollster Biases) and Poll Bias

The model explicitly accounts for systematic variations between polling firms. These "house effects" are modeled as parameters specific to each pollster and party, constrained such that they represent relative deviations (i.e., summing to zero across parties for a given pollster). This captures the tendency of some pollsters to relatively overestimate or underestimate certain parties.

Additionally, an overall poll bias term, also constrained to sum to zero across parties, is included. This captures any average systematic deviation of poll results from the underlying national trend, distinct from individual pollster effects.

### 6. Latent Score, Transformation, and Likelihood

The national trend components (sum of the three calendar-time GPs) are combined with the relevant bias terms (house effects and poll bias for poll observations, or the static district base offsets for district predictions) to produce a latent score representing underlying support.

A softmax transformation converts these unbounded latent scores into a set of probabilities (vote shares) for each party that necessarily sum to one.

Finally, the observed data—vote counts from polls, district-level election results, **and national-level election results**—are linked to these modeled probabilities through a statistical likelihood function. The chosen likelihood (typically a Dirichlet-Multinomial distribution) is suitable for count data representing proportions and includes parameters to accommodate potential overdispersion (more variability than predicted by simpler models). The inclusion of both district and national results helps anchor the national trend prediction and inform the district offsets.

## Statistical Methodology

The inference and structure rely on key statistical concepts:

### Gaussian Processes

Gaussian Processes provide a flexible, non-parametric Bayesian approach to function estimation. Here, they are used to model the unobserved national support trends over time without imposing rigid functional forms. The choice of covariance kernel and its parameters (lengthscale, amplitude) encode prior beliefs about the smoothness and variability of these trends.

### Hierarchical Modeling

The model employs a hierarchical structure, particularly for house effects and district offsets. Parameters for individual pollsters or districts are assumed to be drawn from common distributions, allowing the model to borrow strength across units and make more robust estimates, especially for units with less data.

### Bayesian Inference

The model parameters are estimated within a Bayesian framework, typically using Markov Chain Monte Carlo (MCMC) methods. This yields a full posterior distribution for all parameters and derived quantities (like vote shares and seat predictions), naturally quantifying uncertainty.

### Computational Techniques

To make Bayesian inference computationally feasible, the model utilizes:
*   **GP Approximations:** Efficient methods (like basis function expansions) are used to approximate the full Gaussian Processes, reducing the computational complexity.
*   **Reparameterization:** Techniques like non-centered parameterization are used for certain hierarchical parameters to improve the geometry of the posterior distribution and the efficiency of MCMC sampling algorithms.

### Overdispersion Modeling

The use of a likelihood function that explicitly models overdispersion (like the Dirichlet-Multinomial) is crucial for realistically capturing the noise characteristics of polling and election data.

## Making Predictions

Generating forecasts involves several steps:

1.  Draw samples from the joint posterior distribution of all model parameters obtained via Bayesian inference.
2.  For each sample, compute the latent national support trend (sum of the three calendar-time GPs) at the desired future date(s).
3.  Apply the relevant district-specific **static base offset** parameters (as estimated from the posterior) to the national latent trend to get district-level latent scores.
4.  Convert these latent scores into predicted vote share probabilities using the softmax transformation.
5.  Simulate the seat allocation process (D'Hondt method) using these predicted vote shares for each posterior sample.

Aggregating the results across all posterior samples provides a probabilistic forecast for vote shares and seat counts, inherently reflecting model uncertainty.

### District Vote Share Prediction

District-level vote share predictions are derived by combining the posterior distribution of the national latent trend (sum of the three calendar-time GPs) with the posterior distribution of the **static** district base offsets.

Specifically, for each posterior sample and each district:
1.  The estimated **static base offset** for that district and party is added to the national latent trend value (sum of the three GPs) for that party at the target date.
2.  The resulting adjusted latent scores are transformed into probabilities (vote shares summing to 1) via the softmax function.

This procedure yields a full posterior distribution of predicted vote shares for each party within each district.

### Seat Allocation

Once we have vote share predictions, we simulate the election outcome using the D'Hondt method, which allocates seats proportionally based on each party's votes. By running this simulation thousands of times across our posterior samples, we generate a probability distribution over possible seat outcomes for each party.

## Limitations and Future Improvements

Like all models, ours has limitations based on its current structure:

1. It assumes that historical patterns of *static* district behavior (relative to the nation, captured by the base offsets) will continue into the future. The model currently does not account for potential dynamic changes in how districts respond to national swings within a cycle.
2. It does not incorporate non-polling data such as economic indicators or government approval ratings.
3. The district effects model could potentially be enhanced in future versions by re-introducing dynamic components (like sensitivity/beta), adding district-level covariates, or incorporating spatial correlation structures.

Future versions may address these limitations by incorporating additional data sources (like economic indicators), activating dynamic district effects, using district-level covariates (such as demographics or past voting patterns) to better model the static offsets, or implementing spatial modeling techniques to capture correlations between neighboring districts.
