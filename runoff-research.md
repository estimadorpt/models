# Runoff Election Forecasting: Research Summary

## Context

The Portuguese presidential election first round occurred on January 18, 2026. No candidate achieved >50%, resulting in a second round scheduled for February 8, 2026, between António José Seguro (PS) and André Ventura (CH). This document summarizes approaches used by established election forecasting models for handling two-round/runoff elections and recommends a principled Bayesian approach.

## Research Findings

### 1. FiveThirtyEight Approach (US Senate/House)

FiveThirtyEight's methodology for runoff elections in states like Georgia and Louisiana:

- **Total Party Vote**: Uses the total party vote as a primary predictor for runoff outcomes
- **First-Round Margin**: Research found that the margin separating the top two candidates in the first round has additional predictive power beyond party vote totals
- **Example Formula**: In 2021 Georgia Senate runoffs, when Warnock finished ahead of Loeffler by 7 points, the model implied he would have won a two-way race by about 1.5 points
- **Simulation Approach**: Win probabilities come from simulating the election 20,000 times, producing a distribution of possible outcomes

**Limitation**: FiveThirtyEight primarily handles US contexts where runoffs occur in limited circumstances. Their model was shut down in March 2025.

Sources:
- [How 538's 2024 Senate election forecast works](https://abcnews.go.com/538/538s-2024-senate-election-forecast-works/story?id=114997770)
- [Why A Split Verdict In Georgia Isn't That Crazy](https://fivethirtyeight.com/features/why-a-split-verdict-in-georgia-isnt-that-crazy/)

### 2. The Economist / Andrew Gelman Approach

The Economist's model (developed by Andrew Gelman and collaborators) uses a dynamic multilevel Bayesian framework:

- **Fundamentals as Priors**: Uses historical voting patterns, economic indicators, and political fundamentals as informative priors
- **Kalman Filter-like Updates**: National and state polling serve as "noisy observations" that update a hidden state representing true voter preferences
- **Dynamic Linear Model**: Captures how public opinion drifts over time
- **Random Walk Innovation**: The standard deviation of the random walk governs how quickly the model responds to new information

**Key Methodological Insight**: The model uses fundamentals-based predictions as the prior for Election Day, then updates this prior with polling data. This framework naturally extends to using first-round results as an informative prior for second-round forecasting.

**Limitation**: The Economist model focuses on US presidential elections (single-round system). No specific runoff methodology published.

Sources:
- [The Economist's US Presidential Model GitHub](https://github.com/TheEconomist/us-potus-model)
- [An Updated Dynamic Bayesian Forecasting Model](https://hdsr.mitpress.mit.edu/pub/nw1dzd02)
- [Grappling With Uncertainty in Forecasting the 2024 U.S. Presidential Election](https://hdsr.mitpress.mit.edu/pub/yoa73r1m/release/1)

### 3. Cepesp Parsimonious Model (Brazil)

The most directly relevant research comes from Cepesp (Centro de Política e Economia do Setor Público) which developed a parsimonious model specifically for runoff elections:

**Key Findings**:
- First-round results are highly predictive of second-round outcomes
- Using 128 presidential elections from 44 countries, they found a strong positive relationship between first-round vote share and second-round outcomes
- Confirmed with 287 Brazilian executive elections

**Why First-Round Results Are Predictive**:
1. First-round results incorporate candidate and campaign-specific characteristics
2. Abstention rates increase slightly across rounds
3. Trailing candidates often need an implausibly large share of remaining votes
4. Voters of eliminated candidates rarely have homogeneous enough preferences to overturn results

**Model Specification**: The paper suggests that simple vote share ratios from the first round perform nearly as well as more sophisticated specifications including additional predictors.

Sources:
- [Cepesp: Forecasting Runoff Elections Using First-round Results](http://www.cepesp.io/publicacoes/forecasting-runoff-elections-a-parsimonious-model-using-first-round-results/)

### 4. French Presidential Election Models

France, which uses a similar two-round system, has developed several approaches:

**Vote Transfer Models**:
- Researchers use a "transfer function" from first-round votes to second-round
- Results are calibrated to a Dirichlet distribution, then combined with vote-transfer models
- The transfer function distinguishes between "partisan vote" (loyal supporters) and "opportunistic vote" (strategic voters)

**SUR Regionalized Models**:
- Pooled time-series models across regions and election years
- Successfully predicted the 2007 second round (53.5% vs actual 53.05%)
- Include economic (unemployment) and political (approval ratings) variables

**Challenges**:
- Vote transfer predictions are sensitive to campaign events between rounds
- Eliminated candidates' endorsements can shift vote patterns
- Turnout changes between rounds affect results

Sources:
- [Electoral forecasting in France: A multi-equation solution](https://www.researchgate.net/publication/46497635_Electoral_forecasting_in_France_A_multi-equation_solution)
- [Forecasting the 2022 French Presidential Election with a SUR Regionalized Model](https://www.researchgate.net/publication/363073615_Forecasting_the_2022_French_Presidential_Election_with_a_SUR_Regionalized_Model)

### 5. Academic Bayesian Approaches

Several academic papers address hierarchical Bayesian models for multi-round election forecasting:

**Vote Transfer Matrix Methods**:
- Estimate voter shifts between elections using row-standardized proportion matrices
- Hierarchical Bayesian models can solve both consistency and completeness problems
- Can use prior election results as informative priors

**Multiparty Extensions**:
- Backward random-walk approaches extended to compositional (sum-to-one) settings
- Use Dirichlet regression with Gaussian process priors for multiparty elections
- Combine fundamentals with polls in a fully integrated Bayesian framework

**Key Insight**: The Bayesian framework naturally allows using first-round results as informative priors, which are then updated by second-round polling.

Sources:
- [Forecasting Elections in Multiparty Systems: A Bayesian Approach](https://www.cambridge.org/core/journals/political-analysis/article/forecasting-elections-in-multiparty-systems-a-bayesian-approach-combining-polls-and-fundamentals/CA929544F672A09A0E34C5529EBFA482)
- [Forecasting Elections from Partial Information Using a Bayesian Model](https://arxiv.org/html/2104.02924v2)

### 6. Público GPS Model (Portugal)

The GPS (Grande Portal de Sondagens) model was not directly accessible for methodology review. Per user context, the owner considers it "less principled" and does not recommend direct copying. We should develop our own approach based on the academic literature.

---

## Recommended Approach for Portuguese Second Round

Based on this research, I recommend a **Bayesian model that treats the first round as a strong informative prior** rather than treating the second round as completely fresh.

### Option A: First-Round Results as Fixed Prior (Recommended)

**Concept**: Use actual first-round results to set the starting support levels, then allow a GP or random walk to model movement based on second-round polls.

**Implementation**:
```python
# First-round results (actual)
seguro_r1 = 0.31  # 31%
ventura_r1 = 0.235  # 23.5%

# Normalize to two-candidate race as prior mean
total = seguro_r1 + ventura_r1
prior_seguro = seguro_r1 / total  # ~56.9%
prior_ventura = ventura_r1 / total  # ~43.1%

# Convert to log-odds for model
prior_logit = np.log(prior_seguro / prior_ventura)  # ~0.28
```

**Advantages**:
- First-round results incorporate all campaign-specific factors up to election day
- Aligns with Cepesp finding that first-round results are highly predictive
- Provides principled starting point for second-round dynamics
- House effects estimated from first round can carry forward

**Model Structure**:
1. Prior: First-round normalized vote shares (two-candidate basis)
2. Time dynamics: Short GP or random walk from first round to second round
3. Polls: Update the prior with second-round polling (likely only 1-3 polls)
4. Likelihood: DirichletMultinomial on two-candidate polls

### Option B: Vote Transfer Model

**Concept**: Explicitly model how votes from eliminated candidates transfer to finalists.

**Implementation**:
- Collect exit poll data or voter intention surveys asking eliminated candidate supporters who they'll vote for
- Build transfer matrix: P(vote_Seguro | voted_X_R1) for each eliminated candidate X
- Propagate uncertainty through transfer probabilities

**Challenges**:
- Requires second-round specific polling data on vote transfers
- Transfer rates can change rapidly with endorsements
- May not have enough data for Portugal's rare runoffs

### Option C: Fresh Model with Informative Priors

**Concept**: Build a new two-candidate model, but use first-round data to inform priors.

**Implementation**:
- Candidate prior means: Derived from first-round results (as in Option A)
- House effects: Carry forward estimates from first-round model
- Uncertainty: Start with wider uncertainty than first-round final estimates

**Trade-off**: More flexible but may underweight the information in first-round results.

---

## Specific Recommendations

### 1. Model Architecture

**Recommended**: Create a `SecondRoundModel` class that:
- Takes first-round results as informative priors
- Uses a shorter GP timescale (3-7 days) given the compressed campaign
- Supports only two candidates (simplifies to logit/probit space)
- Carries forward house effect estimates from first-round model

### 2. Prior Specification

```python
# First-round results define prior mean
r1_seguro = 0.31
r1_ventura = 0.235

# Normalize to two-candidate race
normalized_seguro = r1_seguro / (r1_seguro + r1_ventura)  # 0.569

# Prior SD: Allow for vote transfer uncertainty
# Other candidates got ~45% of vote; their redistribution is uncertain
# SD of ~5-8% on final two-candidate share is reasonable
prior_sd = 0.06  # in probability space, ~0.25 logits

# In logit space:
prior_logit_mean = logit(normalized_seguro)  # ~0.28
prior_logit_sd = 0.25  # allows ~5-6pp movement
```

### 3. Incorporating Vote Transfers (Optional Enhancement)

If second-round polling asks about vote transfers:
```python
# Survey: "Who will Cotrim voters support?"
# e.g., 60% Seguro, 25% Ventura, 15% abstain/undecided
cotrim_to_seguro = 0.60
cotrim_to_ventura = 0.25

# Weight by Cotrim's first-round vote share
cotrim_r1 = 0.22
seguro_transfer = cotrim_r1 * cotrim_to_seguro
ventura_transfer = cotrim_r1 * cotrim_to_ventura
```

### 4. Timeline Considerations

- **First round**: Jan 18
- **Second round**: Feb 8
- **Gap**: 21 days

This is short. The model should:
- Use first-round results heavily
- Allow limited movement based on 2-3 weeks of campaigning
- Weight any second-round polls appropriately

### 5. House Effect Handling

**Options**:
1. **Carry forward**: Use house effects estimated from first-round model
2. **Re-estimate**: If enough second-round polls, estimate new two-candidate house effects
3. **Hybrid**: Use first-round estimates as informative priors for second-round

**Recommended**: Carry forward first-round house effects, unless pollsters use significantly different methodology for second-round polling.

---

## Comparison with Current Model

Our current model (`PresidentialElectionModel`) forecasts runoff pairs probabilistically by:
1. Sampling posterior trajectories for all candidates
2. Computing who reaches second round in each sample
3. Simulating head-to-head outcomes

For the actual second round, we should:
- Condition on the known first-round outcome (Seguro vs Ventura)
- Use the normalized first-round shares as the new prior
- Model only the two finalists going forward

---

## Implementation Roadmap

### Phase 1: Minimal Viable Second-Round Model
1. Create `SecondRoundModel` class (2-candidate logit model)
2. Use first-round normalized shares as prior mean
3. Allow GP/RW dynamics with short timescale
4. Carry forward house effects from first-round model

### Phase 2: Enhanced with Vote Transfers (if data available)
1. Parse exit poll or intention data on vote transfers
2. Build transfer probability model
3. Incorporate transfer uncertainty into prior

### Phase 3: Dashboard Integration
1. Export second-round forecast JSON files
2. Update estimador-web to display two-candidate race
3. Show vote transfer assumptions (if used)

---

## Key Takeaways

1. **First-round results are highly predictive** of second-round outcomes (Cepesp research)
2. **Bayesian updating is natural**: First round → prior, second-round polls → posterior
3. **Keep it simple**: Parsimonious models using first-round shares perform nearly as well as complex specifications
4. **Short timeline = less movement**: Only 21 days means first-round results should be weighted heavily
5. **House effects can carry forward**: No need to re-estimate if using same pollsters

---

## References

### Primary Academic Sources
- Avelino, Russo, Pimentel Jr. (Cepesp) - "Forecasting Runoff Elections: A Parsimonious Model Using First-round Results"
- Linzer (2013) - "Dynamic Bayesian Forecasting of Presidential Elections in the States"
- Lock & Gelman (2010) - Combining polls with fundamentals
- Heidemanns, Gelman, Morris (2020) - "An Updated Dynamic Bayesian Forecasting Model for the U.S. Presidential Election"

### Methodology Documentation
- [FiveThirtyEight Senate/House Methodology](https://fivethirtyeight.com/methodology/how-fivethirtyeights-house-and-senate-models-work/)
- [The Economist Model GitHub](https://github.com/TheEconomist/us-potus-model)
- [Cambridge Core: Forecasting Elections in Multiparty Systems](https://www.cambridge.org/core/journals/political-analysis/article/forecasting-elections-in-multiparty-systems-a-bayesian-approach-combining-polls-and-fundamentals/CA929544F672A09A0E34C5529EBFA482)

### Portuguese Election Context
- [2026 Portuguese Presidential Election - Wikipedia](https://en.wikipedia.org/wiki/2026_Portuguese_presidential_election)
- [Estimador.pt](https://estimador.pt/en/)
