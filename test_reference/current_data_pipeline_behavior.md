# Current Data Pipeline Behavior Documentation

This document captures the behavior of the current data loading pipeline as of the initial regression test implementation. This serves as the baseline for detecting any changes.

## Data Loading Structure

### Poll Data (Marktest)
- **Source**: `load_marktest_polls()`  
- **Records**: 602 polls total
- **Date Range**: 2011-2025 (includes future tracking polls)
- **Structure**: Multinomial counts (vote counts sum to sample_size)
- **Sample Sizes**: Range from 100 to 33,798
- **Parties**: `['PS', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L', 'AD']`

### Election Results
- **Source**: `load_election_results()`
- **Historical Elections**: 5 elections (2011, 2015, 2019, 2022, 2024)
- **Geographic Levels**: 
  - National aggregation: 5 records
  - District level: 100 records (20 districts × 5 elections)
- **Districts**: 20 total districts including mainland + islands

### Current Coalition Handling
- **AD Coalition**: Treated as single party entity
- **No PSD/CDS separation**: System assumes AD represents PSD+CDS alliance
- **Government Mapping**: 
  - 2011-2015: PSD+CDS (mapped to AD=1 in government status)
  - 2015-2024: PS government  
  - 2024+: AD government

## Data Processing Pipeline

### 1. Raw Data Loading
```
load_marktest_polls() → DataFrame[602 rows]
├── Tracking poll consolidation (73 → 28 entries)  
├── Pollster name cleaning/standardization
├── Missing value filling (parties default to 0)
└── Sample size validation/imputation
```

### 2. Multinomial Conversion  
```
cast_as_multinomial() → vote counts
├── Convert percentages to counts based on sample_size
├── Ensure vote_sums == sample_sizes (exactly)
└── Maintain integer precision
```

### 3. Election Date Assignment
```
find_closest_election_date() → election cycle mapping
├── Polls → next upcoming election
├── Post-2024 polls → 2026-01-01 (target)
└── Countdown calculation (days to election)
```

### 4. Geographic Processing
```
load_election_results() → District aggregation
├── Parish-level data → District totals  
├── 20 districts with seat allocations
└── Multi-election time series (5 elections)
```

### 5. Coordinate Generation
```
ElectionDataset → Model coordinates
├── Political families: 8 parties
├── Pollsters: Factorized from poll data
├── Elections: 6 cycles (5 historical + 1 target)
└── Districts: 20 geographic units
```

## Data Quality Characteristics

### Poll Data Quality
- **Completeness**: All polls have required columns
- **Consistency**: Vote counts exactly equal sample sizes
- **Range Validation**: Sample sizes 100-50,000 are acceptable
- **Temporal Coverage**: Spans multiple election cycles

### Result Data Quality  
- **Historical Coverage**: 5 complete election cycles
- **Geographic Coverage**: All 20 districts represented
- **Aggregation Consistency**: District totals available
- **Date Alignment**: Results match expected election dates

### Coalition Data Handling
- **Current Approach**: AD treated as unified party
- **Historical Consistency**: Government status correctly mapped
- **No Disaggregation**: PSD/CDS not separately tracked
- **Works Well**: Current forecasting performance is good

## System Dependencies

### Critical Assumptions
1. **AD = PSD+CDS**: Hardcoded coalition assumption
2. **District Structure**: 20 districts with fixed boundaries  
3. **Party List**: 8-party system is complete
4. **Multinomial Data**: Poll data converted to counts

### Data Format Requirements
- **Polls**: `date`, `pollster`, `sample_size`, party columns
- **Results**: `election_date`, `Circulo`, party columns, `sample_size`
- **Districts**: District name → seat count mapping
- **Dates**: ISO format YYYY-MM-DD

## Performance Characteristics
- **Dataset Initialization**: ~6.8 seconds
- **Memory Usage**: Reasonable for 602 polls + 100 district results
- **Deterministic**: Repeated runs produce identical coordinates
- **Scalable**: Handles multiple elections and pollsters

## Success Metrics (Baseline)
- ✅ All 8 regression tests pass
- ✅ 602 polls loaded successfully  
- ✅ 5 historical elections with complete data
- ✅ 20 districts with seat allocations
- ✅ Multinomial conversion maintains exact counts
- ✅ Coalition handling works with current AD structure
- ✅ Coordinate generation is deterministic

## Critical Behaviors to Preserve
1. **Exact Vote Count Equality**: `sum(party_votes) == sample_size`
2. **Coalition Representation**: AD remains unified party
3. **District Count**: Exactly 20 districts
4. **Historical Elections**: 5 complete election records  
5. **Deterministic Coordinates**: Same inputs → same coordinates
6. **Government Status**: Correct party-in-power mapping

---

**Generated**: 2025-09-11  
**Purpose**: Baseline documentation for regression testing  
**Status**: All tests passing - this represents working system behavior