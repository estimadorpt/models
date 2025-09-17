# Current Coalition Handling Documentation

**Date**: September 16, 2025  
**Status**: Golden Master Reference for Regression Testing  
**File**: `src/data/coalition_manager.py`

## Overview

The current coalition management system uses a **TARGET-ELECTION-DRIVEN approach** that prioritizes Bayesian modeling consistency over strict historical accuracy. This approach is crucial for the success of the election forecasting system.

## Core Philosophy: Why AD = PSD + CDS Works

### The Modeling Challenge

Portuguese coalition politics creates a fundamental problem for Bayesian time series models:

- **2019-2022**: Polls show `PSD: 25%, CDS: 2%` separately  
- **2024+**: Polls show `AD: 29%` (PSD+CDS coalition)

This creates **discontinuous time series** that break Bayesian modeling assumptions.

### The Solution: Consistent Party Coordinates

The coalition manager creates **stable party coordinates** by:

1. **Starting with target election structure** (e.g., 2026 with AD coalition)
2. **Projecting coalitions backwards** through all historical data
3. **Combining component parties retroactively** for consistent time series
4. **Ensuring no artificial discontinuities** from coalition formation dates

### Example: AD Time Series Consistency

```
Target Election (2026): AD (from target structure)
2024 Election Data:     AD (actual coalition data)  
2022 Election Data:     AD = PSD + CDS (combined retroactively)
2019 Election Data:     AD = PSD + CDS (combined retroactively) 
2015 Election Data:     AD = PSD + CDS (combined retroactively)
```

**Result**: Smooth `AD` time series from 2015→2026 for proper Bayesian trend modeling.

## Key Components

### 1. TargetElectionStructure

Defines the party structure for the target election:

```python
target_2026 = TargetElectionStructure(
    coalitions={
        'PS': ['PS'],
        'AD': ['PSD', 'CDS'],  # Key coalition for stable modeling
        'CH': ['CH'],
        'IL': ['IL'],
        'BE': ['BE'],
        'CDU': ['PCP', 'PEV'],  # Historical communist coalition
        'PAN': ['PAN'],
        'L': ['L']
    },
    election_type='parliamentary'
)
```

### 2. Backward Projection Logic

The `resolve_historical_column_to_target()` method performs the magic:

1. **Single Party**: Direct column matching via aliases
   - `PS` → `PARTIDO SOCIALISTA` or `PS`
   
2. **Coalition**: Multi-level matching
   - `AD` → `PPD/PSD.CDS-PP` (2022 format)
   - `AD` → `PSD` + `CDS` (separate columns combined)

### 3. Party Aliases System

Handles electoral data variations:

```python
party_aliases = {
    'PS': ['PS', 'PARTIDO SOCIALISTA'],
    'PSD': ['PSD', 'PPD/PSD', 'PPD'],
    'CDS': ['CDS-PP', 'CDS'],
    'CH': ['CH', 'CHEGA'],
    # ... etc
}
```

### 4. Municipal Extensions

For municipal elections, adds:

- **DisaggregationRule**: Split national `AD: 30%` → `PSD: 25.5%, CDS: 4.5%`
- **MunicipalCoalitionStructure**: Recombine into local patterns
- **Geographic Overrides**: Municipality-specific coalitions

## Why This Approach Is Successful

### 1. Bayesian Model Requirements

- **Stable Coordinates**: Models need consistent party names across time
- **Continuous Time Series**: No artificial breaks from coalition changes  
- **Proper Trend Modeling**: Long-term patterns captured correctly

### 2. Real-World Benefits

- **Historical Consistency**: AD time series goes back to 2015+ 
- **Accurate Forecasting**: Proper baseline trends for coalition support
- **Municipal Flexibility**: Handles complex local coalition patterns

### 3. Practical Implementation

- **Zero Breaking Changes**: Existing data loading works unchanged
- **Transparent Mapping**: Clear audit trail of column transformations
- **Extensible Design**: Easy to add new coalitions or geographic patterns

## Data Flow Example

### Input: 2022 Election Data
```
Columns: ['PS', 'PPD/PSD.CDS-PP', 'CH', 'IL', 'BE', 'PCP-PEV', 'PAN', 'L']
```

### Target Mapping Process
```
Target Party: AD
Components: ['PSD', 'CDS'] 
Historical Match: 'PPD/PSD.CDS-PP'
Result: df['AD'] = df['PPD/PSD.CDS-PP']
```

### Output: Consistent Model Data
```
Columns: ['PS', 'AD', 'CH', 'IL', 'BE', 'CDU', 'PAN', 'L']
```

## Municipal Election Innovation

### National→Municipal Propagation

**Problem**: National polls show `AD: 30%` but municipalities have different patterns.

**Solution**: 
1. Disaggregate `AD` → `PSD: 25.5% + CDS: 4.5%`
2. Recombine per municipality:
   - **Aveiro**: `PSD` vs `CDS` compete separately  
   - **Lisboa**: `PSD+IL` coalition (33.5% combined)
   - **Rural**: Traditional `AD` coalition (30% recombined)

### Real Municipal Data Integration

Loads municipal coalition patterns from `municipal_coalitions_2025.parquet`:

```python
# Example municipal structures loaded automatically
municipalities = {
    'Aveiro': {'PSD': ['PSD'], 'CDS': ['CDS'], 'IL': ['IL']},
    'Lisboa': {'PSD_IL': ['PSD', 'IL'], 'CDS': ['CDS']},  
    'Porto': {'AD': ['PSD', 'CDS'], 'IL': ['IL']}
}
```

## Critical Success Factors

### 1. Modeling Consistency Over Historical Purity

- **Principle**: Bayesian models need stable coordinates
- **Trade-off**: Some historical "accuracy" sacrificed for modeling consistency
- **Benefit**: Proper trend modeling and accurate forecasts

### 2. Component Flexibility

- Works with raw historical data (different column names/formats)
- Handles coalition formation/dissolution gracefully
- Supports complex municipal variations

### 3. Comprehensive Testing

The system includes extensive validation:
- **Coalition handling calculations** (`test_calculation_validation.py`)
- **Data integrity validation** (`test_data_integrity_validation.py`)  
- **Multi-level regression tests** (current work)

## Impact on Model Performance

### Before Coalition Management
- **Inconsistent time series**: PSD/CDS → AD breaks
- **Poor trend modeling**: Artificial discontinuities 
- **Coalition confusion**: Model uncertainty about party entities

### After Coalition Management  
- **Smooth time series**: Consistent AD from 2015→2026
- **Proper baselines**: Long-term trends captured correctly
- **Clear party semantics**: Stable party coordinates for Bayesian inference

## Regression Testing Requirements

For Issue #31, this system must be preserved exactly:

### Golden Master Validation
- [ ] AD coalition handling works identically
- [ ] Historical data mapping unchanged  
- [ ] Municipal disaggregation patterns preserved
- [ ] Performance characteristics maintained

### Test Coverage
- [ ] All party alias matching works
- [ ] Geographic overrides function correctly
- [ ] Backward projection logic preserved
- [ ] Municipal data loading succeeds

## Future Development Notes

### Strengths to Preserve
- **TARGET-DRIVEN approach**: Core innovation for Bayesian consistency
- **Flexible alias system**: Handles real-world data variations
- **Municipal extensions**: Supports complex local politics

### Areas for Careful Evolution  
- **Add new coalitions**: Follow same target-driven pattern
- **Geographic patterns**: Extend municipal override system
- **Performance optimization**: Maintain same functional behavior

## Dependencies

This coalition system is foundational to:
- **Data loading**: All election and poll data mapping  
- **Model training**: Stable party coordinates for Bayesian models
- **Forecasting**: Consistent party structure for predictions
- **Visualization**: Coherent party names across all outputs

**Any changes to this system must preserve the exact functional behavior captured in the golden masters.**