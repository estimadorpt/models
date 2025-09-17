# Static Model Deprecation Notice

**Date**: September 16, 2025  
**Branch**: test/comprehensive-validation-suite  

## Decision: Archive Static Baseline Model

The `StaticBaselineElectionModel` has been deprecated in favor of the `DynamicGPElectionModel` for the following reasons:

### Why Dynamic GP is Superior

1. **Better Theoretical Foundation**: Models long-term baseline changes over calendar time using Gaussian Processes
2. **More Realistic**: Captures smooth evolution of party support rather than assuming static baselines
3. **Improved Performance**: Empirically better fit to data and more accurate forecasts
4. **Active Development**: All recent development has focused on dynamic GP model

### What We've Done

1. **Updated Default**: Changed `--model-type` default from "static" to "dynamic_gp" in `src/main.py`
2. **Updated Pixi Tasks**: Added explicit `--model-type dynamic_gp` to training commands in `pixi.toml`
3. **Golden Masters**: All regression testing baselines use dynamic GP model

### Files to Consider for Archival

- `src/models/static_baseline_election_model.py` - The static model implementation
- Related test files that only test static model functionality

### Migration Notes

- All existing functionality is preserved in the dynamic GP model
- Better performance and accuracy than static model
- No breaking changes to API or output formats

### Recommendation

Consider moving `static_baseline_election_model.py` to an `archived/` directory or removing it entirely after confirming no dependencies exist in the codebase.

### Impact on Issue #31 (Comprehensive Testing)

This decision ensures our golden masters and regression tests are based on the current, actively-maintained model rather than deprecated code. This provides:

- More relevant test coverage
- Easier maintenance going forward  
- Better detection of actual regressions vs. changes due to model switching

The comprehensive testing infrastructure (Issue #31) will be built around the dynamic GP model exclusively.