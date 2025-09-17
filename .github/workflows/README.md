# GitHub Actions CI/CD Workflows

This directory contains comprehensive CI/CD workflows for the election modeling pipeline, implementing automated regression testing and quality assurance.

## 🔄 Workflow Overview

### 1. 🔍 Regression Testing (`regression-testing.yml`)

**Primary workflow for change validation**

- **Triggers**: Pull requests, pushes to main, manual dispatch
- **Purpose**: Ensure no regressions are introduced by changes
- **Duration**: 10-60 minutes depending on test level

**Test Levels**:
- `quick`: Coalition + deterministic tests (10 min)
- `full`: Complete pipeline integration (45 min) 
- `performance`: Performance benchmarking (90 min)

**Key Features**:
- ✅ Automatic change detection (skips docs-only changes)
- 🔬 Bit-for-bit comparison against golden masters
- 📊 Statistical validation of model outputs
- ⚡ Performance regression detection
- 🤝 Coalition handling validation

### 2. 🌙 Nightly Comprehensive Testing (`nightly-comprehensive-testing.yml`)

**Comprehensive system health monitoring**

- **Triggers**: Daily at 2 AM UTC, manual dispatch
- **Purpose**: Deep validation and early issue detection
- **Duration**: Up to 3 hours

**Features**:
- 🧪 Full integration test suite with coverage
- 🔍 Comprehensive regression detection
- ⚡ Multi-configuration performance benchmarking
- 🔬 Data quality validation
- 🚨 Automatic issue creation on failures
- ✅ Auto-closing of resolved issues

### 3. ✅ Pull Request Validation (`pr-validation.yml`)

**Fast feedback for pull requests**

- **Triggers**: PR events (opened, synchronized, etc.)
- **Purpose**: Quick validation and developer feedback
- **Duration**: 10-45 minutes

**Features**:
- 🚀 Smart change analysis
- 🎯 Targeted testing based on changes
- 📝 Automatic PR comments with results
- 🔍 Regression testing for significant changes
- 🧪 Test validation for test-only changes

## 📊 Testing Infrastructure

### Golden Master Baselines

Located in `test_baselines/`, these provide reference outputs for regression detection:

```
test_baselines/
├── golden_masters_metadata.json    # Configuration and timestamps
└── train/                          # Training outputs
    ├── model_config.json          # Model configuration
    ├── fit_metrics.json           # Fit quality metrics
    ├── predictions/               # Prediction outputs
    │   ├── vote_share_summary_election_day.csv
    │   └── total_seat_summary_direct_election_day.csv
    └── visualizations/            # Generated plots
```

### Regression Detection Tools

The `scripts/regression_detection_tools.py` provides comprehensive comparison capabilities:

- **BitForBitComparator**: Exact reproducibility validation
- **StatisticalComparator**: Model output validation with MCMC tolerance
- **PerformanceRegessionDetector**: Execution time monitoring
- **CoalitionValidationDetector**: TARGET-ELECTION-DRIVEN approach validation

### Test Suites

- `tests/integration/test_current_system_behavior.py`: Full pipeline testing
- `tests/regression/`: Existing validation framework
- `scripts/test_regression_tools.py`: Tool validation

## 🔧 Configuration

### Environment Requirements

- **Python**: 3.11
- **Package Manager**: pixi
- **Key Dependencies**: PyMC, pandas, numpy, pytest

### Environment Variables

```bash
PYTHONHASHSEED=0      # Deterministic behavior
PYTHONUTF8=1          # Consistent encoding
OMP_NUM_THREADS=2     # Control parallelism
NUMBA_NUM_THREADS=2   # Control compilation
```

### Test Thresholds

- **Vote Share Difference**: ±2 percentage points
- **Seat Difference**: ±2 seats
- **Fit Metric Regression**: 30% tolerance
- **Performance Regression**: 2x slowdown threshold

## 🚀 Usage Guide

### For Developers

1. **Making Changes**:
   - Create feature branch from `main`
   - Make your changes
   - Push to trigger PR validation
   - Address any regression test failures

2. **PR Process**:
   - PR validation runs automatically
   - Check PR comments for regression results
   - Green checkmarks = ready for review
   - Red X = fix regressions before merge

3. **Interpreting Results**:
   - ✅ All green: No regressions detected
   - ⚠️ Warnings: Minor changes within tolerance
   - ❌ Failures: Regressions detected, requires fixes

### For Maintainers

1. **Golden Master Updates**:
   ```bash
   # Generate new baselines after approved changes
   pixi run python scripts/generate_golden_masters.py
   
   # Commit and push
   git add test_baselines/
   git commit -m "🔄 Update golden master baselines"
   git push
   ```

2. **Manual Workflow Triggers**:
   - Go to Actions tab in GitHub
   - Select workflow to run
   - Click "Run workflow"
   - Choose parameters (test level, etc.)

3. **Monitoring Health**:
   - Check nightly validation results
   - Review performance trends
   - Address failure issues promptly

## 🔍 Troubleshooting

### Common Issues

1. **Golden Masters Missing**:
   ```bash
   # Generate fresh baselines
   pixi run python scripts/generate_golden_masters.py
   ```

2. **Import Errors**:
   ```bash
   # Check environment setup
   pixi install
   pixi run python -c "import src.main"
   ```

3. **Performance Regressions**:
   - Check for inefficient code changes
   - Review MCMC parameter changes
   - Compare with historical benchmarks

4. **Coalition Handling Issues**:
   - Verify AD=PSD+CDS coalition structure
   - Check TARGET-ELECTION-DRIVEN approach
   - Review coalition manager changes

### Debugging Workflows

1. **Download Artifacts**:
   - Go to failed workflow run
   - Download test result artifacts
   - Review detailed logs and reports

2. **Local Reproduction**:
   ```bash
   # Run same tests locally
   pixi run python -m pytest tests/integration/ -v
   
   # Run regression detection
   pixi run python scripts/regression_detection_tools.py \
     --golden-masters test_baselines \
     --current-output path/to/output \
     --quick
   ```

3. **Check Logs**:
   - Review workflow step logs
   - Look for error messages and stack traces
   - Check artifact contents

## 📈 Monitoring and Metrics

### Key Metrics Tracked

- **Test Success Rate**: Percentage of passing tests
- **Performance Trends**: Execution time over time  
- **Data Quality**: Coalition structure, data consistency
- **Coverage**: Code coverage from integration tests

### Health Indicators

- **✅ Healthy**: All tests pass, no regressions
- **⚠️ Warning**: Minor issues, some warnings
- **❌ Critical**: Test failures, significant regressions

### Alerts

- Nightly validation failures create GitHub issues
- Performance regressions trigger warnings
- Data quality issues generate alerts
- Golden master generation failures notify maintainers

## 🔄 Maintenance

### Regular Tasks

1. **Weekly**: Review nightly validation trends
2. **Monthly**: Update golden masters if needed
3. **Quarterly**: Review and optimize workflows
4. **Annually**: Update CI/CD documentation

### Golden Master Management

- **Update Trigger**: Approved model improvements
- **Validation**: All regression tests must pass
- **Backup**: Previous baselines archived automatically
- **Documentation**: Changes documented in commit messages

### Performance Optimization

- Monitor workflow execution times
- Optimize test parallelization
- Cache dependencies when possible
- Use appropriate test levels for different triggers

---

## 📚 Related Documentation

- [Regression Testing Guide](../docs/regression-testing.md)
- [Golden Master Management](../docs/golden-master-management.md)
- [Coalition Handling Documentation](../docs/coalition_handling_documentation.md)
- [Performance Benchmarking](../docs/performance-benchmarking.md)

## 🤝 Contributing

When modifying workflows:

1. Test changes in feature branches
2. Update documentation
3. Ensure backward compatibility
4. Review impact on CI/CD performance
5. Get approval from maintainers

For questions or issues with CI/CD workflows, please create an issue with the `ci-cd` label.