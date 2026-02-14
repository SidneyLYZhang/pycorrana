# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2026-02-14

### Added

#### Big Data Optimization

- `LargeDataConfig` class for big data configuration
- `smart_sample()` function for intelligent sampling (supports random/stratified sampling)
- `chunked_correlation()` function for chunked correlation calculation
- `chunked_apply()` function for chunked function application
- `optimize_dataframe()` function for memory optimization
- `is_large_data()` function for big data detection
- `estimate_memory_usage()` function for memory usage estimation
- CorrAnalyzer now supports `large_data_config` parameter

#### Semipartial Correlation Analysis

- `semipartial_corr()` function for semipartial (part) correlation

### Changed

- Optimized automatic detection and prompts for large datasets
- Improved memory usage efficiency
- Updated project structure to src layout

## [0.1.0] - 2026-02-13

### Added

#### Core Analysis Features

- `quick_corr()` one-line analysis function
- `CorrAnalyzer` analyzer class
- Automatic method selection (Pearson/Spearman/Kendall/Cramér's V/Eta, etc.)
- Significance testing and p-value correction

#### Data Preprocessing

- Support for multiple data formats (CSV/Excel/pandas/polars)
- Automatic type inference
- Missing value handling (drop/fill)
- Outlier detection

#### Visualization

- Correlation heatmap (with clustering support)
- Scatter plot matrix
- Box plots / violin plots
- Correlation network graph

#### Result Export

- Excel export
- CSV export
- Text summary

#### Partial Correlation Analysis

- `partial_corr()` partial correlation coefficient calculation
- `partial_corr_matrix()` partial correlation matrix
- `PartialCorrAnalyzer` analyzer class

#### Nonlinear Analysis

- `distance_correlation()` distance correlation
- `mutual_info_score()` mutual information
- `maximal_information_coefficient()` MIC
- `NonlinearAnalyzer` analyzer class

#### CLI Tools

- `pycorrana analyze` - complete analysis
- `pycorrana clean` - data cleaning
- `pycorrana partial` - partial correlation analysis
- `pycorrana nonlinear` - nonlinear detection
- `pycorrana-interactive` - interactive tool

#### Sample Datasets

- Iris dataset
- Titanic dataset
- Wine dataset
- Simulated data generator

[0.1.5]: https://github.com/sidneylyzhang/pycorrana/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/sidneylyzhang/pycorrana/releases/tag/v0.1.0
