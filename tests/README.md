# Test Suite for Tourist Trip Optimizer

This directory contains comprehensive unit and integration tests for the Tourist Trip Optimizer project.

## Test Files

- **test_data_utils.py**: Unit tests for data loading, processing, and validation functions
- **test_ga_core.py**: Unit tests for the Genetic Algorithm implementation
- **test_visualization.py**: Unit tests for visualization and reporting functions
- **test_integration.py**: Integration tests for end-to-end workflows

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run tests with verbose output:
```bash
pytest tests/ -v
```

### Run tests with coverage report:
```bash
pytest tests/ --cov=scripts --cov-report=html
```

### Run specific test file:
```bash
pytest tests/test_data_utils.py -v
```

### Run specific test class:
```bash
pytest tests/test_ga_core.py::TestFitnessFunction -v
```

### Run specific test method:
```bash
pytest tests/test_ga_core.py::TestFitnessFunction::test_fitness_valid_tour -v
```

## Test Coverage

The test suite includes:
- **18 tests** for data_utils module
- **21 tests** for ga_core module  
- **18 tests** for visualization module
- **9 tests** for integration workflows

Total: **66 tests** covering all core functionality

## Test Categories

### Unit Tests
- Data loading and validation
- Distance matrix calculations
- Travel time calculations
- Individual and population creation
- Genetic operators (selection, crossover, mutation)
- Fitness function
- Tour validation
- Visualization functions
- Report generation

### Integration Tests
- Complete GA pipeline with real data
- Complete GA pipeline with synthetic data
- Data → optimization → visualization workflow
- Results validation and constraints
- Edge cases (single attraction, tight/loose time constraints)
- Module compatibility

## Requirements

All test dependencies are listed in `requirements.txt`:
- pytest
- pytest-cov
- pandas
- numpy
- matplotlib
- folium
- haversine

Install with:
```bash
pip install -r requirements.txt
```
