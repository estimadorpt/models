# Electoral System Abstraction

This document describes the electoral system abstraction implemented to support both parliamentary (D'Hondt) and municipal (mayoral) elections.

## Overview

The electoral system abstraction provides a flexible framework for different types of elections:

- **Parliamentary Elections**: Use proportional representation (D'Hondt method)
- **Municipal Elections**: Use first-past-the-post for mayoral races

## Architecture

### Base Interface

All electoral systems implement the `BaseElectoralSystem` interface:

```python
from abc import ABC, abstractmethod

class BaseElectoralSystem(ABC):
    @abstractmethod
    def allocate_seats(self, votes_dict: Dict[str, int], num_seats: int) -> Dict[str, int]:
        """Allocates seats based on vote counts."""
        pass

    @abstractmethod
    def supports_independents(self) -> bool:
        """Returns True if this system supports independent candidates."""
        pass

    @abstractmethod
    def get_system_name(self) -> str:
        """Returns the name of this electoral system."""
        pass
```

### Available Systems

#### 1. D'Hondt System (`DHondtSystem`)

- **Use case**: Portuguese parliamentary elections
- **Method**: Proportional representation using D'Hondt method
- **Seats**: Multiple seats per district
- **Independents**: Supported (treated as single-candidate parties)

#### 2. Mayoral System (`MayoralSystem`)

- **Use case**: Portuguese municipal mayoral elections
- **Method**: First-past-the-post (plurality winner)
- **Seats**: Single position (mayor)
- **Independents**: Fully supported

## Usage

### Factory Pattern

Create electoral systems using the factory function:

```python
from src.processing.electoral_systems import create_electoral_system

# Parliamentary election
dhondt_system = create_electoral_system('dhondt')

# Mayoral election
mayoral_system = create_electoral_system('mayoral')

# Mayoral with custom runoff threshold
mayoral_system = create_electoral_system('mayoral', runoff_threshold=0.4)
```

### Seat Allocation

```python
# Example votes
votes = {'PS': 5000, 'AD': 3000, 'CH': 2000}

# Parliamentary allocation (proportional)
dhondt_result = dhondt_system.allocate_seats(votes, 10)
# Result: {'PS': 5, 'AD': 3, 'CH': 2}

# Mayoral allocation (winner-take-all)
mayoral_result = mayoral_system.allocate_seats(votes, 1)
# Result: {'PS': 1, 'AD': 0, 'CH': 0}
```

### Seat Prediction Integration

The seat prediction module supports both systems:

```python
from src.processing.seat_prediction import calculate_seat_predictions_with_system

# Parliamentary election prediction
parliamentary_seats = calculate_seat_predictions_with_system(
    national_forecast_shares=shares,
    last_election_date='2024-03-10',
    political_families=['PS', 'AD', 'CH', 'IL', 'BE'],
    election_dates=election_dates,
    electoral_system_type='dhondt'
)

# Mayoral election prediction
mayoral_results = calculate_seat_predictions_with_system(
    national_forecast_shares=shares,
    last_election_date='2024-03-10',
    political_families=['PS', 'AD', 'CH', 'IL', 'BE'],
    election_dates=election_dates,
    electoral_system_type='mayoral'
)
```

## Backward Compatibility

All existing functions continue to work unchanged:

```python
# This still works exactly as before
result = calculate_dhondt(votes, seats)

# This also works unchanged
result = calculate_seat_predictions(shares, date, parties, dates)
```

## System Properties

| System | Proportional | Winner-Take-All | Independents | Multiple Seats |
|--------|-------------|-----------------|--------------|----------------|
| D'Hondt | ✅ | ❌ | ✅ | ✅ |
| Mayoral | ❌ | ✅ | ✅ | ❌ |

## Testing

Comprehensive tests ensure:

- **Backward compatibility**: Existing `calculate_dhondt()` behavior unchanged
- **System correctness**: Each system produces expected results
- **Interface compliance**: All systems implement the same interface
- **Edge case handling**: Proper handling of ties, zero votes, etc.

## Future Extensions

The framework can easily be extended with new systems:

```python
class RunoffMayoralSystem(BaseElectoralSystem):
    """Mayoral system with runoff if no majority."""

    def allocate_seats(self, votes_dict, num_seats):
        # Implementation for runoff system
        pass
```

Add to factory:

```python
def create_electoral_system(system_type: str, **kwargs):
    if system_type.lower() == 'runoff_mayoral':
        return RunoffMayoralSystem(**kwargs)
    # ... existing systems
```

## Implementation Details

### D'Hondt Algorithm

The D'Hondt method allocates seats proportionally by:

1. Calculate quotients: `votes / (seats_allocated + 1)`
2. Award seat to party with highest quotient
3. Repeat until all seats allocated
4. Handle ties by favoring party with more total votes

### Mayoral Algorithm

The mayoral system is simpler:

1. Find candidate with most votes
2. Award position to that candidate
3. All other candidates get 0

### Performance

- **No performance regression**: New system has same performance as original
- **Memory efficient**: Objects created only when needed
- **Extensible**: Adding new systems doesn't affect existing performance

## Configuration

Future versions may support configuration files:

```yaml
# electoral_systems.yml
parliamentary:
  system: dhondt
  geographic_unit: district

municipal:
  mayor:
    system: mayoral
    geographic_unit: municipality
    runoff_threshold: 0.5
```

This abstraction provides a solid foundation for supporting Portugal's diverse electoral landscape while maintaining full backward compatibility with existing code.