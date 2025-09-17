"""
Tests for electoral system abstractions.

Tests both the new class-based electoral systems and backward compatibility
with the existing calculate_dhondt function.
"""

import pytest
from src.processing.electoral_systems import (
    BaseElectoralSystem,
    DHondtSystem,
    MayoralSystem,
    create_electoral_system,
    calculate_dhondt
)


class TestDHondtSystem:
    """Test the D'Hondt system implementation."""

    def test_dhondt_basic_allocation(self):
        """Test basic D'Hondt seat allocation."""
        system = DHondtSystem()
        votes = {'PS': 1000, 'AD': 800, 'CH': 600, 'IL': 400, 'BE': 200}
        result = system.allocate_seats(votes, 10)

        expected = {'PS': 4, 'AD': 3, 'CH': 2, 'IL': 1, 'BE': 0}
        assert result == expected

    def test_dhondt_backward_compatibility(self):
        """Test that new system matches old function."""
        system = DHondtSystem()
        votes = {'PS': 1000, 'AD': 800, 'CH': 600}

        old_result = calculate_dhondt(votes, 5)
        new_result = system.allocate_seats(votes, 5)

        assert old_result == new_result

    def test_dhondt_properties(self):
        """Test D'Hondt system properties."""
        system = DHondtSystem()
        assert system.supports_independents() == True
        assert system.get_system_name() == "D'Hondt"

    def test_dhondt_edge_cases(self):
        """Test D'Hondt edge cases."""
        system = DHondtSystem()

        # Empty votes
        assert system.allocate_seats({}, 5) == {}

        # Zero seats
        result = system.allocate_seats({'A': 100, 'B': 50}, 0)
        assert result == {'A': 0, 'B': 0}

        # Single party
        result = system.allocate_seats({'A': 100}, 3)
        assert result == {'A': 3}


class TestMayoralSystem:
    """Test the mayoral system implementation."""

    def test_mayoral_basic_allocation(self):
        """Test basic mayoral election."""
        system = MayoralSystem()
        votes = {'Candidate A': 5000, 'Candidate B': 3000, 'Candidate C': 2000}
        result = system.allocate_seats(votes, 1)

        expected = {'Candidate A': 1, 'Candidate B': 0, 'Candidate C': 0}
        assert result == expected

    def test_mayoral_properties(self):
        """Test mayoral system properties."""
        system = MayoralSystem()
        assert system.supports_independents() == True
        assert system.get_system_name() == "First-Past-The-Post (Mayoral)"

    def test_mayoral_tie_handling(self):
        """Test mayoral system with tied votes."""
        system = MayoralSystem()
        votes = {'A': 1000, 'B': 1000, 'C': 500}
        result = system.allocate_seats(votes, 1)

        # Should pick one of the tied candidates (deterministic based on max())
        winner = max(votes, key=votes.get)
        assert result[winner] == 1
        assert sum(result.values()) == 1

    def test_mayoral_edge_cases(self):
        """Test mayoral edge cases."""
        system = MayoralSystem()

        # Empty votes
        assert system.allocate_seats({}, 1) == {}

        # Single candidate
        result = system.allocate_seats({'A': 100}, 1)
        assert result == {'A': 1}

        # Zero votes
        result = system.allocate_seats({'A': 0, 'B': 0}, 1)
        assert result == {'A': 0, 'B': 0}

    def test_mayoral_multiple_seats_warning(self):
        """Test that mayoral system warns about multiple seats."""
        system = MayoralSystem()
        votes = {'A': 100, 'B': 50}

        # Should still work but treat as 1 seat
        result = system.allocate_seats(votes, 3)
        assert result == {'A': 1, 'B': 0}


class TestElectoralSystemFactory:
    """Test the electoral system factory."""

    def test_create_dhondt_system(self):
        """Test creating D'Hondt system via factory."""
        system = create_electoral_system('dhondt')
        assert isinstance(system, DHondtSystem)
        assert system.get_system_name() == "D'Hondt"

    def test_create_mayoral_system(self):
        """Test creating mayoral system via factory."""
        system = create_electoral_system('mayoral')
        assert isinstance(system, MayoralSystem)
        assert system.get_system_name() == "First-Past-The-Post (Mayoral)"

    def test_create_mayoral_with_parameters(self):
        """Test creating mayoral system with parameters."""
        system = create_electoral_system('mayoral', runoff_threshold=0.4)
        assert isinstance(system, MayoralSystem)
        assert system.runoff_threshold == 0.4

    def test_create_unknown_system(self):
        """Test error handling for unknown system types."""
        with pytest.raises(ValueError, match="Unknown electoral system type"):
            create_electoral_system('unknown')

    def test_case_insensitive_creation(self):
        """Test that system creation is case-insensitive."""
        system1 = create_electoral_system('DHONDT')
        system2 = create_electoral_system('Mayoral')

        assert isinstance(system1, DHondtSystem)
        assert isinstance(system2, MayoralSystem)


class TestBackwardCompatibility:
    """Test backward compatibility with existing API."""

    def test_calculate_dhondt_function_unchanged(self):
        """Test that calculate_dhondt function behavior is unchanged."""
        votes = {'PS': 1000, 'AD': 800, 'CH': 600, 'IL': 400}
        result = calculate_dhondt(votes, 8)

        # This should match known behavior
        assert isinstance(result, dict)
        assert all(isinstance(v, int) for v in result.values())
        assert sum(result.values()) <= 8

    def test_calculate_dhondt_edge_cases(self):
        """Test that edge cases still work with old function."""
        # These tests ensure existing code continues to work
        assert calculate_dhondt({}, 5) == {}
        assert calculate_dhondt({'A': 100}, 0) == {'A': 0}

        result = calculate_dhondt({'A': 0, 'B': 0}, 3)
        assert result == {'A': 0, 'B': 0}


class TestSystemComparison:
    """Test comparing different electoral systems."""

    def test_dhondt_vs_mayoral_different_outcomes(self):
        """Test that D'Hondt and mayoral systems give different results."""
        votes = {'A': 400, 'B': 350, 'C': 250}

        dhondt = create_electoral_system('dhondt')
        mayoral = create_electoral_system('mayoral')

        dhondt_result = dhondt.allocate_seats(votes, 3)
        mayoral_result = mayoral.allocate_seats(votes, 1)

        # D'Hondt should distribute seats proportionally
        assert dhondt_result['A'] >= 1
        assert dhondt_result['B'] >= 1
        assert sum(dhondt_result.values()) == 3

        # Mayoral should be winner-take-all
        assert mayoral_result['A'] == 1
        assert mayoral_result['B'] == 0
        assert mayoral_result['C'] == 0

    def test_system_interfaces_consistent(self):
        """Test that all systems implement the same interface."""
        systems = [
            create_electoral_system('dhondt'),
            create_electoral_system('mayoral')
        ]

        for system in systems:
            assert isinstance(system, BaseElectoralSystem)
            assert hasattr(system, 'allocate_seats')
            assert hasattr(system, 'supports_independents')
            assert hasattr(system, 'get_system_name')

            # Test interface methods work
            assert isinstance(system.supports_independents(), bool)
            assert isinstance(system.get_system_name(), str)