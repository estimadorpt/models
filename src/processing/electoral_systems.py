import pandas as pd
import numpy as np
from typing import Dict

def calculate_dhondt(votes_dict: Dict[str, int], num_seats: int) -> Dict[str, int]:
    """
    Calculates the allocation of seats using the D'Hondt method.

    Args:
        votes_dict (Dict[str, int]): A dictionary where keys are party names 
                                     and values are the number of votes they received.
        num_seats (int): The total number of seats to allocate for the district.

    Returns:
        Dict[str, int]: A dictionary where keys are party names and values are the 
                        number of seats allocated to them.
    """
    if not isinstance(votes_dict, dict) or not votes_dict:
        print("Warning: Invalid votes_dict provided to calculate_dhondt. Returning empty allocation.")
        return {}
    if not isinstance(num_seats, int) or num_seats <= 0:
        print(f"Warning: Invalid num_seats ({num_seats}) provided to calculate_dhondt. Returning empty allocation.")
        return {}

    # Filter out parties with zero votes to avoid division by zero issues later
    valid_votes = {party: votes for party, votes in votes_dict.items() if isinstance(votes, (int, float)) and votes > 0}
    if not valid_votes:
        print("Warning: No parties with positive votes found in calculate_dhondt. Returning empty allocation.")
        return {party: 0 for party in votes_dict}
    
    parties = list(valid_votes.keys())
    votes = np.array(list(valid_votes.values()), dtype=float) # Use float for calculations
    seats = {party: 0 for party in parties}
    quotients = pd.Series(index=parties, dtype=float)

    for _ in range(num_seats):
        # Calculate quotients: votes / (seats_allocated + 1)
        for party in parties:
            quotients[party] = valid_votes[party] / (seats[party] + 1)
        
        # Find the party with the highest quotient
        if quotients.empty or quotients.max() <= 0: 
            # Handle cases where no seats can be allocated (e.g., all quotients are zero)
            print("Warning: Could not allocate remaining seats due to zero quotients.")
            break 
            
        # In case of ties, D'Hondt favors the party with more votes overall.
        # If votes are also tied, the standard doesn't specify; we can break ties arbitrarily (e.g., first party in list).
        max_quotient = quotients.max()
        winners = quotients[quotients == max_quotient]
        
        if len(winners) > 1:
            # Tie-breaking: favor party with more total votes
            winner_votes = pd.Series({p: valid_votes[p] for p in winners.index})
            winner = winner_votes.idxmax() # idxmax() selects the first if votes are also tied
            print(f"Tie detected for seat allocation (quotient={max_quotient}). Winners: {winners.index.tolist()}. Selected {winner} based on higher total votes ({winner_votes.max()}).")
        else:
            winner = winners.idxmax()
            
        # Allocate seat to the winner
        seats[winner] += 1
        # Update quotient for the winner (though it's recalculated next loop)
        # quotients[winner] = valid_votes[winner] / (seats[winner] + 1)

    # Include parties that received 0 votes in the final output with 0 seats
    final_seats = {party: 0 for party in votes_dict.keys()}
    final_seats.update(seats)
    
    # Verification: Ensure total allocated seats doesn't exceed num_seats
    allocated_seats_count = sum(final_seats.values())
    if allocated_seats_count > num_seats:
         print(f"Error: Allocated {allocated_seats_count} seats, but only {num_seats} were available!")
         # This case should ideally not happen with correct logic, but as a fallback:
         # Potentially return the state just before the last allocation or an error state.
         # For now, just print error and return the potentially incorrect allocation.

    return final_seats 