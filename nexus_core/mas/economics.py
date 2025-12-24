"""
Economic Mechanisms (Layer 2)
=============================

This module implements the Game Theoretic mechanisms for airspace allocation.

Mechanism: Vickrey Auction (Second-Price Sealed-Bid Auction).
Unlike FCFS (First-Come-First-Serve), this allocates resources to agents 
with the highest economic valuation (e.g., emergency medical transport vs. leisure drone).

Mathematical Formulation:
Let $N = \{1, ..., n\}$ be the set of bidders.
Let $b_i$ be the bid of agent $i$.
The winner $i^*$ is determined by:
$$ i^* = \operatorname{argmax}_{i \in N} b_i $$

The payment $P$ is determined by:
$$ P = \max_{j \neq i^*} b_j $$

Rationale:
In a private value auction, truthful bidding is a dominant strategy in a Vickrey auction.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import heapq

@dataclass
class Bid:
    agent_id: str
    amount: float
    timestamp: float
    urgency_level: int = 1  # 1: Normal, 5: Critical/Emergency

class AuctionResult:
    def __init__(self, winner_id: Optional[str], payment_price: float, all_bids: List[Bid]):
        self.winner_id = winner_id
        self.payment_price = payment_price
        self.all_bids = all_bids

class VickreyAuctioneer:
    """
    Manages the dynamic allocation of voxel space via auctions.
    """
    def __init__(self, reserve_price: float = 1.0):
        self.reserve_price = reserve_price
        # Using a heap to efficiently retrieve top bids
        # Python's heapq is a min-heap, so we store negative values for max extraction
        self._bid_queue: List[Tuple[float, Bid]] = []

    def submit_bid(self, bid: Bid):
        """
        Accepts a sealed bid from an agent.
        """
        if bid.amount < self.reserve_price:
            return # Ignore bids below reserve
        
        # Priority boost for emergency flights (Hybrid mechanism)
        # We artificially boost the economic bid for urgency to ensure safety-critical ops win
        # Effective Bid = Bid * (1 + 0.5 * Urgency)
        effective_bid = bid.amount * (1 + 0.2 * (bid.urgency_level - 1))
        
        # Store as negative for min-heap to act as max-heap
        heapq.heappush(self._bid_queue, (-effective_bid, bid))

    def resolve(self) -> AuctionResult:
        """
        Clears the auction and determines the winner and payment.
        """
        if not self._bid_queue:
            return AuctionResult(None, 0.0, [])

        # Get Highest Bidder
        neg_b1, winner_bid = heapq.heappop(self._bid_queue)
        
        # Determine Payment (Second Highest Price)
        payment = self.reserve_price
        
        if self._bid_queue:
            neg_b2, second_bid = heapq.heappop(self._bid_queue)
            # In standard Vickrey, payment is the raw second highest bid
            # We revert the "Effective Bid" calculation to get real token cost if needed, 
            # but for standard Vickrey, we usually use the sorting metric.
            # Here we assume the payment is the base amount of the second highest *effective* rank
            payment = second_bid.amount 
        
        # Reconstruct bid list for logging/transparency
        all_bids = [winner_bid]
        if 'second_bid' in locals():
            all_bids.append(second_bid)
        while self._bid_queue:
            _, b = heapq.heappop(self._bid_queue)
            all_bids.append(b)

        return AuctionResult(
            winner_id=winner_bid.agent_id,
            payment_price=payment,
            all_bids=all_bids
        )

    def reset(self):
        self._bid_queue = []

