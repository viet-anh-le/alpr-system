"""
core/association.py — Trajectory Matching for Plate-to-Vehicle Association.

Instead of matching plates to vehicles frame-by-frame (which is susceptible to
occlusion and bounding-box jitter), this module tracks plates independently and
uses a Voting mechanism over N frames to permanently bind a Plate Track to a Vehicle Track.
"""

from __future__ import annotations

import collections
import numpy as np


class TrajectoryAssociator:
    def __init__(self, match_frames: int = 3, agreement_ratio: float = 0.6):
        """
        match_frames: Number of frames to observe a plate before locking its association.
        agreement_ratio: The percentage of frames the plate must vote for the same vehicle.
        """
        self.match_frames = match_frames
        self.agreement_ratio = agreement_ratio

        # p_tid -> list of v_tid votes
        self.plate_votes: dict[int, list[int]] = collections.defaultdict(list)

        # Final locked matches: p_tid -> v_tid
        self.plate_to_vehicle: dict[int, int] = {}

        # Cache the last known bounding box of a vehicle to handle momentary track loss
        self.vehicle_cache: dict[int, tuple[int, int, int, int]] = {}

    def process_frame(
        self,
        plate_tracks: list[dict],
        vehicle_tracks: list[dict],
    ) -> list[tuple[int, dict]]:
        """
        Processes a single frame.

        plate_tracks: list of dicts with keys: 'id', 'box' (x1, y1, x2, y2), 'crop', 'det_conf', 'pts'
        vehicle_tracks: list of dicts with keys: 'id', 'box' (x1, y1, x2, y2)

        Returns:
            list of tuples: (v_tid, plate_data) for plates that are firmly associated.
        """
        # Update vehicle cache
        for v in vehicle_tracks:
            self.vehicle_cache[v["id"]] = v["box"]

        firm_matches = []

        for p in plate_tracks:
            p_tid = p["id"]

            # If already firmly associated, just return the match (if vehicle still exists)
            if p_tid in self.plate_to_vehicle:
                v_tid = self.plate_to_vehicle[p_tid]
                firm_matches.append((v_tid, p))
                continue

            # Frame-level voting using Area Heuristic
            cx = (p["box"][0] + p["box"][2]) / 2
            cy = (p["box"][1] + p["box"][3]) / 2

            best_v_tid = None
            best_score = float("inf")

            for v in vehicle_tracks:
                x1, y1, x2, y2 = v["box"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if area < best_score:
                        best_score = area
                        best_v_tid = v["id"]

            if best_v_tid is not None:
                self.plate_votes[p_tid].append(best_v_tid)

                # Check if we have gathered enough votes to lock the association
                votes = self.plate_votes[p_tid]
                if len(votes) >= self.match_frames:
                    # Look at the most recent N votes
                    recent_votes = votes[-self.match_frames :]
                    counter = collections.Counter(recent_votes)
                    most_common_v_tid, count = counter.most_common(1)[0]

                    if count >= self.match_frames * self.agreement_ratio:
                        # Lock association!
                        self.plate_to_vehicle[p_tid] = most_common_v_tid
                        firm_matches.append((most_common_v_tid, p))

        return firm_matches
