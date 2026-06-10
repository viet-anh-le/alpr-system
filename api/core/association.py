"""
core/association.py — Trajectory Matching for Plate-to-Vehicle Association.

Instead of matching plates to vehicles frame-by-frame (which is susceptible to
occlusion and bounding-box jitter), this module tracks plates independently and
uses a voting mechanism over N frames to bind a plate track to a vehicle track.
Locked matches are revalidated against source vehicle and geometry each frame.
"""

from __future__ import annotations

import collections


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _box_center(box: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_area(box: list[int] | tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _point_inside_box(
    point: tuple[float, float],
    box: list[int] | tuple[int, int, int, int],
    *,
    margin_ratio: float = 0.0,
) -> bool:
    x1, y1, x2, y2 = box
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    cx, cy = point
    return (x1 - margin_x) <= cx <= (x2 + margin_x) and (y1 - margin_y) <= cy <= (y2 + margin_y)


class TrajectoryAssociator:
    def __init__(
        self,
        match_frames: int = 5,
        agreement_ratio: float = 0.6,
        *,
        lock_conflict_frames: int = 2,
        lock_margin_ratio: float = 0.08,
    ):
        """
        match_frames: Number of frames to observe a plate before locking its association.
        agreement_ratio: The percentage of frames the plate must vote for the same vehicle.
        """
        self.match_frames = match_frames
        self.agreement_ratio = agreement_ratio
        self.lock_conflict_frames = max(1, lock_conflict_frames)
        self.lock_margin_ratio = lock_margin_ratio

        # p_tid -> list of v_tid votes
        self.plate_votes: dict[int, list[int]] = collections.defaultdict(list)

        # Locked matches: p_tid -> v_tid. Revalidated on every visible plate frame.
        self.plate_to_vehicle: dict[int, int] = {}

        # p_tid -> consecutive frames that contradict the locked vehicle.
        self.lock_conflicts: dict[int, int] = collections.defaultdict(int)

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
        visible_vehicles = {_optional_int(v.get("id")): v for v in vehicle_tracks}
        visible_vehicles = {v_tid: v for v_tid, v in visible_vehicles.items() if v_tid is not None}

        # Update vehicle cache
        for v in vehicle_tracks:
            v_tid = _optional_int(v.get("id"))
            if v_tid is not None:
                self.vehicle_cache[v_tid] = tuple(int(coord) for coord in v["box"])

        firm_matches = []

        for p in plate_tracks:
            p_tid = _optional_int(p.get("id"))
            if p_tid is None:
                continue

            locked_v_tid = self.plate_to_vehicle.get(p_tid)
            if locked_v_tid is not None:
                if self._is_locked_match_valid(p, locked_v_tid, visible_vehicles):
                    self.lock_conflicts.pop(p_tid, None)
                    firm_matches.append((locked_v_tid, p))
                    continue

                self.lock_conflicts[p_tid] += 1
                if self.lock_conflicts[p_tid] < self.lock_conflict_frames:
                    continue

                self.plate_to_vehicle.pop(p_tid, None)
                self.plate_votes.pop(p_tid, None)
                self.lock_conflicts.pop(p_tid, None)

            best_v_tid = self._best_vehicle_for_plate(p, visible_vehicles)

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
                        self.lock_conflicts.pop(p_tid, None)
                        firm_matches.append((most_common_v_tid, p))

        return firm_matches

    def _is_locked_match_valid(
        self,
        plate: dict,
        locked_v_tid: int,
        visible_vehicles: dict[int, dict],
    ) -> bool:
        locked_vehicle = visible_vehicles.get(locked_v_tid)
        if locked_vehicle is None:
            return False

        source_v_tid = _optional_int(plate.get("source_vehicle_id"))
        if (
            source_v_tid is not None
            and source_v_tid != locked_v_tid
            and source_v_tid in visible_vehicles
        ):
            return False

        return _point_inside_box(
            _box_center(plate["box"]),
            locked_vehicle["box"],
            margin_ratio=self.lock_margin_ratio,
        )

    def _best_vehicle_for_plate(
        self,
        plate: dict,
        visible_vehicles: dict[int, dict],
    ) -> int | None:
        center = _box_center(plate["box"])
        source_v_tid = _optional_int(plate.get("source_vehicle_id"))
        source_vehicle = visible_vehicles.get(source_v_tid) if source_v_tid is not None else None

        if source_vehicle is not None and _point_inside_box(
            center,
            source_vehicle["box"],
            margin_ratio=self.lock_margin_ratio,
        ):
            return source_v_tid

        best_v_tid: int | None = None
        best_area = float("inf")
        for v_tid, vehicle in visible_vehicles.items():
            if _point_inside_box(center, vehicle["box"]):
                area = _box_area(vehicle["box"])
                if area < best_area:
                    best_area = area
                    best_v_tid = v_tid

        return best_v_tid
