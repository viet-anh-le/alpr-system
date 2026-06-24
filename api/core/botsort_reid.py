"""BoT-SORT variant that applies ReID in both association passes.

BoxMOT's default BotSort extracts appearance features only for high-confidence
detections. Low-confidence detections in the second association pass are matched
with IoU only. For vehicle tracking in this project, we want the custom ReID
model to participate in both passes so IDs stay stable when detector confidence
drops briefly.
"""

from __future__ import annotations

import numpy as np

from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.botsort_utils import joint_stracks
from boxmot.utils.matching import embedding_distance, iou_distance, linear_assignment


class AlwaysReIDBotSort(BotSort):
    """BotSort with ReID features available for high- and low-score detections."""

    def _confidence_masks(self, dets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        confs = self.detection_layout.confidences(dets)
        first_mask = confs > self.track_high_thresh
        second_mask = np.logical_and(
            confs > self.track_low_thresh,
            confs < self.track_high_thresh,
        )
        return first_mask, second_mask

    def _split_detections(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
        dets = self.detection_layout.with_detection_indices(dets)
        first_mask, second_mask = self._confidence_masks(dets)

        dets_first = dets[first_mask]
        dets_second = dets[second_mask]
        embs_first = embs[first_mask] if embs is not None else None
        embs_second = embs[second_mask] if embs is not None else None
        return dets, dets_first, embs_first, dets_second, embs_second

    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray | None = None,
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        self.kalman_filter = KalmanFilterXYWH(ndim=self._kalman_ndim())
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        dets, dets_first, embs_first, dets_second, embs_second = self._split_detections(
            dets,
            embs,
        )

        if self.with_reid and embs is None:
            first_mask, second_mask = self._confidence_masks(dets)
            features = np.asarray(
                self.model.get_features(self._detection_boxes(dets), img)
            )
            features_high = features[first_mask]
            features_second = features[second_mask]
        else:
            features_high = embs_first if embs_first is not None else []
            features_second = embs_second if embs_second is not None else []

        detections = self._create_detections(dets_first, features_high)
        unconfirmed, active_tracks = self._separate_tracks()
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        _, u_track_first, u_detection_first = self._first_association(
            dets,
            dets_first,
            active_tracks,
            unconfirmed,
            img,
            detections,
            activated_stracks,
            refind_stracks,
            strack_pool,
        )

        self._second_association(
            dets_second,
            features_second,
            activated_stracks,
            lost_stracks,
            refind_stracks,
            u_track_first,
            strack_pool,
        )

        _, _, u_detection_unc = self._handle_unconfirmed_tracks(
            u_detection_first,
            detections,
            activated_stracks,
            removed_stracks,
            unconfirmed,
        )

        self._initialize_new_tracks(
            u_detection_unc,
            activated_stracks,
            [detections[i] for i in u_detection_first],
        )

        self._update_track_states(removed_stracks)
        return self._prepare_output(
            activated_stracks,
            refind_stracks,
            lost_stracks,
            removed_stracks,
        )

    def _second_association(
        self,
        dets_second: np.ndarray,
        features_second: np.ndarray,
        activated_stracks: list[STrack],
        lost_stracks: list[STrack],
        refind_stracks: list[STrack],
        u_track_first: np.ndarray,
        strack_pool: list[STrack],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        detections_second = self._create_detections(dets_second, features_second)
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track_first
            if strack_pool[i].state == TrackState.Tracked
        ]

        ious_dists = iou_distance(
            r_tracked_stracks,
            detections_second,
            is_obb=self.is_obb,
        )

        if self.with_reid:
            ious_dists_mask = ious_dists > self.proximity_thresh
            emb_dists = embedding_distance(r_tracked_stracks, detections_second)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists,
            thresh=self.match_thresh,
        )

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        return matches, u_track, u_detection
