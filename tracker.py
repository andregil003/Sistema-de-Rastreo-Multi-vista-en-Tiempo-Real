import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, track_id, init_pos, max_age):
        self.id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([init_pos[0], init_pos[1], 0, 0])

        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.kf.P = np.eye(4) * 100
        self.kf.R = np.eye(2) * 5
        self.kf.Q = np.eye(4) * 0.1

        self.age = 0
        self.max_age = max_age
        self.history = []
        self.cam_id = init_pos[2] if len(init_pos) > 2 else None
        self.last_cameras = set([self.cam_id]) if self.cam_id is not None else set()
        self.color = tuple(np.random.randint(0, 255, 3).tolist())

    def predict(self):
        self.kf.predict()
        self.age += 1
        return self.kf.x[:2]

    def update(self, pos):
        self.kf.update(np.array(pos[:2]))
        self.history.append((pos[0], pos[1]))
        self.age = 0
        if len(pos) > 2:
            self.cam_id = pos[2]
            self.last_cameras.add(self.cam_id)

class MultiCameraTracker:
    def __init__(self, max_age=30, dist_threshold=1.5, camera_overlap_threshold=1.2):
        self.max_age = max_age
        self.dist_threshold = dist_threshold
        self.camera_overlap_threshold = camera_overlap_threshold
        self.next_id = 1
        self.tracks = []
        self.active_track_ids = set()

    def _calculate_cost_matrix(self, tracks, detections):
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, trk in enumerate(tracks):
            for j, det in enumerate(detections):
                pos_diff = np.linalg.norm(trk.kf.x[:2] - det[:2])
                cam_penalty = 1.0
                if len(det) > 2 and trk.cam_id is not None:
                    if det[2] != trk.cam_id:
                        if pos_diff < self.camera_overlap_threshold:
                            cam_penalty = 0.8
                        else:
                            cam_penalty = 2.0
                cost_matrix[i, j] = pos_diff * cam_penalty
        return cost_matrix

    def _group_detections(self, detections, eps=0.5):
        if not detections:
            return []

        detections_array = np.array(detections)
        grouped = []
        processed = np.zeros(len(detections), dtype=bool)

        for i in range(len(detections)):
            if processed[i]:
                continue

            det_i = detections_array[i]
            cam_i = det_i[2] if len(det_i) > 2 else None
            group_indices = [i]

            for j in range(i + 1, len(detections)):
                if processed[j]:
                    continue

                det_j = detections_array[j]
                cam_j = det_j[2] if len(det_j) > 2 else None

                if cam_i != cam_j and cam_i is not None and cam_j is not None:
                    dist = np.linalg.norm(det_i[:2] - det_j[:2])
                    if dist < eps:
                        group_indices.append(j)
                        processed[j] = True

            processed[i] = True

            if len(group_indices) > 1:
                group_dets = detections_array[group_indices]
                avg_pos = np.mean(group_dets[:, :2], axis=0)
                cams = [int(detections_array[idx][2]) for idx in group_indices]
                dom_cam = max(set(cams), key=cams.count)
                grouped.append((avg_pos[0], avg_pos[1], dom_cam))
            else:
                grouped.append(tuple(det_i))

        return grouped

    def update(self, detections):
        detections = self._group_detections(detections)

        for trk in self.tracks:
            trk.predict()

        active_tracks = []
        for trk in self.tracks:
            if trk.age <= self.max_age:
                active_tracks.append(trk)
            else:
                self.active_track_ids.discard(trk.id)
        self.tracks = active_tracks

        if not detections:
            return self.tracks

        if not self.tracks:
            for det in detections:
                new_id = self._get_next_id()
                self.tracks.append(Track(new_id, det, self.max_age))
                self.active_track_ids.add(new_id)
            return self.tracks

        cost_matrix = self._calculate_cost_matrix(self.tracks, detections)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.dist_threshold:
                self.tracks[r].update(detections[c])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Intentar reutilizar tracks para detecciones no asignadas
        for i, det in enumerate(detections):
            if i in assigned_dets:
                continue

            reused = False
            for trk in self.tracks:
                if trk.cam_id != det[2] and np.linalg.norm(trk.kf.x[:2] - np.array(det[:2])) < self.camera_overlap_threshold:
                    trk.update(det)
                    reused = True
                    break

            if not reused:
                new_id = self._get_next_id()
                self.tracks.append(Track(new_id, det, self.max_age))
                self.active_track_ids.add(new_id)

        return self.tracks

    def _get_next_id(self):
        while self.next_id in self.active_track_ids:
            self.next_id += 1
        return self.next_id
