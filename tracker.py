import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, track_id, init_pos, max_age):
        self.id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([init_pos[0], init_pos[1], 0, 0])  # [x, y, vx, vy]

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
        self.last_cameras = set([self.cam_id]) if self.cam_id else set()
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
            self.last_cameras.add(pos[2])

class MultiCameraTracker:
    def __init__(self, max_age=30, dist_threshold=1.5, camera_overlap_threshold=1.2):
        self.max_age = max_age
        self.dist_threshold = dist_threshold
        self.camera_overlap_threshold = camera_overlap_threshold
        self.next_id = 1
        self.tracks = []

    def _calculate_cost_matrix(self, tracks, detections):
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, trk in enumerate(tracks):
            for j, det in enumerate(detections):
                pos_diff = np.linalg.norm(trk.kf.x[:2] - det[:2])
                cam_penalty = 1.0
                if len(det) > 2 and trk.cam_id is not None:
                    if det[2] != trk.cam_id:
                        if pos_diff < self.camera_overlap_threshold * 1.5:
                            cam_penalty = 1.0
                        else:
                            cam_penalty = 3.0
                cost_matrix[i, j] = pos_diff * cam_penalty
        return cost_matrix

    def _group_detections(self, detections, eps=0.5):
        grouped = []
        used = set()
        for i, a in enumerate(detections):
            if i in used:
                continue
            group = [a]
            for j, b in enumerate(detections):
                if j <= i or j in used:
                    continue
                if np.linalg.norm(np.array(a[:2]) - np.array(b[:2])) < eps:
                    group.append(b)
                    used.add(j)
            xs = [g[0] for g in group]
            ys = [g[1] for g in group]
            cams = [g[2] for g in group]
            dom_cam = max(set(cams), key=cams.count)
            grouped.append((np.mean(xs), np.mean(ys), dom_cam))
        return grouped

    def update(self, detections):
        detections = self._group_detections(detections)
        for trk in self.tracks:
            trk.predict()
        self.tracks = [trk for trk in self.tracks if trk.age <= self.max_age]

        if not detections:
            return self.tracks

        if not self.tracks:
            for det in detections:
                self.tracks.append(Track(self.next_id, det, self.max_age))
                self.next_id += 1
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

        for i, det in enumerate(detections):
            if i not in assigned_dets:
                self.tracks.append(Track(self.next_id, det, self.max_age))
                self.next_id += 1

        return self.tracks
