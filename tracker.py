import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, track_id, init_pos, max_age=30):
        self.id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([init_pos[0], init_pos[1], 0, 0])
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.P *= 10.
        self.kf.R *= 5.
        self.kf.Q = np.eye(4)
        self.age = 0
        self.max_age = max_age
        self.history = []

    def predict(self):
        self.kf.predict()
        self.age += 1
        return self.kf.x[:2]

    def update(self, pos):
        self.kf.update(np.array(pos))
        self.history.append(self.kf.x[:2])
        self.age = 0

class Tracker:
    def __init__(self, max_age=30, dist_threshold=50):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.dist_threshold = dist_threshold

    def update(self, detections):
        dets = detections  # ahora detections ya es una lista de (x, y) tuplas BEV
        if not self.tracks:
            for pos in dets:
                self.tracks.append(Track(self.next_id, pos, self.max_age))
                self.next_id += 1
            return self.tracks

        preds = np.array([trk.predict() for trk in self.tracks])
        if len(dets) == 0:
            return self.tracks

        cost = np.linalg.norm(preds[:, None, :] - np.array(dets)[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.dist_threshold:
                self.tracks[r].update(dets[c])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        for i, pos in enumerate(dets):
            if i not in assigned_dets:
                self.tracks.append(Track(self.next_id, pos, self.max_age))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return self.tracks
