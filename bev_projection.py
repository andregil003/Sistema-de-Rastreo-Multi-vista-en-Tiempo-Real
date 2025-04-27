
import numpy as np
import json


class BEVProjector:

    
    def __init__(self, config_path):
        cfg = json.load(open(config_path))
        self.cameras = {
            c['id']: np.array(c['homography'], dtype=np.float32)
            for c in cfg['cameras']
        }

    def image_to_bev(self, cam_id, point):
        """Proyecta punto (u,v) de la imagen a coordenadas BEV (x,y)"""
        H = self.cameras[cam_id]
        uv1 = np.array([point[0], point[1], 1.0])
        xyw = H.dot(uv1)
        return (xyw[0]/xyw[2], xyw[1]/xyw[2])

    def bev_to_image(self, cam_id, bev_point):
        """Proyecta punto BEV (x,y) de vuelta a coordenadas de imagen"""
        H = self.cameras[cam_id]
        H_inv = np.linalg.inv(H)
        xy1 = np.array([bev_point[0], bev_point[1], 1.0])
        uvw = H_inv.dot(xy1)
        return (int(uvw[0]/uvw[2]), int(uvw[1]/uvw[2]))

    
