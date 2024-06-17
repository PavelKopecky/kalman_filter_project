import itertools
import random

import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.image as image
from scipy.spatial import KDTree
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import BallTree


def add_radar_image(ax, image_path, zoom=1):
    img = image.imread(image_path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
    ax.add_artist(ab)


def add_plane_image(ax, image_path, x, y, angle, zoom=1):
    img = image.imread(image_path)

    angle_d = np.degrees(angle) - 90
    img = ndimage.rotate(img, angle_d, reshape=True)

    imagebox = OffsetImage(img.clip(0, 1), zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)


def draw_radar_background(ax, limits):
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    # Draw concentric circles for the radar grid
    grid_radii = np.linspace(0, max(limits), 5)
    for radius in grid_radii:
        circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle=':')
        ax.add_artist(circle)

    # Set limits and grid style
    ax.set_xlim([-max(limits), max(limits)])
    ax.set_ylim([-max(limits), max(limits)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)  # Turn off the default grid


def draw_confidence_ellipse(ax, mean, cov, n_std=3.0, facecolor='none', **kwargs):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)
    return ax.add_patch(ellipse)


def draw_track_path(track):
    global track_legend_shown, ax

    draw_confidence_ellipse(ax, mean=track.filter.x_upd[-1][:2], cov=track.filter.P_upd[-1][:2, :2],
                            n_std=7.0, edgecolor='red',
                            label='Confidence ellipse derived from P_{k|k}' if not track_legend_shown else None,
                            alpha=track.rk)
    plt.plot(np.array(track.filter.x_upd)[-30:, 0], np.array(track.filter.x_upd)[-30:, 1], '-',
             color='red',
             label="Position estimate" if not track_legend_shown else None, alpha=track.rk)
    plt.plot(np.array(track.filter.x_upd)[-1, 0], np.array(track.filter.x_upd)[-1, 1], '+',
             markersize=12, color='red', alpha=track.rk)

    last_x, last_y = np.array(track.filter.x_upd)[-1, 0], np.array(track.filter.x_upd)[-1, 1]
    plt.text(last_x, last_y - 180, f'rk: {track.rk:.1f}', color='red', ha='center')


def draw_radar_w_measurements():
    add_radar_image(ax, "radar.png", zoom=0.4)
    draw_radar_background(ax, trj.y1_lims)
    plt.plot(Y[0], Y[1], '.', color='grey', markersize=2, label=f"Clutter ({Y[0].size - 2} pts)")

    for i_ in range(NO_OF_PLANES):
        # plt.plot(trj.X[0, i_-30:NO_OF_PLANES * t + i_:NO_OF_PLANES], trj.X[1, i_-30:NO_OF_PLANES * t + i_:NO_OF_PLANES],
        #          '-',
        #          color='blue', label="True position" if i_ == 0 else None)
        add_plane_image(ax, "plane.png", *trj.X[:3, NO_OF_PLANES * t + i_], zoom=.4)


def get_mahalanobis(x_pred, data, S):
    near = []

    min_dist = np.inf
    nearest = None

    Si = np.linalg.inv(S)

    for v in data:
        z = x_pred[0:2] - transition(v)
        dist = np.sqrt(z.T @ Si @ z)
        if dist < 3:
            near.append(transition(v))
        if dist < min_dist:
            nearest = transition(v)
            min_dist = dist

    return near, nearest


def mah(x, z, S):
    return np.sqrt((x - z).T @ np.linalg.inv(S) @ (x - z))


def to_coordinates(distance, angle):
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    return [x, y]


def f(dt, old):
    # state transition function of the CTRV model
    # https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/motion-model-design.html

    x, y, theta, v, omega = old[0], old[1], old[2], old[3], old[4]
    n = [0, 0, dt * omega + theta, v, omega]
    if np.abs(omega) < 1e10:
        n[0] = x + dt * v * np.cos(theta)
        n[1] = y + dt * v * np.sin(theta)
    else:
        n[0] = x - v * (np.sin(theta) / omega) + v * (np.sin(dt * omega + theta) / omega)
        n[1] = y + v * (np.cos(theta) / omega) - v * (np.cos(dt * omega + theta) / omega)
    return n


def get_sigma_points(x, P):
    # as per https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    L = len(x)

    alpha = 1e-3
    kappa = 0
    beta = 2
    lambda_ = (alpha ** 2) * (L + kappa) - L

    try:
        A = np.linalg.cholesky(P)
    except:
        A = (np.linalg.norm(P) / 2) * P

    sigma_points = [x]
    first_W = [lambda_ / (L + lambda_)]
    second_W = [lambda_ / (L + lambda_) + (1 - alpha ** 2 + beta)]

    first_W += [1 / (2 * (L + lambda_)) for _ in range(2 * L)]
    second_W += [1 / (2 * (L + lambda_)) for _ in range(2 * L)]

    for j in range(L):
        s = x + np.sqrt(L + lambda_) * A[:, j]
        sigma_points.append(s)

    for j in range(L):
        s = x - np.sqrt(L + lambda_) * A[:, j]
        sigma_points.append(s)

    return sigma_points, first_W, second_W


def transition(x):
    range_measure = np.sqrt(x[0] ** 2 + x[1] ** 2)
    bearing_measure = np.arctan2(x[1], x[0])
    return np.array([range_measure, bearing_measure])


class UnscentedKalmanFilter:

    def __init__(self, Q, R, H, x0, P0, dt, dim=5):
        self.Q = Q
        self.R = R
        self.H = H
        self.x_pred = []
        self.P_pred = []
        self.x_upd = [x0]
        self.P_upd = [P0]
        self.S = []
        self.W = []
        self.dt = dt
        self.dim = dim
        self.z = []

    def predict(self):
        sp, fo, so = get_sigma_points(self.x_upd[-1], self.P_upd[-1])
        x = np.zeros(self.dim)
        P = np.zeros((self.dim, self.dim))

        for i, point in enumerate(sp):
            sp[i] = np.array(f(self.dt, point))
            x += fo[i] * sp[i]

        for i, point in enumerate(sp):
            P += so[i] * np.outer(sp[i] - x, sp[i] - x)
        P += self.Q

        self.x_pred.append(x)
        self.P_pred.append(P)

        sp, fo, so = get_sigma_points(self.x_pred[-1], self.P_pred[-1])
        zj = [transition(point[:2]) for point in sp]
        z = np.zeros(2)

        for i, zk in enumerate(zj):
            z += fo[i] * zk

        Sk = self.R.copy()
        Cxz = np.zeros((self.dim, 2))
        for i, (sigma_p, zk) in enumerate(zip(sp, zj)):
            Sk += so[i] * np.outer(zk - z, zk - z)
            Cxz += so[i] * np.outer(sigma_p - self.x_pred[-1], zk - z)

        Kk = Cxz @ np.linalg.inv(Sk)

        self.S.append(Sk)
        self.W.append(Kk)
        self.z.append(z)

        return x, P

    def update(self, Z, jpda_weights=None, jpda_beta0=None):

        if not jpda_weights:

            beta_i = []

            weights = [(mvn.pdf(zi, self.z[-1], self.S[-1], allow_singular=True) * PD) / lambd for zi in Z]
            total_w = sum(weights)
            beta_0 = (1 - PD * PG) / (1 - PD * PG + total_w)

            for z in Z:
                weight = (mvn.pdf(z, self.z[-1], self.S[-1], allow_singular=True) * PD) / lambd
                beta = weight / (1 - PD * PG + total_w)
                beta_i.append(beta)

        else:
            beta_i = jpda_weights
            beta_0 = jpda_beta0

        vk = np.zeros(2)
        for i, z in enumerate(Z):
            vk += beta_i[i] * (z - self.z[-1])

        if np.linalg.norm(vk) > 120:
            return False

        P_tilde_no_gain = np.zeros((2, 2)) - np.outer(vk, vk)

        for i, z in enumerate(Z):
            vi = (z - self.z[-1])
            P_tilde_no_gain += beta_i[i] * np.outer(vi, vi)

        Pc = self.P_pred[-1] - self.W[-1] @ self.S[-1] @ self.W[-1].T
        P_tilde = self.W[-1] @ P_tilde_no_gain @ self.W[-1].T

        Pk = beta_0 * self.P_pred[-1] + (1 - beta_0) * Pc + P_tilde
        xk = self.x_pred[-1] + self.W[-1] @ vk

        self.x_upd.append(xk)
        self.P_upd.append(Pk)

        if np.linalg.norm(Pk) >= 2000:
            return False

        return True


class Trajectory:

    def __init__(self, seed=123, ndat=200):
        self.Y_with_clutter = None
        self.ndat = ndat
        self.seed = np.random.randint(0, 1000)
        # self.seed = seed
        self.dt = 1.

        self.Q = np.diag([10, 10, np.pi / 400, .001, np.pi / 50000])

        self.H = np.array([[1., 0, 0, 0, 0],
                           [0., 1, 0, 0, 0]])

        self.R = np.diag([50, np.pi / 8060])

        self.X = np.zeros(shape=(5, NO_OF_PLANES * self.ndat))

        self.Y = [np.zeros(shape=(2, self.ndat)) for _ in range(NO_OF_PLANES)]

        self.y1_lims = np.array([-2000., 2000.])
        self.y2_lims = np.array([-2000., 2000.])
        self.area_volume = (self.y1_lims[1] - self.y1_lims[0]) * (self.y2_lims[1] - self.y2_lims[0])
        self._simulate()

    def _simulate(self):
        np.random.seed(self.seed)

        for i_ in range(NO_OF_PLANES):

            if i_ == 0:
                x_ = np.array([500, 300, -np.pi, 30, 0])
            elif i_ == 1:
                x_ = np.array([-500, 300, 0, 30, 0])
            else:
                x_ = np.array([random.randint(-1500, 1500), random.randint(-1500, 1500),
                               5 * (random.random() - 0.5), random.randint(25, 35), 0])

            for t_ in range(self.ndat):
                x_ = f(self.dt, x_) + mvn.rvs(cov=self.Q)
                y_ = transition(x_[:2]) + mvn.rvs(cov=self.R)
                self.X[:, NO_OF_PLANES * t_ + i_] = x_
                self.Y[i_][:, t_] = y_

    def addClutter(self, lambd=2.):

        self.Y_with_clutter = []

        for t_ in range(self.ndat):
            clutter_count = poisson.rvs(lambd * self.area_volume)
            Yt = np.zeros((2, clutter_count + NO_OF_PLANES))
            Yt[0, NO_OF_PLANES:] = np.random.uniform(low=self.y1_lims[0], high=self.y1_lims[1], size=clutter_count)
            Yt[1, NO_OF_PLANES:] = np.random.uniform(low=self.y2_lims[0], high=self.y2_lims[1], size=clutter_count)

            for i_ in range(NO_OF_PLANES):
                Yt[:, i_] = to_coordinates(*self.Y[i_][:, t_])

            self.Y_with_clutter.append(Yt)


class Track:
    def __init__(self, position, traj):
        self.filter = None
        self.traj = traj
        self.position = position  # dist, angle
        self.covariance = np.diag([200, np.pi / 2])
        self.score = 1
        self.time = 1
        self.low_r_time = 0
        self.PS = .5
        self.rk = .3
        self.PD = .9
        self.rk_apriori = 0

    def init_filter(self, pos):

        coords = to_coordinates(self.position[0], self.position[1])
        diff = pos - coords
        diff_t = transition(diff)

        self.filter = UnscentedKalmanFilter(2 * self.traj.Q, 2 * self.traj.R, self.traj.H,
                                            [(pos[0] + coords[0]) // 2, (pos[1] + coords[1]) // 2, diff_t[1],
                                             min(40, diff_t[0]), 0],
                                            2 * self.traj.Q, 1, 5)
        self.predict()

    def predict(self):
        if self.filter:
            self.filter.predict()
            self.position = transition(self.filter.x_pred[-1][:2])
            self.covariance = self.filter.S[-1]
        self.rk_apriori = self.rk * self.PS

    def update(self, measurements):
        self.time += 1
        nearest_points = []
        indexes_ = []

        if len(measurements):
            measurements_t = [transition(x) for x in measurements]

            mahalanobis_metric = DistanceMetric.get_metric('mahalanobis', V=self.covariance)
            tree = BallTree(measurements_t, metric=mahalanobis_metric)

            dist_, idx_ = tree.query(self.position.reshape(1, -1), k=1)

            first_index = idx_[0][0]
            indexes_ = [first_index]

            dist_ = dist_[0][0]
            tree_index = 2

            while dist_ < 9:
                nearest_points.append(measurements_t[idx_[0][0]])
                if tree_index > len(measurements_t):
                    break
                dist_, idx_ = tree.query(self.position.reshape(1, -1), k=tree_index)
                tree_index += 1
                dist_ = dist_[0][0]
                indexes_.append(idx_[0][0])

        if self.time <= 3 and not nearest_points:
            return False, -1

        likelihood = (1 - self.PD) + (self.PD / lambd) * sum(
            [mvn.pdf(zk, self.position[:2], self.covariance, allow_singular=True) for zk in measurements_t])
        self.rk = (likelihood * self.rk_apriori) / (1 - self.rk_apriori + likelihood * self.rk_apriori)

        if self.rk < 0.15:
            self.low_r_time += 1

            if self.low_r_time > 3:
                return False, None
        else:
            self.low_r_time = 0

        if self.time == 3 and not self.filter:
            self.init_filter(measurements[first_index])

        if self.filter:
            if not self.filter.update(nearest_points):
                return False, None

            else:
                return True, indexes_
        else:
            return True, indexes_

    def update_jpda(self, target_measurements_, target_weights_, misdetect_proba_):
        self.time += 1
        likelihood = (1 - self.PD) + (self.PD / lambd) * sum(
            [mvn.pdf(zk, self.position[:2], self.covariance, allow_singular=True) for zk in target_measurements_])
        self.rk = (likelihood * self.rk_apriori) / (1 - self.rk_apriori + likelihood * self.rk_apriori)

        if self.rk < 0.15:
            self.low_r_time += 1

            if self.low_r_time > 3:
                return False
        else:
            self.low_r_time = 0

        if not self.filter.update(target_measurements_, target_weights_, misdetect_proba_):
            return False

        return True


def get_cost_matrix(measurements_, tracks_):
    matrix_c = np.zeros((len(tracks_), len(measurements_) + len(tracks_)))

    for i_, track_t in enumerate(tracks_):
        for j_, measurement in enumerate(measurements_):
            likelihood = track_t.PD * mvn.pdf(transition(measurement), track_t.position, track_t.covariance,
                                              allow_singular=True)
            matrix_c[i_, j_] = likelihood

        for j_ in range(len(tracks_)):
            if j_ == i_:
                mis_det = lambd * (1 - track_t.rk_apriori)
                matrix_c[i_, j_ + len(measurements_)] = mis_det
            else:
                matrix_c[i_, j_ + len(measurements_)] = 0

    return matrix_c


def generate_hypothesis_matrices(num_tracks, num_measurements):
    choices_per_track = num_measurements + 1

    all_assignments = list(itertools.product(range(choices_per_track), repeat=num_tracks))

    valid_assignments = [assignment for assignment in all_assignments if
                         len(set(assignment) - {num_measurements}) == len(
                             [x for x in assignment if x < num_measurements])]

    # create hypothesis matrices
    hypothesis_matrices = []
    for assignment in valid_assignments:
        matrix = np.zeros((num_tracks, num_tracks + num_measurements))
        for i, assign in enumerate(assignment):
            matrix[i, assign] = 1 if assign < num_measurements else 0
            if assign == num_measurements:
                matrix[i, num_measurements + i] = 1  # misdetection
        hypothesis_matrices.append(matrix)

    return hypothesis_matrices


def generate_valid_hypotheses(matrix_c):
    num_tracks, total_columns = matrix_c.shape
    num_measurements = total_columns - num_tracks
    valid_hypotheses = []

    # generate combinations of hypothesis
    for assignment in np.ndindex(*(num_measurements + 1,) * num_tracks):
        # create the hypothesis matrix for this assignment
        matrix = np.zeros_like(matrix_c)
        valid = True

        for i, assign in enumerate(assignment):
            if assign < num_measurements:
                # check if we are allowed to assign this measurement
                if matrix_c[i, assign] == 0:
                    valid = False
                    break
                matrix[i, assign] = 1
            else:
                # handle misdetection case
                misdet_index = num_measurements + i
                if matrix_c[i, misdet_index] == 0:
                    valid = False
                    break
                matrix[i, misdet_index] = 1

        # ensure there is at most one 1 in each column
        if valid and np.all(matrix.sum(axis=0) <= 1):
            valid_hypotheses.append(matrix)

    return valid_hypotheses


def get_target_associations(targets__, Y_transformed_):
    target_associations_ = []
    for target_ in targets__:
        mahalanobis_metric = DistanceMetric.get_metric('mahalanobis', V=target_.covariance)
        tree = BallTree(Y_transformed_, metric=mahalanobis_metric)
        indices_within_gate = tree.query_radius(target_.position.reshape(1, -1), r=8)

        associated_measurements = indices_within_gate[0].tolist()
        target_associations_.append(associated_measurements)

    return target_associations_


def get_overlapping_groups(target_associations__):
    overlapping_groups_ = []

    for i, target_i_associations in enumerate(target_associations__):
        added = False
        for group in overlapping_groups_:
            if any(measurement in target_i_associations for measurement in group['measurements']):
                group['targets'].append(i)
                group['measurements'].update(target_i_associations)
                added = True
                break

        if not added:
            overlapping_groups_.append({'targets': [i], 'measurements': set(target_i_associations)})

    return overlapping_groups_


def run_jpda():
    global Y_t, printed, track_legend_shown

    # For each target find possible associated measurements:
    target_associations = get_target_associations(jpda_targets, Y_transformed)

    # get overlapping groups
    overlapping_groups = get_overlapping_groups(target_associations)

    # construct a cost matrix and hypothesis matrices for each group
    for group in overlapping_groups:
        if len(group['targets']) > 1:
            print("Targets in the same group:", group['targets'])
            printed = True
        group['cost_matrix'] = get_cost_matrix([Y_t[k] for k in group['measurements']],
                                               [jpda_targets[k] for k in group['targets']])
        group['hypothesis'] = generate_valid_hypotheses(group['cost_matrix'])

    # get normalized association probabilities
    for group in overlapping_groups:
        association_probs = []
        for association in group['hypothesis']:
            association_probs.append(np.trace(np.matmul(association.T, group['cost_matrix'])))
        association_probs /= sum(association_probs)
        group['association_probs'] = association_probs

    # for each target in each group for each of its associations, get its weight and execute the update step:
    for group in overlapping_groups:
        for tar_index, target in enumerate(group['targets']):
            target_measurements = []
            target_weights = []
            for meas_index, measurement in enumerate(group['measurements']):
                total_weight = 0
                for assoc_index, association in enumerate(group['hypothesis']):
                    if association[tar_index][meas_index] == 1:
                        total_weight += group['association_probs'][assoc_index]
                target_measurements.append(Y_transformed[measurement])
                target_weights.append(total_weight)

            misdetect_proba = 0
            for assoc_index, association in enumerate(group['hypothesis']):
                if association[tar_index][len(group['measurements']) + tar_index] == 1:
                    misdetect_proba += group['association_probs'][assoc_index]

            track = jpda_targets[target]

            if not track.update_jpda(target_measurements, target_weights, misdetect_proba):
                tracks.remove(track)
                continue

            if track.rk > 0.7:
                draw_track_path(track)
                track_legend_shown = True

    # remove used indexes
    removed_indexes = set()
    for group in overlapping_groups:
        for measurement in group['measurements']:
            removed_indexes.add(measurement)

    Y_t = np.delete(Y_t, list(removed_indexes), 0)


def run_ipda():
    global Y_t, track_legend_shown

    current_tree = None

    for i, track in enumerate(non_jpda_targets):

        if not len(Y_t):
            tracks.remove(track)
            continue

        res, removed_indexes = track.update(Y_t)

        if not res:
            tracks.remove(track)
            continue

        if len(removed_indexes):
            Y_t = np.delete(Y_t, removed_indexes, 0)

        if track.filter:
            if (track.time > 15 and track.rk > 0.5) or (track.rk > 0.9 and track.time > 8):

                if len(displayed_targets):
                    current_tree = KDTree(displayed_targets)
                    dist, idx = current_tree.query(track.filter.x_upd[-1][:2].reshape(1, -1), k=1)
                    if dist < 300:
                        continue

                displayed_targets.append(track.filter.x_upd[-1][:2])
                draw_track_path(track)
                track_legend_shown = True

            elif track.time > 6:

                if len(displayed_targets):
                    if current_tree is None:
                        current_tree = KDTree(displayed_targets)
                    dist, idx = current_tree.query(track.filter.x_upd[-1][:2].reshape(1, -1), k=1)
                    if dist < 300:
                        continue

                plt.plot(np.array(track.filter.x_upd)[-30:, 0], np.array(track.filter.x_upd)[-30:, 1], '-',
                         color='black', alpha=0.3)

    for x in Y_t:
        tracks.append(Track(transition(x), trj))


if __name__ == "__main__":
    # config
    ndat = 250
    lambd = 0.000002
    pause = 0.4
    NO_OF_PLANES = 4
    SHOW_LEGEND = False
    PD = 0.9
    PG = 0.9993

    # init trajectory
    trj = Trajectory(111, ndat=ndat)
    trj.addClutter(lambd)
    tracks = []

    # init matplotlib
    plt.ion()
    plt.figure(figsize=(12, 7.5))
    ax = plt.gca()
    plt.cla()

    cost_matrix = None

    # run simulation
    for t in range(ndat):
        plt.cla()

        track_legend_shown = False
        printed = False
        displayed_targets = []

        Y = trj.Y_with_clutter[t]
        Y_t = Y.T
        Y_transformed = [transition(t__) for t__ in Y_t]

        draw_radar_w_measurements()

        tracks = sorted(tracks, key=lambda x: x.time, reverse=True)
        tracks = sorted(tracks, key=lambda x: x.rk, reverse=True)

        for track_ in tracks:
            track_.predict()

        # Get all tracks with filters and high rk:
        jpda_targets = [track for track in tracks if track.filter and track.rk > 0.8]
        non_jpda_targets = [track for track in tracks if track not in jpda_targets]

        run_jpda()
        run_ipda()

        if printed:
            print("-----")

        SHOW_LEGEND and plt.legend(loc='lower left')

        plt.pause(pause)
