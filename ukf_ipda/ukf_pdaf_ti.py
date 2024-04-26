import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
import matplotlib.lines as mlines
import matplotlib.image as image
from scipy.spatial import KDTree
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import BallTree
import sys


def exit_program(event):
    sys.exit()


def draw_confidence_ellipse(ax, mean, cov, n_std=3.0, facecolor='none', **kwargs):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # eigenvalues of the covariance matrix are equal to the variance of the data's coordinates
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)

    return ax.add_patch(ellipse)


def to_coordinates(distance, angle):
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    return [x, y]


def f(dt, old):
    # state transition function of the CTRV model
    # https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/motion-model-design.html

    x, y, theta, v, omega = old[0], old[1], old[2], old[3], old[4]
    n = [0, 0, dt * omega + theta, v, omega]
    if omega == 0:
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

    A = np.linalg.cholesky(P)

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

    def update(self, Z):

        beta_i = []

        weights = [(mvn.pdf(zi, self.z[-1], self.S[-1], allow_singular=True) * PD) / lambd for zi in Z]
        total_w = sum(weights)
        beta_0 = (1 - PD * PG) / (1 - PD * PG + total_w)

        for z in Z:
            weight = (mvn.pdf(z, self.z[-1], self.S[-1], allow_singular=True) * PD) / lambd
            beta = weight / (1 - PD * PG + total_w)
            beta_i.append(beta)

        vk = np.zeros(2)
        for i, z in enumerate(Z):
            vk += beta_i[i] * (z - self.z[-1])

        # print("vk norm: ", np.linalg.norm(vk))

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

        #print("Pk norm: ", np.linalg.norm(Pk))

        if np.linalg.norm(Pk) >= 2000:
            return False

        return True


class Trajectory:

    def __init__(self, seed=123, ndat=200):
        self.ndat = ndat
        self.seed = np.random.randint(0, 1000)
        # self.seed = seed
        self.dt = 1.

        self.Q = np.diag([10, 10, np.pi / 400, .001, np.pi / 50000])

        self.H = np.array([[1., 0, 0, 0, 0],
                           [0., 1, 0, 0, 0]])

        self.R = np.diag([300, np.pi / 6060])
        if start_center:
            self.m0 = np.array([0, 0, 0, 30, 0])
        else:
            self.m0 = np.array([1800, 1000, -0.8 * np.pi, 30, 0])

        self.X = np.zeros(shape=(5, 2 * self.ndat))
        self.Y = np.zeros(shape=(2, self.ndat))
        self.Y2 = np.zeros(shape=(2, self.ndat))
        self.y1_lims = np.array([-2000., 2000.])
        self.y2_lims = np.array([-2000., 2000.])
        self.area_volume = (self.y1_lims[1] - self.y1_lims[0]) * (self.y2_lims[1] - self.y2_lims[0])
        self._simulate()

    def _simulate(self):
        np.random.seed(self.seed)

        x = self.m0
        x2 = np.array([1800, 1000, -0.8 * np.pi, 30, 0])
        for t in range(self.ndat):
            x = f(self.dt, x) + mvn.rvs(cov=self.Q)
            x2 = f(self.dt, x2) + mvn.rvs(cov=self.Q)
            y = transition(x[:2]) + mvn.rvs(cov=self.R)
            y2 = transition(x2[:2]) + mvn.rvs(cov=self.R)
            self.X[:, 2*t] = x
            self.X[:, 2 * t + 1] = x2
            self.Y[:, t] = y
            self.Y2[:, t] = y2


    def addClutter(self, lambd=2.):
        self.Y_with_clutter = []

        for t in range(self.ndat):
            clutter_count = poisson.rvs(lambd * self.area_volume)
            Yt = np.zeros((2, clutter_count + 2))
            Yt[:, 0] = to_coordinates(*self.Y[:, t])
            Yt[:, 1] = to_coordinates(*self.Y2[:, t])
            # Yt[0, 1:] = np.random.uniform(low=self.y1_lims[0], high=self.y1_lims[1], size=clutter_count)
            # Yt[1, 1:] = np.random.uniform(low=self.y2_lims[0], high=self.y2_lims[1], size=clutter_count)
            Yt[0, 2:] = np.random.uniform(low=0, high=self.y1_lims[1], size=clutter_count)
            Yt[1, 2:] = np.random.uniform(low=0, high=2 * np.pi, size=clutter_count)

            for i in range(2, clutter_count + 2):
                transformed = to_coordinates(*Yt[:, i])
                Yt[0, i] = transformed[0]
                Yt[1, i] = transformed[1]

            self.Y_with_clutter.append(Yt)


class Track:
    def __init__(self, position, traj):
        self.state = "INIT"
        self.last_detected = [1, 0, 0]
        self.filter = None
        self.traj = traj
        self.position = position  # dist, angle
        self.covariance = np.diag([250, np.pi / 6])
        self.score = 1
        self.time = 1

        self.PS = 0.95
        self.rk = 1

    def init_filter(self, pos):

        coords = to_coordinates(self.position[0], self.position[1])
        diff = pos - coords
        diff_t = transition(diff)
        self.filter = UnscentedKalmanFilter(2 * self.traj.Q, self.traj.R, self.traj.H,
                                            [pos[0], pos[1]] + [0, diff_t[0], 0],
                                            20 * self.traj.Q, 1, 5)
        self.predict()

    def predict(self):
        if self.filter:
            self.filter.predict()
            self.position = transition(self.filter.x_pred[-1][:2])
            self.covariance = self.filter.S[-1]

    def update(self, measurements):
        self.predict()
        self.time += 1

        measurements_t = [transition(x) for x in measurements]

        mahalanobis_metric = DistanceMetric.get_metric('mahalanobis', V=self.covariance)
        tree = BallTree(measurements_t, metric=mahalanobis_metric)
        dist, idx = tree.query(self.position.reshape(1, -1), k=1)

        dist = dist[0][0]

        if dist < 5:

            i = 2
            nearest_points = []
            while dist < 5 and i < 5:
                nearest_points.append(measurements_t[idx[0][0]])
                if i > len(measurements_t):
                    break
                dist, idx = tree.query(self.position.reshape(1, -1), k=i)
                i += 1
                dist = dist[0][0]

            if dist < 2:
                self.score *= 1.15
            elif dist < 3:
                self.score *= 1.1
            else:
                self.score *= 1.05

            self.last_detected = [1] + self.last_detected[:2]

            if self.state == "INIT":
                self.state = "TENTATIVE"
                self.init_filter(measurements[idx[0][0]])
                self.last_detected = [0, 0, 0]
                self.time = 0
            elif self.state == "TENTATIVE" and sum(self.last_detected) >= 2:
                self.state = "CONFIRMED"

            if not self.filter.update(nearest_points):
                return False, 0
            # print("UPDATED: dist: ", dist, " prev state:", self.state, " last det:", self.last_detected)
            return True, idx[0][0]

        else:

            self.last_detected = [0] + self.last_detected[:2]

            if dist > 10:
                self.score /= 1.5
            elif dist > 7:
                self.score /= 1.3
            else:
                self.score /= 1.1

            if self.state == "INIT":
                return False, 0

            elif self.state == "TENTATIVE" and sum(self.last_detected) <= 1 and self.time > 3:
                return False, 0

            elif self.state == "CONFIRMED" and sum(self.last_detected) <= 1:
                self.state = "TENTATIVE"

            if not self.filter.update([]):
                return False, 0
            # print("UPDATED: dist: ", dist, " prev state:", self.state, " last det:", self.last_detected)
            return True, -1


def add_radar_image(ax, image_path, zoom=1):
    img = image.imread(image_path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
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


if __name__ == "__main__":
    ndat = 250
    lambd = 0.000001
    pause = .05
    start_center = True
    trj = Trajectory(111, ndat=ndat)
    trj.addClutter(lambd)
    PD = 1
    PG = 0.9993
    errs = []

    tracks = []

    plt.ion()
    plt.figure(figsize=(12, 7.5))
    ax = plt.gca()
    plt.cla()

    for t in range(ndat):
        plt.cla()

        track_legend_shown = False

        add_radar_image(ax, "radar.png", zoom=0.06)
        draw_radar_background(ax, trj.y1_lims)

        Y = trj.Y_with_clutter[t]
        Y_t = Y.T

        tracks = sorted(tracks, key=lambda x: x.score, reverse=True)

        for i, track in enumerate(tracks):

            if not len(Y_t):
                tracks.remove(track)
                continue

            res, index = track.update(Y_t)

            if not res:
                tracks.remove(track)
                continue

            if index != -1 and i > 0:
                Y_t = np.delete(Y_t, index, 0)

            if track.filter:
                if track.state == "CONFIRMED":
                    draw_confidence_ellipse(ax, mean=track.filter.x_upd[-1][:2], cov=track.filter.P_upd[-1][:2, :2],
                                            n_std=5.0, edgecolor='red',
                                            label='Confidence ellipse derived from P_{k|k}' if not track_legend_shown else None)
                    plt.plot(np.array(track.filter.x_upd)[-30:, 0], np.array(track.filter.x_upd)[-30:, 1], '-',
                             color='red',
                             label="Position estimate" if not track_legend_shown else None)
                    plt.plot(np.array(track.filter.x_upd)[-1, 0], np.array(track.filter.x_upd)[-1, 1], '+',
                             markersize=12, color='red')

                    track_legend_shown = True
                elif track.score >= 2.5:
                    draw_confidence_ellipse(ax, mean=track.filter.x_upd[-1][:2], cov=track.filter.P_upd[-1][:2, :2],
                                            n_std=5.0, edgecolor='black', alpha=.9)
                    plt.plot(np.array(track.filter.x_upd)[-1, 0], np.array(track.filter.x_upd)[-1, 1], '+',
                             markersize=12, color='gray', alpha=.9)

        for x in Y_t:
            tracks.append(Track(transition(x), trj))

        plt.plot(Y[0], Y[1], '.', color='grey', markersize=2, label=f"Clutter ({Y[0].size} pts)")
        plt.plot(Y[0, 0], Y[1, 0], '.', color='red', markersize=10)
        plt.plot(trj.X[0, 2 * t], trj.X[1, 2 * t], 'x', markersize=15, color='blue', label="True position")
        plt.plot(trj.X[0, 2 * t + 1], trj.X[1, 2 * t + 1], 'x', markersize=15, color='blue')

        plt.legend(loc='lower left')

        if np.abs(trj.X[0, t]) > 2000 or np.abs(trj.X[1, t]) > 2000:
            break

        plt.pause(pause)
