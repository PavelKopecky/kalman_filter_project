import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
import matplotlib.lines as mlines
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)


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
        n[0] = x - v * (np.sin(theta) / omega) + v * (np.sin(dt*omega + theta) / omega)
        n[1] = y + v * (np.cos(theta) / omega) - v * (np.cos(dt*omega + theta) / omega)
    return n


def get_sigma_points(x, P):
    # as per https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    L = len(x)

    alpha = 1e-3
    kappa = 0
    beta = 2
    lambda_ = (alpha**2)*(L + kappa) - L

    A = np.linalg.cholesky(P)

    sigma_points = [x]
    first_W = [lambda_ / (L + lambda_)]
    second_W = [lambda_ / (L + lambda_) + (1 - alpha**2 + beta)]

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
        self.score = 1
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
        weights = [(mvn.pdf(zi, self.z[-1], self.S[-1]) * PD) / lambd for zi in Z]
        total_w = sum(weights)
        beta_0 = (1 - PD*PG) / (1 - PD*PG + total_w)

        for z in Z:
            weight = (mvn.pdf(z, self.z[-1], self.S[-1]) * PD) / lambd
            beta = weight / (1 - PD * PG + total_w)
            beta_i.append(beta)

        vk = np.zeros(2)
        for i, z in enumerate(Z):
            vk += beta_i[i] * (z - self.z[-1])

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

        return xk, Pk


class Trajectory:

    def __init__(self, seed=123, ndat=200):
        self.ndat = ndat
        self.seed = np.random.randint(0, 1000)
        #self.seed = seed
        self.dt = 1.

        self.Q = np.diag([20, 20, np.pi/100, 10, np.pi/10000])

        self.H = np.array([[1., 0, 0, 0, 0],
                           [0., 1, 0, 0, 0]])

        self.R = np.diag([600, np.pi/6060])
        self.m0 = np.array([0, 0, 0, 100, 0])

        self.X = np.zeros(shape=(5, self.ndat))
        self.Y = np.zeros(shape=(2, self.ndat))
        self.y1_lims = np.array([-2000., 2000.])
        self.y2_lims = np.array([-2000., 2000.])
        self.area_volume = (self.y1_lims[1] - self.y1_lims[0]) * (self.y2_lims[1] - self.y2_lims[0])
        self._simulate()

    def _simulate(self):
        np.random.seed(self.seed)

        x = self.m0
        for t in range(self.ndat):
            x = f(self.dt, x) + mvn.rvs(cov=self.Q)
            y = transition(x[:2]) + mvn.rvs(cov=self.R)
            self.X[:, t] = x
            self.Y[:, t] = y

    def addClutter(self, lambd=2.):
        self.Y_with_clutter = []

        for t in range(self.ndat):
            clutter_count = poisson.rvs(lambd * self.area_volume)
            Yt = np.zeros((2, clutter_count + 1))
            Yt[:, 0] = to_coordinates(*self.Y[:, t])
            Yt[0, 1:] = np.random.uniform(low=self.y1_lims[0], high=self.y1_lims[1], size=clutter_count)
            Yt[1, 1:] = np.random.uniform(low=self.y2_lims[0], high=self.y2_lims[1], size=clutter_count)
            self.Y_with_clutter.append(Yt)


def add_radar_image(ax, image_path, zoom=1):

    img = image.imread(image_path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
    ax.add_artist(ab)


# Function to draw radar-like circular gridlines
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
    lambd = 0.00002
    pause = .00005
    trj = Trajectory(111, ndat=ndat)
    trj.addClutter(lambd)
    PD = 1
    PG = 0.9973

    ukf = UnscentedKalmanFilter(
        trj.Q,
        trj.R,
        trj.H,
        trj.m0,
        .0001 * np.eye(5),
        1,
        5
    )

    plt.ion()
    plt.figure(figsize=(12, 7.5))
    ax = plt.gca()
    plt.cla()

    for t in range(ndat):
        plt.cla()

        add_radar_image(ax, "radar.png", zoom=0.06)
        draw_radar_background(ax, trj.y1_lims)

        Y = trj.Y_with_clutter[t]

        pred_x, P_pred = ukf.predict()

        close, _ = get_mahalanobis(transition(pred_x[:2]), Y.T, ukf.S[-1])
        #print([x for x in close])
        close_n = np.array([to_coordinates(*x) for x in close]).T

        upd_x, P_upd = ukf.update(close)

        draw_confidence_ellipse(ax, mean=upd_x[:2], cov=P_upd[:2, :2], n_std=3.0, edgecolor='red',
                                label='Confidence ellipse derived from P_{k|k}')

        plt.plot(Y[0], Y[1], '.', color='grey', markersize=2, label=f"Clutter ({Y[0].size} pts)")[0]
        if len(close):
            plt.plot(close_n[0], close_n[1], '.', color='lime', alpha=0.8, markersize=3)[0]

        plt.plot(Y[0, 0], Y[1, 0], '.', color='red', markersize=10)
        plt.plot(np.array(ukf.x_upd)[:, 0], np.array(ukf.x_upd)[:, 1], '-', color='red',
                 label="Position estimate")
        plt.plot(trj.X[0, :t + 1], trj.X[1, :t + 1], '-', color='blue', label="True position")

        plt.legend(loc='lower left')

        if np.abs(trj.X[0, t + 1]) > 2000 or np.abs(trj.X[1, t + 1]) > 2000:
            break

        plt.pause(0.5)

    plt.waitforbuttonpress()
