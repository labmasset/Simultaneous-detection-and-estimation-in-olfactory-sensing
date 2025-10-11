import numpy as np
import matplotlib.pyplot as plt
import time
from math import gamma


# -------------------------------
# Set neuron parameters
# -------------------------------
nSens = 300       # number of sensors
nOdor = 500        # number of odors

r0 = 1             # baseline activity
nLow = 0         # number of low odors
nHigh = 5        # number of high odors

cLow = 0        # low concentration
cHigh = 40       # high concentration

a = 0.5          # regularization strength (unused in subsequent code)
normFlag = 'max' # flag (unused)

# -------------------------------
# Set simulation parameters
# -------------------------------
dt = 1e-5        # timestep in seconds
tMax = 1         # total simulation time in seconds
tOnLow = 0.000   # stimulus onset time for low concentration
tOnHigh = 0.200  # stimulus onset time for high concentration
tOff = 0.800     # stimulus offset time

threshold = 0.5
# np.random.seed(0)  # Set seed for reproducibility
# A = np.random.gamma(shape=0.37, scale=0.36, size=(nSens, nOdor))

# Affinity matrix A
A = (np.random.rand(nSens, nOdor) < 0.1).astype(float)
# np.save("saved/A.npy", A)
# A = np.load("saved/A.npy")
A = A / np.max(A, axis=1, keepdims=True)

# -------------------------------
# Generate the sensory scene
# -------------------------------
start_time = time.time()

# Time vector
t = np.arange(0, tMax, dt)  # shape: (nT,)
nT = len(t)

# Check ordering of onset and offset times
if not (tOnLow <= tOnHigh <= tOff <= tMax):
    raise ValueError('Invalid onset and offset times')

# Generate concentration matrix (4 x nOdor)
# Row 1: zeros; Row 2: low concentration; Row 3: low then high; Row 4: zeros.
row1 = np.zeros(nOdor)
row2 = np.concatenate([np.full(nLow, cLow), np.zeros(nOdor - nLow)])
row3 = np.concatenate([np.full(nLow, cLow), np.full(nHigh, cHigh), np.zeros(nOdor - nLow - nHigh)])
row4 = np.zeros(nOdor)
cMat = np.vstack([row1, row2, row3, row4])

# Generate presence matrix (4 x nOdor): ones if odor is present, zeros otherwise.
row1_o = np.zeros(nOdor)
row2_o = np.concatenate([np.ones(nLow), np.zeros(nOdor - nLow)])
row3_o = np.concatenate([np.ones(nLow), np.ones(nHigh), np.zeros(nOdor - nLow - nHigh)])
row4_o = np.zeros(nOdor)
oMat = np.vstack([row1_o, row2_o, row3_o, row4_o])

# Form the Poisson rate matrix: lambdaMat = r0 + (cMat .* oMat) * A'
# (cMat * oMat is elementwise multiplication; then matrix multiply with A transpose)
lambdaMat = r0 + (cMat * oMat) @ A.T  # resulting shape: (4, nSens)
# print("cMat * oMat", cMat * oMat)
# Sample from Poisson distribution for sensor activities (4 x nSens)
pMat = np.random.poisson(lam=lambdaMat)
# pMat = lambdaMat

# Expand the Poisson matrix in time to create sMat (nT x nSens)
# Each time interval uses a different row of pMat
sMat = ((t < tOnLow)[:, None] * pMat[0, :]) + \
    (((t >= tOnLow) & (t < tOnHigh))[:, None] * pMat[1, :]) + \
    (((t >= tOnHigh) & (t < tOff))[:, None] * pMat[2, :]) + \
    ((t >= tOff)[:, None] * pMat[3, :])

# Generate the true concentration matrix (nT x nOdor)
# For odors with low concentration (nLow) and high concentration (nHigh)
cMatTrue = np.hstack([
    cLow * ((t >= tOnLow) & (t <= tOff))[:, None].astype(float) * np.ones((nT, nLow)),
    cHigh * ((t >= tOnHigh) & (t <= tOff))[:, None].astype(float) * np.ones((nT, nHigh)),
    np.zeros((nT, nOdor - nLow - nHigh))
])
oMatTrue = np.hstack([
    ((t >= tOnLow) & (t <= tOff))[:, None].astype(float) * np.ones((nT, nLow)),
    ((t >= tOnHigh) & (t <= tOff))[:, None].astype(float) * np.ones((nT, nHigh)),
    np.zeros((nT, nOdor - nLow - nHigh))
])
# smoothing the sensory scene
# sMat = sMat.astype(np.float32)
# k = half_gaussian_kernel(sigma=0.0025, dt=1e-5)
# for i in range(sMat.shape[1]):
#     sMat[:, i] = np.convolve(sMat[:, i], k, mode='full')[:sMat.shape[0]]

# print(f"Generated sensory scene in {time.time() - start_time:.6f} seconds")

# -------------------------------
# Define the MAP estimate function
# -------------------------------
def MAP_estimate(s, nOdor, r0, A, iter_num, gamma_val, w, lambda_val, dt, tau_c, tau_p, tau_h, alpha, beta):
    """
    s: sensor data array with shape (iter_num, nSens)
    nOdor: number of odors
    r0: baseline activity
    A: affinity matrix (nSens x nOdor)
    iter_num: number of iterations (should match number of timesteps)
    gamma_val: gain parameter
    w: parameter used for computing u_prior
    lambda_val: lambda parameter in the update equations
    dt: time step
    tau_c, tau_p: time constants for concentration and presence
    alpha, beta: parameters for the MAP update
    """
    # Initialize u as ones (nOdor,)
    u = np.ones(nOdor)
    # TODO: Check
    # u_prior = -2
    u_prior = -np.log((1 - w) / w) / gamma_val
    # Find u_prior such that |1/(1+exp(-gamma_val*x)) - w| < 1e-2
    # x_vals = np.arange(-20, 10, 0.001)
    # for x in x_vals:
    #     if abs(1/(1+np.exp(-gamma_val*x)) - w) < 1e-2:
    #         u_prior = x
    #         break
    print("u_prior:", u_prior)
    
    # Initialize c from a gamma distribution; in MATLAB: gamrnd(6,10,nOdor,1)
    c = np.random.gamma(shape=6, scale=4, size=nOdor)
    # c = np.zeros(nOdor, dtype=float) + 1.  # Set all concentrations to zero initially

    u = u * u_prior
    # u = u * 0

    # Lists to store dynamics
    DU = []
    DC = []
    U = []
    C = []
    M = []
    M_gated = []
    AM = []
    AM_gated = []

    PAM = []
    PAM_gated = []
    Theta = []
    # Initialize m and m_gated
    ep = 1e-3
    theta = 1 / (1 + np.exp(-gamma_val * u))
    theta_m = theta.copy()
    # Apply threshold: if theta_m < 0.2, set to 1e-5; if >= 0.2, set to 1
    theta_m[theta_m < threshold] = 1e-5
    theta_m[theta_m >= threshold] = 1
    m_gated = s[0,:] / (r0 + A @ (c * theta_m))
    m = s[0,:] / (r0 + A @ (c * theta))
    z =  (alpha - 1) / (c + ep)
    m = np.zeros(nSens, dtype=float)   # Set all concentrations to zero initially
    m_gated = np.zeros(nSens, dtype=float)   # Set all concentrations to zero initially
    print("m:", m.shape)
    print("loop")

    # Iterate and update estimates
    for i in range(iter_num):
        # Compute theta from u
        theta = 1 / (1 + np.exp(-gamma_val * u))
        theta_m = theta.copy()
        # Apply threshold
        theta_m[theta_m < threshold] = 1e-5
        theta_m[theta_m >= threshold] = 1.

        # For the current iteration, get the sensor data vector
        s_i = s[i, :]  # shape: (nSens,)

        dc = ( theta_m * ( A.T @ (m_gated- 1)    - beta ) ) * dt
        du = gamma_val * ( theta * (1 - theta) * ( c * (A.T @ (m - 1)) + np.log(w/(1-w)))- 2 * theta + 1 ) * dt
        dm = s_i - m * (r0 + A @ (c * theta))
        dm_gated = s_i - m_gated * (r0 + A @ (c * theta_m))
        dz = (alpha - 1) - z * (c+ep)

        c = c + dc / tau_c + np.sqrt(2 * dt / tau_c) * np.random.randn(nOdor) 
        u = u + du / tau_p + np.sqrt(2 * dt / tau_p) * np.random.randn(nOdor) 
        m = m + dm*dt / (tau_h)
        m_gated = m_gated + dm_gated * dt / (tau_h)
        # print(m_gated-m)
        z = z + dz * dt / (tau_h)

        c[c < 0] = ep

        # Store updates for analysis
        DU.append(du)
        DC.append(dc)
        U.append(u.copy())
        C.append(c.copy())
        Theta.append(theta.copy())
        PAM.append(  theta * (1 - theta) * c * ( A.T @ m.copy()))
        PAM_gated.append(theta_m * (A.T @ m_gated.copy()))

        AM.append(   A.T @ m.copy())
        AM_gated.append( A.T @  m_gated.copy()) 

        M.append(m.copy())
        M_gated.append(m_gated.copy())

    # Convert stored lists to arrays (each with shape (nOdor, iter_num))
    DU = np.column_stack(DU)
    DC = np.column_stack(DC)
    U = np.column_stack(U)
    C = np.column_stack(C)
    M = np.column_stack(M)
    M_gated = np.column_stack(M_gated)
    Theta = np.column_stack(Theta)
    AM = np.column_stack(AM)
    AM_gated = np.column_stack(AM_gated)
    PAM = np.column_stack(PAM)
    PAM_gated = np.column_stack(PAM_gated)
    
    return c, theta, u, DU, DC, U, C, Theta, M, M_gated, PAM, PAM_gated, AM, AM_gated



print("Starting MAP estimate...")
start_time = time.time()

iter_num = int(1e5)  # number of iterations (should equal nT)
w = 0.01
gamma_val = 1.
lambda_val = 0.1
beta = 1 / 10.
alpha = 5
tau_c = 0.03 
tau_p = 0.03
tau_h = 0.02 


# print(c)
# Run the MAP_estimate function. Note: sMat shape is (nT, nSens) and iter_num equals nT.
c_final, theta_final, u_final, DU, DC, U, C, Theta, M, M_gated, PAM, PAM_gated, AM, AM_gated  = MAP_estimate(sMat, nOdor, r0, A, iter_num,
                                                                    gamma_val, w, lambda_val,
                                                                    dt, tau_c, tau_p,tau_h, alpha, beta)
print(f"MAP estimate completed in {time.time() - start_time:.6f} seconds")
# np.save(f'M_{cHigh}.npy', M)
# np.save(f'M_gated_{cHigh}.npy', M_gated)
# np.save(f'AM_{cHigh}.npy', AM)
# np.save(f'AM_gated_{cHigh}.npy', AM_gated)
# np.save(f'PAM_{cHigh}.npy', PAM)
# np.save(f'PAM_gated_{cHigh}.npy', PAM_gated)
# -------------------------------
# Plotting the results
# -------------------------------
# Create a time axis for plotting (same as the timestep vector)
print("Plotting results...")
timestep = np.arange(0, 1, dt)
darkRed = "#D95319"
fig_size = (8,6)
# Plot dynamics for concentration
plt.figure(figsize=fig_size)
# Plot trajectories for odors not present (indices from nHigh to nOdor-1)
for i in range(nHigh, min(nOdor, 100)): # for clarity, only plot last 200 odors
    plt.plot(timestep, C[i, :], color=[0.7, 0.7, 0.7], linewidth=0.5)
# Plot trajectories for odors present (first nHigh odors)
for i in range(nHigh):
    plt.plot(timestep, C[i, :], color=darkRed, linewidth=1.5)
# Plot the true concentration for the first odor (as an example)
plt.plot(timestep, cMatTrue[:, 0], '--', color='darkRed', linewidth=3)
# plt.title('Dynamics for concentration', fontsize=16)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Concentration", fontsize=20)

h1, = plt.plot([], [], color=darkRed, linewidth=2)
h2, = plt.plot([], [], 'k', linewidth=0.5)
h3, = plt.plot([], [], '--', color='darkRed', linewidth=3)
plt.legend([h1, h2, h3], ['odors present', 'odors not present', 'true value'], loc='best',frameon=False, fontsize=20)
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)

plt.tight_layout()
plt.savefig("dynamics_concentration_split.png")
plt.show()

# Plot dynamics for presence (u)
plt.figure(figsize=fig_size)
# Sample every 100th timestep for clarity
timestep_sample = timestep[::100]
for i in range(nHigh,  min(nOdor, 100)):
    plt.plot(timestep_sample, U[i, ::100], color=[0.7, 0.7, 0.7], linewidth=0.5)
for i in range(nHigh):
    plt.plot(timestep_sample, U[i, ::100], color=darkRed, linewidth=1.5)
# plt.title('Dynamics for presence (u)', fontsize=16)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Presence (u)", fontsize=20)
h1, = plt.plot([], [], color=darkRed, linewidth=1.5)
h2, = plt.plot([], [], 'k', linewidth=0.5)
plt.legend([h1, h2], ['odors present', 'odors not present'], loc='best',frameon=False, fontsize=20)
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)

# plt.ylim([-2., 2.])
plt.tight_layout()
plt.savefig("dynamics_presence_split.png")
plt.show()

# Plot dynamics for probability (Theta)
plt.figure(figsize=fig_size)
timestep_sample = timestep[::100]
for i in range(nHigh, min(nOdor, 100)):  # for odors not present (MATLAB used 500 as end index)
    plt.plot(timestep_sample, Theta[i, ::100], color=[0.7, 0.7, 0.7], linewidth=0.5)
for i in range(nHigh):
    plt.plot(timestep_sample, Theta[i, ::100], color=darkRed, linewidth=1.5)
plt.plot(timestep_sample, threshold * np.ones_like(timestep_sample), '--', color="darkred", linewidth=3)
# plt.title('Dynamics for presence (p)', fontsize=16)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Presence (p)", fontsize=20)
h1, = plt.plot([], [], color=darkRed, linewidth=1.5)
h2, = plt.plot([], [], 'k', linewidth=0.5)
h3, = plt.plot([], [], '--', color="darkred", linewidth=3)
plt.legend([h1, h2, h3], ['odors present', 'odors not present', 'threshold'], loc='best',frameon=False, fontsize=20)
plt.ylim([0, 1.25])
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)

plt.tight_layout()

plt.savefig("dynamics_probability_split.png")
plt.show()

plt.figure(figsize=fig_size)
print("M shape:", M.shape)
timestep_sample = timestep[::]

for i in range(nSens):
    plt.plot(timestep_sample, M[i, ::], linewidth=1.5)

plt.xlabel("Time", fontsize=20)
plt.ylabel("h", fontsize=20)
h1, = plt.plot([], [], color=darkRed, linewidth=1.5)
h2, = plt.plot([], [], 'k', linewidth=0.5)
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)
plt.title('soft threshold', fontsize=20)
plt.tight_layout()
plt.savefig("h_nongated.png")

plt.figure(figsize=fig_size)
print("M shape:", M_gated.shape)
timestep_sample = timestep[::]

for i in range(nSens):
    plt.plot(timestep_sample, M_gated[i, ::], linewidth=1.5)
plt.xlabel("Time", fontsize=20)
plt.ylabel("h_hard_gated", fontsize=20)
h1, = plt.plot([], [], color=darkRed, linewidth=1.5)
h2, = plt.plot([], [], 'k', linewidth=0.5)
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)
plt.title('hard threshold', fontsize=20)
plt.tight_layout()

plt.savefig("h_gated.png")



plt.figure(figsize=fig_size)
print("M shape:", AM.shape)
timestep_sample = timestep[::100]

for i in range(nOdor):
    plt.plot(timestep_sample, AM[i, ::100],  linewidth=1.5)

plt.xlabel("Time", fontsize=20)
plt.ylabel("Ah(nongated)", fontsize=20)
h1, = plt.plot([], [], color=darkRed, linewidth=1.5)
h2, = plt.plot([], [], 'k', linewidth=0.5)
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)

plt.tight_layout()

plt.savefig("Ah_nongated.png")

plt.figure(figsize=fig_size)
# print("M shape:", M.shape)
timestep_sample = timestep[::100]

for i in range(nOdor):
    plt.plot(timestep_sample, AM_gated[i, ::100], linewidth=1.5)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Ah_gated", fontsize=20)
h1, = plt.plot([], [], color=darkRed, linewidth=1.5)
h2, = plt.plot([], [], 'k', linewidth=0.5)
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)

plt.tight_layout()

plt.savefig("Ah_gated.png")

plt.figure(figsize=fig_size)
print("M shape:", M.shape)
timestep_sample = timestep[::100]

for i in range(nOdor):
    plt.plot(timestep_sample, PAM[i, ::100], linewidth=1.5)

plt.xlabel("Time", fontsize=20)
plt.ylabel("p.*(1-p).*c.*(Ah_nongated)", fontsize=20)
h1, = plt.plot([], [], color=darkRed, linewidth=1.5)
h2, = plt.plot([], [], 'k', linewidth=0.5)
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)

plt.tight_layout()

plt.savefig("pAh_nongated.png")

plt.figure(figsize=fig_size)
print("M shape:", M.shape)
timestep_sample = timestep[::100]

for i in range(nOdor):
    plt.plot(timestep_sample, PAM_gated[i, ::100], linewidth=1.5)
plt.xlabel("Time", fontsize=20)
plt.ylabel("p_gated .* (Ah_gated)", fontsize=20)
h1, = plt.plot([], [], color=darkRed, linewidth=1.5)
h2, = plt.plot([], [], 'k', linewidth=0.5)
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)

plt.tight_layout()

plt.savefig("pAh_gated.png")

## plot average M, M_gated
plt.figure(figsize=fig_size)
print("M shape:", M.shape)


plt.plot(timestep, np.mean(M, axis=0), linewidth=1.5, label='soft threshold')
plt.plot(timestep, np.mean(M_gated, axis=0), linewidth=1.5, label='hard threshold')
plt.xlabel("Time", fontsize=20)
plt.ylabel("Average h", fontsize=20)
h1, = plt.plot([], [], color=darkRed, linewidth=1.5)
h2, = plt.plot([], [], 'k', linewidth=0.5)
ax = plt.gca()
for spine in [ 'bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Increase tick label size and tick width
plt.tick_params(axis='both', which='major', labelsize=20, width=5)
plt.legend(loc='best', frameon=False, fontsize=20)
plt.tight_layout()
plt.savefig(f"average_h_P{cHigh}.png")

