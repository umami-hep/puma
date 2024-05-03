""" Computes a vertex fit on batched track data using Billoir algorithm.

Results in singular exported function: billoir_vertex_fit. 
    This computes a vertex fit on batched track data according to user-provided
        track weights and vertex seed.
    The function is wrapped into a differentiable jax function and can be placed inside
        any larger jax / flax function.

code follows: https://www.sciencedirect.com/science/article/pii/0168900292908593
This is a modified version where the backward propagation functions are commented out. The original script is stored at https://github.com/rachsmith1/NDIVE/blob/main/diffvert/utils/billoir_vertex_fit.py
"""
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


def get_qmeas(track):
    """ return measured perigee track parameters for given track

    Args:
        track: dim 'num_track_inputs' sized array
    Returns:
        'num_perigee_params' x 1 array containing track measurements
    """ 
    d0     = track[0]
    z0     = track[1]
    phi    = track[2] 
    theta  = track[3]
    rho    = track[4]
    q = jnp.stack([d0, z0, phi, theta, rho]).reshape(-1, 1)

    #print("I finished estimated the perigee track parameters for a given track")
    return q


def get_qpred(rv, pv):
    """ return predicted track parameters given vertex of measurement and predicted mtm

    Extrapolates track parameters from fit vertex to origin
    Args:
        rv: x, y, z coordinates for vertex 
        pv: momentum of track in phi, theta, rho coordinates
    Returns:
        'num_perigee_params' x 1 array containing extrapolated track params
    """
    phiv = pv[0]
    theta = pv[1]
    rho = pv[2]

    Q = rv[0]*jnp.cos(phiv) + rv[1]*jnp.sin(phiv)
    R = rv[1]*jnp.cos(phiv) - rv[0]*jnp.sin(phiv)

    h1 = -R - Q**2 * (rho)/2
    h2 = rv[2] - Q*(1-R*(rho))/jnp.tan(theta)
    h3 = phiv - Q*(rho)
    h4 = theta
    h5 = rho
    h = jnp.stack([h1,h2,h3,h4,h5])
    h = jnp.reshape(h,(-1,1))

    
    return h


def get_cov(track):
    """ return measured errors of track parameters
    
    Args:
        track: dim 'num_track_inputs' sized array
    Returns:
        'num_track_params' x 1 array containing track measurement errors
    """
    d0_error     = track[5]
    z0_error     = track[6]
    phi_error    = track[7] 
    theta_error  = track[8]
    rho_error    = track[9] 
    track_error = jnp.stack([d0_error, z0_error, phi_error, theta_error, rho_error])

    #print("Shape covariance")
    #print(track_error.shape)

    return abs(track_error)


def calculate_chi2(tracks,fitvars,we):
    """ calculate chi2 of vertex fit for given billoir fit and weights

    Args:
        tracks: 'num_tracks' x 'num_track_params' input tracks
        fitvars: '3+3*n_tracks' fit of vertex fit and track mtm fit
        we: 'n_tracks' array of weights for tracks affecting chi2
    Returns:
        chi2 value of fit
    """
    n_trks = tracks.shape[0]
    r = fitvars[0:3]
    p = fitvars[3:].reshape(n_trks,3)

    t = tracks
    w = we
    chi2 = 0

    for i in range(n_trks):
        qmeas = get_qmeas(t[i])
        qpred = get_qpred(r, p[i])
        G = jnp.diag((1./get_cov(t[i])**2))
        dq = qmeas - qpred
        chi2 = chi2 + (jnp.transpose(dq) @ G @ dq) * w[i]

    return chi2


@jax.jit
def billoir_forward(tracks,weights,seed):
    """ fit a vertex to tracks given weights and initial vertex guess

    Args:
        tracks: 'n_tracks' x 'n_perigee_params' array of tracks parameterized around origin
        weights: 'n_tracks' array of track weights for vertex contribution (scales covariance)
        seed: length 3 array of initial vertex guess for billoir fit in cartesian coords
    Returns:
        vertex_fit: length 3 array of vertex x,y,z
        vtx_fit_covariance: 3x3 array of fit covariacne
        fit_chi2: chi2 value of the fit (calculated from input covariances)
        mtm_fit: 'n_tracks' x 3 array of fit for track mtm at vertex fit (used in backwards pass)
    """
    n_trks = tracks.shape[0]

    #print("Billoir forward tracks")
    #print(tracks.shape)
        
    def getA_B(theta,phiv,rho,Q,R):

        c = jnp.cos(phiv)
        s = jnp.sin(phiv)
        t = 1./jnp.tan(theta)
        useful_zeros = jnp.zeros(1)
        useful_ones = jnp.ones(1)

        A_1 = jnp.stack([s, -c, useful_zeros])
        A_2 = jnp.stack([-t*c, -t*s, useful_ones])
        A_3 = jnp.stack([-rho*c, -rho*s, useful_zeros])
        A_4 = jnp.stack([useful_zeros, useful_zeros, useful_zeros])
        A_5 = jnp.stack([useful_zeros, useful_zeros, useful_zeros])

        A = jnp.stack([A_1, A_2, A_3, A_4, A_5])

        B_1 = jnp.stack([Q, useful_zeros, -(Q**2)/2])
        B_2 = jnp.stack([-R*t, Q*(1+t**2), Q*R*t])
        B_3 = jnp.stack([useful_ones, useful_zeros, -Q])
        B_4 = jnp.stack([useful_zeros, useful_ones, useful_zeros])
        B_5 = jnp.stack([useful_zeros, useful_zeros, useful_ones])

        B = jnp.stack([B_1, B_2, B_3, B_4, B_5])

        return A, B

    def get_per_track(rv, pv, track):

        qmeas = get_qmeas(track)

        phiv = pv[0]
        theta = pv[1]
        rho = pv[2]

        Q = rv[0]*jnp.cos(phiv) + rv[1]*jnp.sin(phiv)
        R = rv[1]*jnp.cos(phiv) - rv[0]*jnp.sin(phiv)

        A, B = getA_B(theta,phiv,rho,Q,R)
        A = jnp.squeeze(A)
        B = jnp.squeeze(B)

        h = get_qpred(rv, pv)

        rv = jnp.reshape(rv,(3,1))
        pv = jnp.reshape(pv,(3,1))

        G = jnp.diag(1./get_cov(track)**2)
     
        Di = jnp.transpose(A) @ G @ B     ## Laura, code complaints here because of the transpose bc dot_general requires contracting dimensions to have the same shape, got (5,) and (1,).
        D0 = jnp.transpose(A) @ G @ A
        E = jnp.transpose(B) @ G @ B
        W = jnp.linalg.pinv(E, hermitian=True)
        q_c = qmeas - (h - A @ rv - B @ pv)

        return A,B,Di,D0,G,W,q_c

    def make_estimate(point, mom, weight):
        """ do singular billoir fit, linearized around point, mom """
        def per_track_v_estimate(i, params):
            v, cov = params
            A,B,Di,D0,G,W,q_c = get_per_track(point, mom[i], tracks[i])
            v = v + (jnp.transpose(A) @ G @ (jnp.eye(5) - B @ W @ jnp.transpose(B) @ G) @ q_c) * weight[i]
            cov = cov + (D0 - Di @ W @ jnp.transpose(Di)) * weight[i]
            return (v,cov)

        params = jax.lax.fori_loop(
            0,
            n_trks,
            per_track_v_estimate,
            (jnp.zeros((3,1), dtype=jnp.float64),jnp.zeros((3,3), dtype=jnp.float64))
        )

        vn_wo_Cn, Cn_inv = params

        Cn = jnp.linalg.pinv(Cn_inv, hermitian=True)
        vn = Cn @ vn_wo_Cn

        mn = jnp.zeros((n_trks,3,1), dtype=jnp.float64)
        def per_track_p_estimate(i, params):
            m = params
            A,B,Di,D0,G,W,q_c = get_per_track(point, mom[i], tracks[i])
            mi = W @ jnp.transpose(B) @ G @ (q_c - A @ vn)
            m = m.at[i].set(mi)
            return m

        mn = jax.lax.fori_loop(0, n_trks, per_track_p_estimate, mn)

        return jnp.reshape(vn,(3,)), mn, Cn

    def estimate_vertex(weights):
        """ Do vertex fit for given track weights (iterative billoir). """
        vertex = seed
        cov = jnp.zeros((3,3), dtype=jnp.float64)
        track_mtm = jnp.zeros((n_trks,3,1), dtype=jnp.float64)

        # initialize track momentum guesses to perigee rep around 0's
        for i in range(n_trks):
            phi0   = jnp.array([tracks[i][2]])  
            theta  = jnp.array([tracks[i][3]])
            rho    = jnp.array([tracks[i][4]]) 

            pv = jnp.stack([phi0, theta, rho])
            pv = jnp.reshape(pv,(3,1))

            track_mtm = track_mtm.at[i].set(pv)

        for i in range(10):
            vertex, track_mtm, cov = jax.lax.stop_gradient(
                make_estimate(vertex, track_mtm, weights)
            )

        fit_vars = jnp.concatenate((vertex, jnp.ravel(track_mtm))).reshape(-1)

        chi2 = calculate_chi2(tracks,fit_vars,weights)

        return vertex, cov, chi2, track_mtm

    #print("Estimating vertex from weights:")
    #print(weights)
    
    vertex_fit, vtx_fit_covariance, fit_chi2, mtm_fit = estimate_vertex(weights)

    return vertex_fit, vtx_fit_covariance, fit_chi2, mtm_fit


@jax.jit
def billoir_gradient(tracks, weights, vertex_fit, mtm_fit):
    """ compute gradient at minimization of chi2 of vertex output w.r.t track weights
    
    linearization for chi2 is around fit vertex and momentum
    Args:
        tracks: 'n_tracks' x 'n_track_params' array of input tracks
        weights: 'n_tracks' length array of weight for tracks in vertex fit
        vertex_fit: length 3 array of fit vertex coordinates
        mtm_fit: 'n_tracks' x 3 x 1 array of fit momentum variables at output vertex
    Returns: 
        3 x 'n_tracks' array of vertex coord gradient w.r.t each input track weight
    """
    n_trks = tracks.shape[0]
    vp = jnp.concatenate((vertex_fit, jnp.ravel(mtm_fit))).reshape(-1)

    chi2_hessian = jax.hessian(
        calculate_chi2,
        argnums=(1,2),
        has_aux=False,
    )(tracks, vp, weights)

    # use implicit function theorem (main optimization as a layer trick)
    grad_v = -jnp.linalg.inv(chi2_hessian[0][0]) @ chi2_hessian[0][1]
    grad_v = jnp.nan_to_num(grad_v, nan=0., posinf=1e200, neginf=-1e200)

    grad_v = grad_v.reshape(3+3*n_trks,n_trks)

    # only interested in vertex output, not predicted momentum
    grad_v = grad_v[0:3,:]
    return grad_v


billoir_fit_vmap = jax.jit(jax.vmap(billoir_forward, in_axes=(0, 0, 0), out_axes=(0, 0, 0, 0)))
billoir_grad_vmap = jax.jit(jax.vmap(billoir_gradient, in_axes=(0,0,0,0), out_axes=(0)))


@jax.custom_vjp
def vertex_layer(tracks, weights, seed):
    """ Call billoir vertexing, no gradient requested

    Calls the billoir submodule on tracks, weights, seeds

    Args:
        tracks: 'n_tracks' x 'n_track_params' sized input array
        weights: 'n_tracks' of weights of track importance for vertex fit
        seed: length 3 initial vertex guess
    Returns:
        tuple of vertex prediction, covariance of prediction, chi2 of prediction
    """
    return billoir_fit_vmap(tracks, weights, seed)[0:3]


def vertex_layer_forward(tracks, weights, seed):
    """ forward method for differentiable billoir

    Args:
        tracks: track input of dim 'num_jets' x 'num_tracks' x 'num_track_vars'
        weights: weights for track chi2 contribution of dim 'num_jets' x 'num_tracks'
        seed: initial guess for vertex of dim 'num_jets' x 3
    Returns:
        model outputs, and residuals used to calculate derivative
    """
    fit_outs = billoir_fit_vmap(tracks, weights, seed)
    layer_outs = fit_outs[0:3]
    # residuals are everything needed to compute derivative in bwd step
    #    contains tracks, weights, output vertex fit, and output mtm fit
    layer_residuals = (tracks, weights, fit_outs[0], fit_outs[3])
    return layer_outs, layer_residuals


def vertex_layer_backward(res, dy):
    """ backward method for computing billoir gradient w.r.t track weights

    Args:
        res: residuals from fwd method (taken here to be inputs to the model)
        dy: tangents of the outputs of the function (gradient to back-prop)
    Returns:
        derivative of vertex position with respect to track weights
        also returns emtpy dict and two None to match params, inputs shape as required
            by nn.custom_jvp
    """
    tracks, weights, vertex_fit, mtm_fit = res
    n_trks = tracks.shape[1]
    grad_v = billoir_grad_vmap(tracks, weights, vertex_fit, mtm_fit)

    grad_output_v = jnp.reshape(dy[0], (-1, 1, 3))
    grad_v = jnp.reshape(grad_v, (-1, 3, n_trks))

    batch_grads_v = jnp.einsum("bij,bjk->bik", grad_output_v, grad_v).reshape(-1, n_trks)
    batch_grads_w = batch_grads_v

    return None, batch_grads_w, None


vertex_layer.defvjp(vertex_layer_forward, vertex_layer_backward)

#@jax.jit
def billoir_vertex_fit(tracks, weights, seed):
    """ Differentiable billoir vertex fitter for batched jet data.

    Args:
        tracks: 'num_jets' x 'num_tracks' x 'num_track_params' input track data
        seed: 'num_jets' x 3 initial vertex seed in cartesian coordinates
    """
    return vertex_layer(tracks, weights, seed)
