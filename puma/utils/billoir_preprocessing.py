import numpy as np
import jax.numpy as jnp
import h5py
from ftag.hdf5 import H5Reader # I use ftag tools to read the file

from puma.utils.vertexing import  build_vertices, clean_reco_vertices, clean_truth_vertices


def ListVariables(file_path):
    with h5py.File(file_path, "r") as f:
        print(f.keys())
        for k in list(f.keys()):
            print(k)
            print(f[k].dtype.fields.keys())

    return

def MaskTracks(my_data, n_jets, n_tracks):

    n_real_tracks = np.repeat(my_data["jets"]["n_tracks"], n_tracks).reshape(n_jets, n_tracks) # This is needed because jets have a different format than tracks
    track_indices = np.tile(
         np.arange(0,n_tracks,dtype=np.int32),
         n_jets,
    ).reshape(n_jets, n_tracks)
    
    track_mask = np.where(track_indices < n_real_tracks, 1, 0)
    
    return track_mask, n_real_tracks
    
def TransformData(my_data, good_jets, n_tracks=40, drop_unrelated_hadrons = True):

    # Function to calculate the track parameters in the perigree representation.
    # Returns data x with the format n_jets x n_tracks x n_parameters
    # The n_parameters will first have the variables needed for the billoir fit, some will have to be build by hand because not everything is available

    n_jets, max_tracks = my_data["tracks"].shape

    track = my_data["tracks"][:, 0:n_tracks]
    jet = my_data["jets"][:] # Only needed if you need to calculate the track phi from dphi.

    # Start by getting a mask of the real tracks

    # Get real tracks
    track_mask, n_real_tracks = MaskTracks(my_data, n_jets, n_tracks)

    # Compute Input Variables for Billoir Vertex Fit
    ### set parameters for dummy tracks to 1. They will be masked out by the track weight and if you choose a very low value the fit will not work well.

    d0 = jnp.where(track_mask == 0, 1, -track["d0RelativeToBeamspot"])  # d0RelativeToBeamspot # NEGATIVE for Billoir fit (different definitions between ATLAS and the billoir paper)
    z0 = jnp.where(track_mask == 0, 1, track["z0RelativeToBeamspot"]) 
    
    jet_phi = jnp.repeat(jet["phi"], n_tracks).reshape(n_jets, n_tracks) # This is needed because jets have a different format than tracks
    #phi = track["phi"] # take track phi directly
    # if you calculate track phi from dphi you need the following 3 lines
    phi = jet_phi + my_data["tracks"]["dphi"]
    phi = np.where(phi < -np.pi, 2*np.pi + (jet_phi + my_data["tracks"]["dphi"]), phi)
    phi = np.where(phi > np.pi,  -2*np.pi + (jet_phi + my_data["tracks"]["dphi"]), phi)

    phi = jnp.where(track_mask == 0, 1, phi)

    theta  = jnp.where(track_mask == 0, 1, track["theta"])
    rho    = jnp.where(track_mask == 0, 1, -track["qOverP"]*2*0.2299792*0.001/jnp.sin(track["theta"])) #NEGATIVE for Billoir fit (different definitions between ATLAS and the billoir paper)

    d0_error     = jnp.where(track_mask == 0, 1, track["d0RelativeToBeamspotUncertainty"])
    z0_error     = jnp.where(track_mask == 0, 1, track["z0RelativeToBeamspotUncertainty"])

    phi_error    = jnp.where(track_mask == 0, 1, track["phiUncertainty"])
    theta_error  = jnp.where(track_mask == 0, 1, track["thetaUncertainty"])

    rho_error    = jnp.where(track_mask == 0, 1, jnp.sqrt((2*0.2299792*0.001/jnp.sin(track["theta"]) * track["qOverPUncertainty"])**2  + (track["qOverP"]*2*0.2299792*0.001/(jnp.sin(track["theta"])**2 * jnp.cos(track["theta"])) * track["thetaUncertainty"] )**2) )

    track_origin = jnp.where(track_mask == 0, 1, track["GN2v01_aux_TrackOrigin"])
    track_vertex = jnp.where(track_mask == 0, 1, track["GN2v01_aux_VertexIndex"])
     
    x = jnp.stack([d0, z0, phi, theta, rho, d0_error, z0_error, phi_error, theta_error, rho_error, track_origin, track_vertex, n_real_tracks], axis = 2)

    if drop_unrelated_hadrons == True:
        x = x[good_jets]
        track = track[good_jets]
        track_mask = track_mask[good_jets]

    return x, track, track_mask



# Get the vertex indices and track weights! Which tracks belong to which vertex according? The track origin is used for the cleaning
def GetTrackWeights(track_data, incl_vertexing=False, truth=False, max_sv=1):

    if truth:
        raw_vertex_index = track_data["ftagTruthVertexIndex"] # your raw vertex
        track_origin = track_data["ftagTruthOriginLabel"]

    else:
        # Reco Level
        raw_vertex_index = track_data["GN2v01_aux_VertexIndex"] # your raw vertex
        track_origin = track_data["GN2v01_aux_TrackOrigin"]

    # Now clean vertices
    vertex_index  = raw_vertex_index.copy()

    # Prepare mask for filling up
    #dummy_track_weights = jnp.zeros((vertex_index.shape[0], max_sv, n_tracks))
    track_weights = jnp.zeros((vertex_index.shape[0], max_sv, track_data.shape[1]))

    #track_weights = jnp.where(dummy_track_weights == 0, 0, dummy_track_weights) #why np.nan?
    
    for i in range(track_data["GN2v01_aux_VertexIndex"].shape[0]):

        if truth:
            vertex_index[i] = clean_truth_vertices(
                vertex_index[i], track_origin[i], incl_vertexing=incl_vertexing
            )
        
        else:
            vertex_index[i] = clean_reco_vertices(
                vertex_index[i], track_origin[i], incl_vertexing=incl_vertexing
            )

        vertices = build_vertices(vertex_index[i]) # Convert indices to true/false

        for j in range(0, max_sv):
            try:
                track_weights = track_weights.at[i, j].set(vertices[j])
            except:
                continue


    return track_weights, vertex_index


def LoadDataset(file_path,kinematic_cuts,  n_jets=-1, n_tracks=40):    
    
    track_var = ["d0", "z0SinTheta", "dphi", "d0Uncertainty", "z0SinThetaUncertainty", "phiUncertainty", "thetaUncertainty", "qOverPUncertainty", "qOverP", "deta", "theta", "dphi"] # for vertex fit
    track_var += ["d0RelativeToBeamspot", "d0RelativeToBeamspotUncertainty","z0RelativeToBeamspot", "z0RelativeToBeamspotUncertainty",  "ftagTruthOriginLabel",  "GN2v01_aux_TrackOrigin", "GN2v01_aux_VertexIndex",  "ftagTruthVertexIndex", "ftagTruthParentBarcode"]
    track_var += ["JFVertexIndex", "pt"]
    
    jet_var = ["eventNumber","GN2v01_pb", "GN2v01_pc", "GN2v01_pu", "n_tracks", "jetPtRank", "phi", "eta", "HadronConeExclTruthLabelID", "HadronConeExclExtendedTruthLabelID", "HadronConeExclTruthLabelPdgId", "HadronConeExclTruthLabelLxy", "SV1_Lxy", "JetFitterSecondaryVertex_displacement2d", "SV1_L3d", "JetFitterSecondaryVertex_displacement3d", "JetFitter_nVTX", "mcEventWeight", "nPrimaryVertices"] # phi is needed for vertex fit if track phi is not available # v00 instead of v01
    
    jet_var +=['primaryVertexToBeamDisplacementX', 'primaryVertexToBeamDisplacementY', 'primaryVertexToBeamDisplacementZ', 'primaryVertexToTruthVertexDisplacementX', 'primaryVertexToTruthVertexDisplacementY', 'primaryVertexToTruthVertexDisplacementZ', 'truthPrimaryVertexX', 'truthPrimaryVertexY', 'truthPrimaryVertexZ', 'primaryVertexDetectorZ']
    jet_var += ['JetFitterSecondaryVertex_nTracks', 'JetFitter_nTracksAtVtx', 'SV1_masssvx', 'JetFitter_mass', 'JetFitterSecondaryVertex_mass']
    truth_hadrons = ['pt', 'mass', 'energy', 'eta', 'phi', 'deta', 'dphi', 'dr', 'displacementX', 'displacementY', 'displacementZ','Lxy', 'charge', 'flavour', 'pdgId', 'barcode', 'ftagTruthParentBarcode', 'valid',  'decayVertexDPhi', 'decayVertexDEta'] 
    
    
    ## read it!
    my_reader = H5Reader(file_path, precision="full", shuffle=False, batch_size=100)

    if n_jets == -1:
        my_data = my_reader.load({"jets": jet_var, "tracks" : track_var, "truth_hadrons" : truth_hadrons},  cuts=kinematic_cuts)
    else:
        my_data = my_reader.load({"jets": jet_var, "tracks" : track_var, "truth_hadrons" : truth_hadrons}, num_jets=n_jets, cuts=kinematic_cuts)

    return my_data
