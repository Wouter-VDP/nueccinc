# Columns for daughter frames:
col_pfp = ["pfnplanehits_U", "pfnplanehits_V", "pfnplanehits_Y", "pfnhits"]
col_trk = [
    "trk_score_v",
    "trk_distance_v",
    "trk_theta_v",
    "trk_phi_v",
    "trk_len_v",
    "trk_pid_chipr_v",
    "trk_pid_chimu_v",
    "trk_pid_chipr_v_v",
    "trk_pid_chimu_v_v",
    "trk_pid_chipr_u_v",
    "trk_pid_chimu_u_v",
    "trk_llr_pid_u_v",
    "trk_llr_pid_v_v",
    "trk_llr_pid_y_v",
    "trk_llr_pid_v",
    "trk_llr_pid_score_v",
]
col_shr = [
    "shr_dist_v",
    "shr_energy_y_v",
    "shr_openangle_v",
    "shr_tkfit_start_x_v",
    "shr_tkfit_start_y_v",
    "shr_tkfit_start_z_v",
    "shr_tkfit_theta_v",
    "shr_tkfit_phi_v",
    "shr_tkfit_dedx_u_v",
    "shr_tkfit_dedx_v_v",
    "shr_tkfit_dedx_y_v",
    "shr_tkfit_gap10_dedx_u_v",
    "shr_tkfit_gap10_dedx_v_v",
    "shr_tkfit_gap10_dedx_y_v",
    "shr_tkfit_nhits_v",
    "shr_tkfit_dedx_nhits_u_v",
    "shr_tkfit_dedx_nhits_v_v",
    "shr_tkfit_dedx_nhits_y_v",
    "shr_nclus0_v", 
    "shr_nclus0_v",
    "shr_nclus0_v",
    "shr_clushitfrac2_v"
    'shr_moliere_avg_v'
    'shr_moliere_rms_v'
]
col_backtracked = [
    "backtracked_pdg",
    "backtracked_e",
    "backtracked_completeness",
    "backtracked_purity",
    "backtracked_overlay_purity",
]
col_mc = [
    "true_nu_vtx_x",
    "true_nu_vtx_y",
    "true_nu_vtx_z",
    "true_nu_vtx_sce_x",
    "true_nu_vtx_sce_y",
    "true_nu_vtx_sce_z",
    "nu_e",
    "nu_pdg",
    "theta",
    "ccnc",
    "interaction",
    "weightSpline",
    "leeweight",
    
    "nu_completeness_from_pfp",
    "nu_purity_from_pfp"
]
col_event = [
    "run",
    "sub",
    "evt",
    "topological_score",
    "n_pfps",
    "n_showers",
    "n_tracks",
    "reco_nu_vtx_sce_x",
    "reco_nu_vtx_sce_y",
    "reco_nu_vtx_sce_z",
    "reco_nu_vtx_x",
    "reco_nu_vtx_y",
    "reco_nu_vtx_z",
    "crtveto",
    "crthitpe",
    "slclustfrac",
    "hits_ratio",
    "nu_flashmatch_score",
    'selected',
    "category",
    "nslice",
    "CosmicIP",
    "n_tracks_contained",
    "n_tracks_contained",
    "contained_sps_ratio"
]

cols_run3 = ["_closestNuCosmicDist"]

cols_flatten = col_trk + col_shr + col_pfp
cols_reco = col_event + cols_flatten

# Columsn for truth info per event
table_cols = [
    "mc_pdg",
    "mc_E",
    "ccnc",
    "theta",
    "true_nu_vtx_x",
    "true_nu_vtx_y",
    "true_nu_vtx_z",
    "n_pfps",
    "interaction",
    "weightSpline",
    "leeweight",
    "category",
    "nu_e",
    "nu_pdg",
    'nelec',
    'npion',
    'npi0',
    'nproton',
    'nmuon'
]
