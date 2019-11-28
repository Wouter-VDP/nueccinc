import numpy as np
import pandas as pd
import scipy.stats
from helpers import plot_dicts
import matplotlib.patches as patches
import matplotlib.pyplot as plt

gr = 1.618

def efficiency(
    num, den, num_w=None, den_w=None, n_bins=10, x_min=0, x_max=10, conf_level=None
):
    """
    Calculate the efficiency given two populations: one containg 
    the totatility of the events,and one containing only events 
    that pass the selection.
    It uses a frequentist approach to evaluate the uncertainty.
    Other methods are to be implemented.
    
     Arguments:
        num {tuple} -- The totality of the events
        den {tuple} -- The events that pass the selection
        num_w {tuple} -- Optional, the weight for every event
        den_w {tuple} -- Optional, the weight for every selected event
        n_bins {int} -- Optional, the number of bins
        x_min {float} -- Optional, the minimum number along x
        x_max {float} -- Optional, the maximum number along x
        conf_level {float} -- Optional, the confidence level to be used
        
    Outputs:
        eff {tuple} -- The efficiency per bin
        unc_low {tuple} -- The lower uncertainty per bin
        unc_up {tuple} -- The upper uncertainty per bi
        bins {tuple} -- The bin edges
    """

    if num_w is None:
        num_w = [1.0] * len(num)

    if den_w is None:
        den_w = [1.0] * len(den)

    if conf_level is None:
        conf_level = 0.682689492137

    num = np.asarray(num, dtype=np.float32)
    num_w = np.asarray(num_w, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    den_w = np.asarray(den_w, dtype=np.float32)

    bins = np.linspace(x_min, x_max, n_bins)

    num_h, _ = np.histogram(num, bins=bins)
    num_w_h, _ = np.histogram(num, weights=num_w, bins=bins)
    num_w2_h, _ = np.histogram(num, weights=num_w ** 2, bins=bins)

    den_h, _ = np.histogram(den, bins=bins)
    den_w_h, _ = np.histogram(den, weights=den_w, bins=bins)
    den_w2_h, _ = np.histogram(den, weights=den_w ** 2, bins=bins)

    eff = num_w_h / den_w_h

    variance = (num_w2_h * (1.0 - 2 * eff) + den_w2_h * eff * eff) / (den_w_h * den_w_h)
    sigma = np.sqrt(variance)
    prob = 0.5 * (1.0 - conf_level)
    delta = -scipy.stats.norm.ppf(prob) * sigma

    unc_up = []
    unc_low = []

    for eff_i, delta_i in zip(eff, delta):
        if eff_i - delta_i < 0:
            unc_low.append(eff_i)
        else:
            unc_low.append(delta_i)

        if eff_i + delta_i > 1:
            unc_up.append(1.0 - eff_i)
        else:
            unc_up.append(delta_i)

    return eff, unc_low, unc_up, bins

def efficiency_post(num, den, num_w=None, den_w=None, n_bins=10, x_min=0, x_max=10, conf_level=None):
    eff, unc_low, unc_up, edges = efficiency(num, den, num_w, den_w, n_bins, x_min, x_max, conf_level)
    eff = np.append(eff, eff[-1])
    unc_low = np.append(unc_low, unc_low[-1])
    unc_up = np.append(unc_up, unc_up[-1])
    return eff, unc_low, unc_up, edges

# get the purity of a selection, very data format specific!
def get_purity(data, selector, cats):
    ## Calculate the purity:
    purity_denom = sum(data["nu"]["daughters"].eval(selector)) * data["nu"]["scaling"]
    purity_denom += (
        sum(data["dirt"]["daughters"].eval(selector + "*weightSpline"))
        * data["dirt"]["scaling"]
    )
    purity_denom += (
        sum(data["off"]["daughters"].eval(selector)) * data["off"]["scaling"]
    )
    purity_nom = 0
    for cat in cats:
        purity_nom += sum(
            data["nu"]["daughters"].query("category==@cat").eval(selector + "*weightSpline")
        )* data["nu"]["scaling"]
    return purity_nom / purity_denom


def plot_panel_data_mc(
    data,
    ax,
    field,
    x_label,
    N_bins,
    x_min,
    x_max,
    query="",
    title_str = "",
    legend=True,
    y_max_scaler=1.2,
    kind="event_category",
):
    plot_data = []
    weights = []
    labels = []
    colors = []

    if kind == "event_category":
        kind_labs = plot_dicts.category_labels
        kind_colors = plot_dicts.category_colors
        column_check = "category"

    elif kind == "event_pdg":
        kind_labs = plot_dicts.pdg_labels
        kind_colors = plot_dicts.pdg_colors
        column_check = "backtracked_pdg"

    # MC contribution
    for cat in kind_labs.keys():
        cat_data = (
            data["nu"]["daughters"]
            .query(query)
            .query("abs({})==@cat".format(column_check))
            .eval(field)
        )
        if len(cat_data) > 0 and cat != 6:
            plot_data.append(cat_data)
            weights.append(
                data["nu"]["daughters"]
                .query(query)
                .query("abs({})==@cat".format(column_check))["weightSpline"]
                * data["nu"]["scaling"]
            )
            labels.append(kind_labs[cat] + ": {0:#.1f}".format(sum(weights[-1])))
            colors.append(kind_colors[cat])
            print(
                "MC category:",
                labels[-1],
                "\t#entries",
                len(plot_data[-1])
            )

    # LEE contribution
    plot_data.append(data["nue"]["daughters"].query(query).eval(field))
    weights.append(
        data["nue"]["daughters"].query(query)["leeweight"] * data["nue"]["scaling"]
    )
    labels.append(r"$\nu_e$ LEE" + ": {0:#.2g}".format(sum(weights[-1])))

    # DRT contribution
    plot_data.append(data["dirt"]["daughters"].query(query).eval(field))
    weights.append(
        data["dirt"]["daughters"].query(query)["weightSpline"] * data["dirt"]["scaling"]
    )
    labels.append("Out of Cryo" + ": {0:#.1f}".format(sum(weights[-1])))
    # Off Contribution
    plot_data.append(data["off"]["daughters"].query(query).eval(field))
    weights.append(len(plot_data[-1]) * [data["off"]["scaling"]])
    labels.append("BNB Off" + ": {0:#.1f}".format(sum(weights[-1])))
    # On Contribution
    plot_data.append(data["on"]["daughters"].query(query).eval(field))
    weights.append([1.0] * len(plot_data[-1]))
    labels.append("BNB On" + ": {0:0.0f}".format(sum(weights[-1])))

    mc_weights =  data["nu"]["daughters"].query(query)["weightSpline"]*data["nu"]["scaling"]
    ratio = sum(weights[-1]) / (sum(weights[-2])+sum(weights[-3])+sum(mc_weights))
    
    flattened_MC = np.concatenate(plot_data[:-1]).ravel()
    flattened_weights = np.concatenate(weights[:-1]).ravel()
    ks_test_d, ks_test_p = ks_w2(flattened_MC, plot_data[-1], flattened_weights, np.array(weights[-1]))
    
    #Start binning
    edges, edges_mid, bins, max_val = histHelper(
        N_bins, x_min, x_max, plot_data, weights=weights
    )
    err_on = hist_bin_uncertainty(list(plot_data[-1]), list(weights[-1]), edges)
    err_off = hist_bin_uncertainty(list(plot_data[-2]), list(weights[-2]), edges)
    err_drt = hist_bin_uncertainty(list(plot_data[-3]), list(weights[-3]), edges)
    err_mc = hist_bin_uncertainty(
        list(data["nu"]["daughters"].query(query).eval(field)),
        list(
            data["nu"]["daughters"].query(query)["weightSpline"] * data["nu"]["scaling"]
        ),
        edges,
    )
    err_comined = np.sqrt(err_off ** 2 + err_drt ** 2 + err_mc ** 2)

    widths = edges_mid - edges[:-1]

    # On
    ax[0].errorbar(
        edges_mid,
        bins[-1],
        xerr=widths,
        yerr=err_on,
        color="k",
        fmt=".",
        label=labels[-1],
    )
    # Off
    ax[0].bar(
        edges_mid, bins[-2], lw=2, label=labels[-2], width=2 * widths, color="grey"
    )
    bottom = bins[-2]
    for bin_i, lab_i, col_i in zip(bins[:-2], labels[:-2], colors):
        ax[0].bar(
            edges_mid,
            bin_i,
            lw=2,
            label=lab_i,
            width=2 * widths,
            bottom=bottom,
            color=col_i,
        )
        bottom += bin_i
    # DRT
    print("DRT: {0:#.2g}".format(sum(bins[-3])))
    if sum(bins[-3]) > 0.2:
        ax[0].bar(
            edges_mid, bins[-3], lw=2, label=labels[-3], width=2 * widths, bottom=bottom, color='C1'
        )
        bottom += bins[-3]
    # LEE
    ax[0].bar(
        edges_mid,
        bins[-4],
        lw=2,
        label=labels[-4],
        width=2 * widths,
        bottom=bottom,
        color=plot_dicts.category_colors[111],
    )
    bottom += bins[-4]
    val = bottom
    for m, v, e, w in zip(edges_mid, val, err_comined, widths):
        ax[0].add_patch(
            patches.Rectangle(
                (m - w, v - e),
                2 * w,
                2 * e,
                hatch="\\\\\\\\\\",
                Fill=False,
                linewidth=0,
                alpha=0.4,
            )
        )
        sc_err = e / v
        ax[1].add_patch(
            patches.Rectangle(
                (m - w, 1 - sc_err),
                2 * w,
                sc_err * 2,
                hatch="\\\\\\\\\\",
                Fill=False,
                linewidth=0,
                alpha=0.4,
            )
        )
   
    ax[0].set_ylabel("Events per bin")
    ax[0].set_title(title_str, loc="right")
    ax[0].set_title("Data/MC ratio: {0:#.2f}".format(ratio), loc="left")
    ax[0].set_ylim(0, y_max_scaler * max_val[-1])
    ax[0].set_xlim(x_min, x_max)

    # Ratio plots
    ax[1].set_ylim(0.0, 2)
    ax[1].set_xlim(x_min, x_max)
    ax[1].errorbar(
        edges_mid,
        bins[-1] / val,
        xerr=widths,
        yerr=err_on / val,
        alpha=1.0,
        color="k",
        fmt=".",
        label="Data error",
    )
    ax[1].set_ylabel(r"$\frac{Beam\ ON}{Beam\ OFF + MC}$")
    ax[1].set_xlabel(x_label)

    if legend:
        ax[0].legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

    purity = get_purity(data, query, [1, 10, 11])
    return ratio, purity, ks_test_p


# @in N: number of bins 
# @in x_min, x_max: range of the plot
# @in data: list of data arrays
# @in weights: list of weights
# @in if where='post': duplicate the last bin
# @in log=True: return x-axis log
def histHelper(N, x_min, x_max, data, weights=0, where="mid", log=False):
    if log:
        edges = np.logspace(np.log10(x_min), np.log10(x_max), N + 1)
    else:
        edges = np.linspace(x_min, x_max, N + 1)
    edges_mid = [edges[i] + (edges[i + 1] - edges[i]) / 2 for i in range(N)]
    if weights == 0:
        weights = [[1] * len(d) for d in data]

    bins = [
        np.histogram(data_i, bins=edges, weights=weights_i)[0]
        for data_i, weights_i in zip(data, weights)
    ]
    max_val = [max(x) for x in bins]
    if where == "post":
        bins = [np.append(b, b[-1]) for b in bins]
    return edges, edges_mid, bins, max_val

# weighted histogram error  
def hist_bin_uncertainty(data, weights, bin_edges):
    # Bound the data and weights to be within the bin edges
    in_range_index = [idx for idx in range(len(data)) if data[idx] > min(bin_edges) and data[idx] < max(bin_edges)]
    in_range_data = np.asarray([data[idx] for idx in in_range_index])
    in_range_weights = np.asarray([weights[idx] for idx in in_range_index])

    # Bin the weights with the same binning as the data
    bin_index = np.digitize(in_range_data, bin_edges)
    # N.B.: range(1, bin_edges.size) is used instead of set(bin_index) as if
    # there is a gap in the data such that a bin is skipped no index would appear
    # for it in the set
    binned_weights = np.asarray(
        [in_range_weights[np.where(bin_index == idx)[0]] for idx in range(1, len(bin_edges))])
    bin_uncertainties = np.asarray(
        [np.sqrt(np.sum(np.square(w))) for w in binned_weights])
    return bin_uncertainties

# https://stackoverflow.com/questions/40044375/how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples/40059727
def ks_w2(data1, data2, wei1, wei2):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[[np.searchsorted(data1, data, side='right')]]
    cdf2we = cwei2[[np.searchsorted(data2, data, side='right')]]
    d = np.max(np.abs(cdf1we - cdf2we))
    # Note: d absolute not signed distance
    n1 = sum(wei1)
    n2 = sum(wei2)
    en = np.sqrt(n1*n2/float(n1+n2))
    #try:
    prob = scipy.stats.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    #except:
    #    print("ks_failure")
    #    prob = 1.0
    return d, prob
