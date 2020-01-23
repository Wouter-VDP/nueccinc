import numpy as np
import pandas as pd
import uproot
import scipy.stats
from helpers import plot_dicts_nue
from helpers import plot_dicts_numu
from helpers import helpfunction
import matplotlib.patches as patches
import matplotlib.pyplot as plt

pi = 3.14


class Plotter:
    """
    Class to make self.data/MC plots
    Initialise using a data dictionary as produced by RootLoader.py
    """

    # Fields shared of any class instance
    gr = 1.618

    # Fields for initialisation
    def __init__(self, data, signal="nue", genie_version="mcc9.1"):
        self.signal = signal
        if signal == "nue":
            self.dicts = plot_dicts_nue
            self.cats = [1, 10, 11]
        elif signal == "numu":
            self.dicts = plot_dicts_numu
            self.cats = [30,31,32]
            
        else:
            print("Error, unknown signal string, choose nue or numu!")
        if not all([k in data.keys() for k in ["nu", "nue", "on", "off", "dirt"]]):
            print("Error, missing samples in the data set!")
            
        weights_field = "weightSplineTimesTune"
        # Use the Genie v2 model by upweighting.
        if genie_version == "mcc8":
            numu_spline = uproot.open("./helpers/numu_v13_v12_Ratio.root")
            nue_spline = uproot.open("./helpers/nue_v13_v12_Ratio.root")
            spline_x = np.append(numu_spline["Graph"].__dict__["_fX"], 100)
            spline_y_numu = np.append(numu_spline["Graph"].__dict__["_fY"], 1)
            spline_y_nue = np.append(nue_spline["Graph"].__dict__["_fY"], 1)

            data["nue"]["daughters"]["weightSpline"] = spline_y_nue[
                np.digitize(data["nue"]["daughters"]["nu_e"], spline_x)
            ] ** (-1)
            data["nu"]["daughters"]["weightSpline"] = spline_y_numu[
                np.digitize(data["nu"]["daughters"]["nu_e"], spline_x)
            ] ** (-1)
            print(
                "Using the energy/pdg dependent spline weights as in MCC8 Genie V2 tune1"
            )
            weights_field = "weightSpline"
        if genie_version == "mcc9.0":
            weights_field = "weightSpline" 
            print("Using the spline weights as in MCC9.0 Genie V3")
        if genie_version == "mcc9.1":
            print("Using the spline weights as in MCC9.1 Genie V3 tune 1")

        # Use the nue CC inc sample for plotting
        nue_tpc = helpfunction.is_tpc(
            data["nue"]["daughters"]["true_nu_vtx_x"],
            data["nue"]["daughters"]["true_nu_vtx_y"],
            data["nue"]["daughters"]["true_nu_vtx_z"],
        )
        nue_nuecctpc = (
            data["nue"]["daughters"].eval("ccnc==0 & abs(nu_pdg)==12") & nue_tpc
        )
        if len(data["nue"]["daughters"]) != sum(nue_nuecctpc):
            print(
                "Error: I was assuming the nue sample only contained cc nue events inside the TPC."
            )

        nu_tpc = helpfunction.is_tpc(
            data["nu"]["daughters"]["true_nu_vtx_x"],
            data["nu"]["daughters"]["true_nu_vtx_y"],
            data["nu"]["daughters"]["true_nu_vtx_z"],
        )
        nu_nuecctpc = data["nu"]["daughters"].eval("ccnc==0 & abs(nu_pdg)==12") & nu_tpc

        data["nu"]["daughters"]["plot_weight"] = (
            data["nu"]["daughters"][weights_field]
            * data["nu"]["scaling"]
            * (nu_nuecctpc == 0)
        )
        data["nue"]["daughters"]["plot_weight"] = (
            data["nue"]["daughters"][weights_field]
            * data["nue"]["scaling"]
            * (nue_nuecctpc == 1)
        )
        data["dirt"]["daughters"]["category"] = 7
        data["dirt"]["daughters"]["cat_int"] = 7
        data["dirt"]["daughters"]["plot_weight"] = (
            data["dirt"]["daughters"][weights_field] * data["dirt"]["scaling"]
        )
        data["on"]["daughters"]["plot_weight"] = 1
        data["off"]["daughters"]["plot_weight"] = data["off"]["scaling"]
        self.mc_daughters = pd.concat(
            [
                data["nu"]["daughters"],
                data["nue"]["daughters"],
                data["dirt"]["daughters"],
            ],
            copy=False,
            sort=False,
        )
        self.on_daughters = data["on"]["daughters"]
        self.off_daughters = data["off"]["daughters"]

        print("Initialisation completed!")

    # Get the purity of a selection
    def get_purity(self, selector, cats):
        weighted_selector = "(" + selector + ")*plot_weight"
        purity_denom = sum(self.mc_daughters.eval(weighted_selector)) + sum(
            self.off_daughters.eval(weighted_selector)
        )

        purity_nom = 0
        for cat in cats:
            purity_nom += sum(
                self.mc_daughters.query("category==@cat").eval(weighted_selector)
            )

        return purity_nom / purity_denom
    
    def get_ratio_and_purity(
        self,
        query=""
    ):
        mc_weights = sum(self.mc_daughters.query(query)["plot_weight"])
        off_weights = sum(self.off_daughters.query(query)["plot_weight"])
        on_weights = sum(self.on_daughters.query(query)["plot_weight"])

        ratio1 = (on_weights - off_weights) / mc_weights
        ratio1_err = np.sqrt(mc_weights + off_weights) / mc_weights
        ratio2 = on_weights / (mc_weights + off_weights)
        ratio = [ratio1, ratio2, ratio1_err]
        
        purity = self.get_purity(query, self.cats)
        return ratio, purity, 
    
    def plot_panel_data_mc(
        self,
        ax,
        field,
        x_label,
        N_bins,
        x_min,
        x_max,
        query="",
        title_str="",
        legend=True,
        y_max_scaler=1.1,
        kind="cat",
        show_data = True
    ):

        """
        Plot a data/MC panel with the ratio.

        Arguments:
            ax {tuple} -- matplotlib axes, length should be 2
            field {string} -- string that can be given as an .eval() pandas command
            x_label {string} -- x-axis label
            N_bins -- number or bins
            x_min {float} -- the minimum number along x
            x_max {float} -- the maximum number along x
            query {string} -- string that can be given as a .query() pandas command
            title_str {string} -- right title string of the upper plot
            legend {bool} -- plot the legend on the right of the panel
            y_max_scaler {float} -- scale the y-axis of the plot
            kind {string} -- cat / pdg / int 

        Outputs:
            ratio -- [(on-off)/MC, on/(MC+off), (on-off)/MC Error]
            purity -- purity of the inclusive channel depending on the signal
            ks_test_p -- p-value of the KS-test
            dict -- Output of the plot {labels: string, bins: 1d array}
        """
        ratio, purity = self.get_ratio_and_purity(query)
        
        plot_data = []
        weights = []
        labels = []
        colors = []
        bin_err = []

        if kind == "cat":
            kind_labs = self.dicts.category_labels
            kind_colors = self.dicts.category_colors
            column_check = "category"

        elif kind == "pdg":
            kind_labs = self.dicts.pdg_labels
            kind_colors = self.dicts.pdg_colors
            column_check = "backtracked_pdg"

        elif kind == "int":
            kind_labs = self.dicts.int_labels
            kind_colors = self.dicts.int_colors
            column_check = "cat_int"
        else:
            print("Unknown plotting type, please choose from int/pdg/cat")

        # MC contribution
        temp_view = self.mc_daughters.query(query)
        mc_data = temp_view.eval(field)
        mc_weights = temp_view["plot_weight"]

        for cat in kind_labs.keys():
            temp_view_cat = temp_view.query("abs({})==@cat".format(column_check))
            if len(temp_view_cat.index) > 0 and cat != 6:
                plot_data.append(temp_view_cat.eval(field))
                weights.append(temp_view_cat["plot_weight"])
                
                num_events = sum(weights[-1])
                precision = int(max(np.floor(np.log10(num_events))+1,2))
                labels.append(kind_labs[cat] + ": {:#.{prec}g}".format(num_events, prec=precision))
                colors.append(kind_colors[cat])
                print("MC category:", labels[-1], "\t#entries", len(plot_data[-1]))

        if self.signal == "nue":
            # LEE contribution
            plot_data.append(temp_view.query("leeweight>0.001").eval(field))
            weights.append(
                temp_view.query("leeweight>0.001").eval("leeweight*plot_weight")
            )
            labels.append(r"$\nu_e$ LEE" + ": {0:#.2g}".format(sum(weights[-1])))
            colors.append(self.dicts.category_colors[111])

        # Off Contribution
        temp_view = self.off_daughters.query(query)
        plot_data.append(temp_view.eval(field))
        weights.append(temp_view["plot_weight"])
        num_events = sum(weights[-1])
        precision = int(max(np.floor(np.log10(num_events))+1,2))
        labels.append("BNB Off" + ": {:#.{prec}g}".format(num_events, prec=precision))
        colors.append("grey")
        # On Contribution
        temp_view = self.on_daughters.query(query)
        plot_data.append(temp_view.eval(field))
        weights.append(temp_view["plot_weight"])
        labels.append("BNB On" + ": {0:0.0f}".format(sum(weights[-1])))
        colors.append('k')

        # KS-test
        flattened_MC = np.concatenate(plot_data[:-1]).ravel()
        flattened_weights = np.concatenate(weights[:-1]).ravel()
        ks_test_d, ks_test_p = kstest_weighted(
            flattened_MC, plot_data[-1], flattened_weights, weights[-1]
        )

        # Start binning   hist_bin_uncertainty(data, weights, x_min, x_max, bin_edges)
        edges, edges_mid, bins, max_val = histHelper(
            N_bins, x_min, x_max, plot_data, weights=weights
        )
        
        for data_i, weight_i in zip(plot_data[:-2], weights[:-2]):
            bin_err.append(hist_bin_uncertainty(data_i, weight_i, x_min, x_max, edges))
        err_on = hist_bin_uncertainty(plot_data[-1], weights[-1], x_min, x_max, edges)
        err_off = hist_bin_uncertainty(plot_data[-2], weights[-2], x_min, x_max, edges)
        err_mc = hist_bin_uncertainty(mc_data, mc_weights, x_min, x_max, edges)
        err_comined = np.sqrt(err_off ** 2 + err_mc ** 2)
        widths = edges_mid - edges[:-1]
        
        bin_err.extend([err_off, err_on])
        bin_dict = dict(zip(labels, zip(bins,colors,bin_err)))
        
        if show_data:
            # On
            ax[0].errorbar(
                edges_mid,
                bins[-1],
                xerr=widths,
                yerr=err_on,
                color=colors[-1],
                fmt=".",
                label=labels[-1],
            )
        # Off
        ax[0].bar(
            edges_mid, bins[-2], lw=2, label=labels[-2], width=2 * widths, color=colors[-2]
        )
        bottom = np.copy(bins[-2])
        for bin_i, lab_i, col_i in zip(bins[:-2], labels[:-2], colors[:-2]):
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
        ax[0].set_title(
            "(On-Off)/MC:{0:.2f}$\pm${1:.2f}".format(ratio[0], ratio[2]), loc="left"
        )
        ax[0].set_ylim(0, y_max_scaler * max(max_val[-1], max(val)))
        ax[0].set_xlim(x_min, x_max)

        # Ratio plots
        y_min_r = max(0, min((bins[-1]-err_on)/val)*0.9)  
        y_max_r = min(2, max((bins[-1]+err_on)/val)*1.1)  
        ax[1].set_ylim(y_min_r, y_max_r)
        ax[1].set_xlim(x_min, x_max)
        if show_data:               
            ax[1].errorbar(
                edges_mid,
                bins[-1] / val,
                xerr=widths,
                yerr=err_on / val,
                alpha=1.0,
                color="k",
                fmt=".",
            )
        ax[1].set_ylabel(r"$\frac{Beam\ ON}{Beam\ OFF + MC}$")
        ax[1].set_xlabel(x_label)

        if legend:
            ax[0].legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

        return ratio, purity, ks_test_p, (bin_dict, edges_mid)
    
    
    def plot_variable(
        self,
        ax,
        field,
        x_label,
        N_bins,
        x_min,
        x_max,
        query="",
        title_str="",
        kind="cat",
        y_max_scaler=1.1,
    ):
        ratio, purity, ks_p, (bin_dict, edges_mid) = self.plot_panel_data_mc(
            ax.T[1],
            field,
            x_label,
            N_bins,
            x_min,
            x_max,
            query,
            title_str,
            kind,
            y_max_scaler,
        )
        ax[0][1].text(
            ax[0][1].get_xlim()[1] * 0.8,
            ax[0][1].get_ylim()[1] * 0.8,
            r"$\nu_e$"
            + " CC purity: {0:<3.1f}%\nKS p-value: {1:<5.2f}".format(purity * 100, ks_p),
            horizontalalignment="right",
            fontsize=12,
        )
        ax[1][0].remove()

        exclude = list(bin_dict.keys())[-4:]
        for k, (v, c, e) in bin_dict.items():
            if k not in exclude:
                sum_cat = float(k.split(":")[-1])
                v_norm = v / sum_cat
                e_norm = e / sum_cat
                print(k)
                ax[0][0].fill_between(
                    edges_mid,
                    v_norm - e_norm,
                    v_norm + e_norm,
                    alpha=0.3,
                    step="mid",
                    color=c,
                )
                ax[0][0].set_xlabel(x_label)
                ax[0][0].set_ylabel("Area Normalised")
                ax[0][0].set_title("Background/Signal shape", loc="left")


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


# Helper class duplicating the last bin, useful to use in combination with matplotlib step function.
def efficiency_post(
    num, den, num_w=None, den_w=None, n_bins=10, x_min=0, x_max=10, conf_level=None
):
    eff, unc_low, unc_up, edges = efficiency(
        num, den, num_w, den_w, n_bins, x_min, x_max, conf_level
    )
    eff = np.append(eff, eff[-1])
    unc_low = np.append(unc_low, unc_low[-1])
    unc_up = np.append(unc_up, unc_up[-1])
    return eff, unc_low, unc_up, edges


def hist_bin_uncertainty(data, weights, x_min, x_max, bin_edges):
    """
        Calculate the error on the bins in the histogram including the weights.

        Arguments:
            edges {1d tuple} -- The bin edges
            data {1d tuple} -- array with the data of a variable
            weigths {1d tuple} -- weights, same shape as data
            
        Outputs:
            bin_uncertainties {1d tuple} -- Uncertainty on each bin
        """
    # Bound the data and weights to be within the bin edges
    mask_in_range = (data > x_min) & (data < x_max)
    in_range_data = data[mask_in_range]
    in_range_weights = weights[mask_in_range]

    # Bin the weights with the same binning as the data
    bin_index = np.digitize(in_range_data, bin_edges)
    # N.B.: range(1, bin_edges.size) is used instead of set(bin_index) as if
    # there is a gap in the data such that a bin is skipped no index would appear
    # for it in the set
    binned_weights = np.asarray(
        [
            in_range_weights[np.where(bin_index == idx)[0]]
            for idx in range(1, len(bin_edges))
        ]
    )
    bin_uncertainties = np.asarray(
        [np.sqrt(np.sum(np.square(w))) for w in binned_weights]
    )
    return bin_uncertainties


def kstest_weighted(data1, data2, wei1, wei2):
    """
    2-sample KS test unbinned probability.
    Takes into account the weight of the events.
    stackoverflow.com/questions/40044375/
    how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples/40059727
    
    Arguments:
        data1 {1d tuple} -- array with the data of a variable
        wei1 {1d tuple} -- weights, same shape as data
        data2 {1d tuple} -- array with the data of a variable
        wei2 {1d tuple} -- weights, same shape as data

    Outputs:
        d -- KS-text max separation
        prob -- KS-test p-value
    """
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])
    cdf1we = cwei1[[np.searchsorted(data1, data, side="right")]]
    cdf2we = cwei2[[np.searchsorted(data2, data, side="right")]]
    d = np.max(np.abs(cdf1we - cdf2we))
    # Note: d absolute not signed distance
    n1 = sum(wei1)
    n2 = sum(wei2)
    en = np.sqrt(n1 * n2 / float(n1 + n2))
    prob = scipy.stats.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    return d, prob


def histHelper(n_bins, x_min, x_max, data, weights=0, where="mid", log=False):
    """
    Wrapper around the numpy histogram function.

    Arguments:
        n_bins {int} -- the number of bins
        x_min {float} -- the minimum number along x
        x_max {float} -- the maximum number along x
        data {2d tuple} -- array or list of arrays
        weigths {2d tuple} -- same shape as data or 0 if all weights are equal
        where {string} -- if where='post': duplicate the last bin
        log {bool} -- log==True: return x-axis log

    Outputs:
        edges {1d tuple} -- The bin edges
        edges_mid {1d tuple} -- The middle of the bins
        bins {2d tuple} -- The bin values, same depth as data
        max_val {1d tuple}-- the maximum value, same depth as data
    """
    if log:
        edges = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
    else:
        edges = np.linspace(x_min, x_max, n_bins + 1)
    edges_mid = [edges[i] + (edges[i + 1] - edges[i]) / 2 for i in range(n_bins)]
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
