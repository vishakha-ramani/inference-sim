package trace

import (
	"encoding/csv"
	"io"
	"strconv"
)

// edppCSVHeader lists the per-decision CSV columns in write order.
var edppCSVHeader = []string{
	"request_id", "clock", "class", "skip_reason",
	"ap", "wp", "ap_prefill", "wp_prefill", "delta_pf_chunk",
	"qd_raw", "qp_raw", "qd", "qp",
	"mu_d_nom", "mu_p_nom", "w_star_d", "w_star_p",
	"tau_ttft", "tau_itl",
	"ttft_p", "ttft_d",
	"t_adm_p", "t_adm_d", "remote_lead", "local_service", "disagg_first",
	"itl_p", "itl_d",
	"z_ttft", "z_itl",
	"balance_term_d", "balance_term_p",
	"transfer_term", "ttft_term", "itl_term", "prefill_stability_term",
	"var_local_decode", "var_local_colloc_prefill", "var_local_total",
	"var_disagg_decode", "var_disagg_colloc_prefill", "var_disagg_prefill_pool", "var_disagg_total",
	"self_good_local", "self_good_disagg",
	"kairos_mode", "kairos_alpha", "kairos_alpha_threshold",
	"kairos_ttft_gate_required", "kairos_ttft_gate_passed",
	"kairos_resident_tau_itl", "kairos_tbt_budget",
	"kairos_first_chunk", "kairos_min_chunk", "kairos_chunk_steps",
	"lhs", "rhs", "disaggregate",
}

// WriteEDPPDecisionCSV writes the EDPP per-decision rule-term records to w as CSV
// (header + one row per record). Floats use the shortest round-trippable form.
// Used by the --edpp-decision-trace output path; analysis tools consume the result.
func WriteEDPPDecisionCSV(w io.Writer, records []EDPPDecisionRecord) error {
	cw := csv.NewWriter(w)
	if err := cw.Write(edppCSVHeader); err != nil {
		return err
	}
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, r := range records {
		row := []string{
			r.RequestID, strconv.FormatInt(r.Clock, 10), r.Class, r.SkipReason,
			strconv.Itoa(r.Ap), f(r.Wp), strconv.Itoa(r.ApPrefill), f(r.WpPrefill), f(r.DeltaPfChunk),
			f(r.QdRaw), f(r.QpRaw), f(r.Qd), f(r.Qp),
			f(r.MuDNom), f(r.MuPNom), f(r.WStarD), f(r.WStarP),
			f(r.TauTTFT), f(r.TauITL),
			f(r.TTFTP), f(r.TTFTD),
			f(r.TAdmP), f(r.TAdmD), f(r.RemoteLead), f(r.LocalService), f(r.DisaggFirst),
			f(r.ITLP), f(r.ITLD),
			f(r.ZTTFT), f(r.ZITL),
			f(r.BalanceTermD), f(r.BalanceTermP),
			f(r.TransferTerm), f(r.TTFTTerm), f(r.ITLTerm),
			f(r.PrefillStabilityTerm),
			f(r.VarLocalDecode), f(r.VarLocalCollocPrefill), f(r.VarLocalTotal),
			f(r.VarDisaggDecode), f(r.VarDisaggCollocPrefill), f(r.VarDisaggPrefillPool), f(r.VarDisaggTotal),
			f(r.SelfGoodLocal), f(r.SelfGoodDisagg),
			r.KairosMode, f(r.KairosAlpha), f(r.KairosAlphaThreshold),
			strconv.FormatBool(r.KairosTTFTGateRequired), strconv.FormatBool(r.KairosTTFTGatePassed),
			f(r.KairosResidentTauITL), f(r.KairosTBTBudget),
			f(r.KairosFirstChunk), f(r.KairosMinChunk), strconv.Itoa(r.KairosChunkSteps),
			f(r.LHS), f(r.RHS), strconv.FormatBool(r.Disaggregate),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
