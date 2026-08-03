package trace

import (
	"bytes"
	"encoding/csv"
	"testing"
)

func TestWriteEDPPDecisionCSV(t *testing.T) {
	records := []EDPPDecisionRecord{
		{
			RequestID: "r1", Clock: 1000, Class: "standard",
			Ap: 800, Wp: 8000, ApPrefill: 200, WpPrefill: 2000, DeltaPfChunk: 8000,
			QdRaw: 38664, QpRaw: 0, Qd: 0.3945, Qp: 0,
			TAdmP: 1000, TAdmD: 2000, RemoteLead: 9000,
			LocalService: 3000, DisaggFirst: 4000,
			TransferTerm: 0.05, TTFTTerm: 0, ITLTerm: 0,
			PrefillStabilityTerm: 0.01,
			BalanceTermD:         0.0322, BalanceTermP: 0,
			VarLocalDecode: 0.3, VarLocalTotal: 0.3,
			VarDisaggDecode: 0.1, VarDisaggPrefillPool: 0.05, VarDisaggTotal: 0.15,
			SelfGoodLocal: 0.4, SelfGoodDisagg: 0.7,
			KairosMode: "paper", KairosAlpha: 1.3, KairosAlphaThreshold: 2600,
			KairosTTFTGateRequired: true, KairosTTFTGatePassed: true,
			KairosResidentTauITL: 50_000, KairosTBTBudget: 50_000,
			KairosFirstChunk: 256, KairosMinChunk: 88, KairosChunkSteps: 3,
			LHS: 0.0322, RHS: 0.05, Disaggregate: false,
		},
		{RequestID: "r2", Clock: 2000, Class: "standard", SkipReason: "empty-prompt"},
	}

	var buf bytes.Buffer
	if err := WriteEDPPDecisionCSV(&buf, records); err != nil {
		t.Fatalf("WriteEDPPDecisionCSV: %v", err)
	}

	rows, err := csv.NewReader(&buf).ReadAll()
	if err != nil {
		t.Fatalf("parse CSV: %v", err)
	}
	if len(rows) != 3 { // header + 2 data rows
		t.Fatalf("rows = %d, want 3 (header + 2)", len(rows))
	}

	header := rows[0]
	// Build a column index so the test is order-independent.
	col := map[string]int{}
	for i, h := range header {
		col[h] = i
	}
	for _, want := range []string{
		"request_id", "clock", "class", "skip_reason", "ap", "wp",
		"ap_prefill", "wp_prefill",
		"t_adm_p", "t_adm_d", "remote_lead", "local_service", "disagg_first",
		"lhs", "rhs", "transfer_term", "ttft_term", "itl_term",
		"prefill_stability_term", "var_local_decode", "var_local_total",
		"var_disagg_decode", "var_disagg_prefill_pool", "var_disagg_total",
		"self_good_local", "self_good_disagg",
		"kairos_mode", "kairos_alpha", "kairos_alpha_threshold",
		"kairos_ttft_gate_required", "kairos_ttft_gate_passed",
		"kairos_resident_tau_itl", "kairos_tbt_budget",
		"kairos_first_chunk", "kairos_min_chunk", "kairos_chunk_steps",
		"disaggregate",
	} {
		if _, ok := col[want]; !ok {
			t.Errorf("missing expected column %q in header %v", want, header)
		}
	}

	r1 := rows[1]
	if r1[col["request_id"]] != "r1" {
		t.Errorf("row1 request_id = %q, want r1", r1[col["request_id"]])
	}
	if r1[col["disaggregate"]] != "false" {
		t.Errorf("row1 disaggregate = %q, want false", r1[col["disaggregate"]])
	}
	if r1[col["lhs"]] == "" || r1[col["rhs"]] == "" {
		t.Errorf("row1 lhs/rhs should be populated, got lhs=%q rhs=%q", r1[col["lhs"]], r1[col["rhs"]])
	}
	if got := r1[col["var_disagg_total"]]; got != "0.15" {
		t.Errorf("row1 var_disagg_total = %q, want 0.15", got)
	}
	if got := r1[col["self_good_disagg"]]; got != "0.7" {
		t.Errorf("row1 self_good_disagg = %q, want 0.7", got)
	}
	if got := r1[col["ap_prefill"]]; got != "200" {
		t.Errorf("row1 ap_prefill = %q, want 200", got)
	}
	if got := r1[col["remote_lead"]]; got != "9000" {
		t.Errorf("row1 remote_lead = %q, want 9000", got)
	}
	if got := r1[col["kairos_mode"]]; got != "paper" {
		t.Errorf("row1 kairos_mode = %q, want paper", got)
	}
	if got := r1[col["kairos_chunk_steps"]]; got != "3" {
		t.Errorf("row1 kairos_chunk_steps = %q, want 3", got)
	}

	r2 := rows[2]
	if r2[col["skip_reason"]] != "empty-prompt" {
		t.Errorf("row2 skip_reason = %q, want empty-prompt", r2[col["skip_reason"]])
	}
}
