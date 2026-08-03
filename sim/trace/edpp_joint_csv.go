package trace

import (
	"encoding/csv"
	"io"
	"strconv"
)

// edppJointCSVHeader lists the scorer-vs-joint divergence CSV columns in write order.
var edppJointCSVHeader = []string{
	"request_id", "clock", "class",
	"scorer_d", "joint_d", "scorer_p", "joint_p",
	"agree_d", "agree_p",
	"j_scorer", "j_joint", "disaggregate",
}

// WriteEDPPJointDecisionCSV writes the scorer-vs-joint divergence records to w as CSV
// (header + one row per record). Floats use the shortest round-trippable form. Used by the
// --edpp-joint-trace output path; analysis tools consume the result to quantify how often
// and by how much the joint objective overrides the composable scorer.
func WriteEDPPJointDecisionCSV(w io.Writer, records []EDPPJointDecisionRecord) error {
	cw := csv.NewWriter(w)
	if err := cw.Write(edppJointCSVHeader); err != nil {
		return err
	}
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, r := range records {
		row := []string{
			r.RequestID, strconv.FormatInt(r.Clock, 10), r.Class,
			r.ScorerD, r.JointD, r.ScorerP, r.JointP,
			strconv.FormatBool(r.AgreeD), strconv.FormatBool(r.AgreeP),
			f(r.JScorer), f(r.JJoint), strconv.FormatBool(r.Disaggregate),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}

var edppJointCandidateCSVHeader = []string{
	"request_id", "clock", "class", "decode_instance", "prefill_instance",
	"local", "chosen", "router_decode",
	"var_decode", "var_colloc_prefill", "var_prefill_pool", "var_total",
	"best_var", "chosen_var_regret",
	"slo_externality", "own_good", "net_good_cost",
	"capacity_queue_decode", "capacity_queue_prefill",
	"capacity_demand_decode", "capacity_demand_prefill",
	"capacity_decode", "capacity_prefill", "capacity_total",
	"score", "best_score", "chosen_score_regret",
}

// WriteEDPPJointCandidateCSV writes one row for every action considered by the
// joint argmin, preserving deterministic request and candidate order.
func WriteEDPPJointCandidateCSV(w io.Writer, records []EDPPJointCandidateRecord) error {
	cw := csv.NewWriter(w)
	if err := cw.Write(edppJointCandidateCSVHeader); err != nil {
		return err
	}
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, r := range records {
		row := []string{
			r.RequestID, strconv.FormatInt(r.Clock, 10), r.Class, r.DecodePod, r.PrefillPod,
			strconv.FormatBool(r.Local), strconv.FormatBool(r.Chosen), strconv.FormatBool(r.RouterDecode),
			f(r.VarDecode), f(r.VarCollocPrefill), f(r.VarPrefillPool), f(r.VarTotal), f(r.BestVar), f(r.ChosenVarRegret),
			f(r.SLOExternality), f(r.OwnGood), f(r.NetGoodCost),
			f(r.CapacityQueueDecode), f(r.CapacityQueuePrefill),
			f(r.CapacityDemandDecode), f(r.CapacityDemandPrefill),
			f(r.CapacityDecode), f(r.CapacityPrefill), f(r.CapacityTotal),
			f(r.Score), f(r.BestScore), f(r.ChosenScoreRegret),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
