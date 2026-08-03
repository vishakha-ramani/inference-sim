package trace

import (
	"bytes"
	"encoding/csv"
	"testing"
)

func TestWriteEDPPJointCandidateCSV(t *testing.T) {
	records := []EDPPJointCandidateRecord{{
		RequestID: "r1", Clock: 10, Class: "batch", DecodePod: "d0",
		Local: true, Chosen: true, RouterDecode: true,
		VarDecode: 0.2, VarCollocPrefill: 0.1, VarTotal: 0.3,
		BestVar: 0.1, ChosenVarRegret: 0.2,
		SLOExternality: 0.3, OwnGood: 0.4, NetGoodCost: -0.1,
		CapacityQueueDecode: 2_000_000, CapacityDemandDecode: 50_000,
		CapacityDecode: 0.05, CapacityPrefill: 0.02, CapacityTotal: 0.07,
		Score: -0.13, BestScore: -0.13, ChosenScoreRegret: 0,
	}}
	var buf bytes.Buffer
	if err := WriteEDPPJointCandidateCSV(&buf, records); err != nil {
		t.Fatal(err)
	}
	rows, err := csv.NewReader(&buf).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || len(rows[0]) != len(edppJointCandidateCSVHeader) {
		t.Fatalf("CSV dimensions = %dx%d", len(rows), len(rows[0]))
	}
	if rows[1][0] != "r1" || rows[1][5] != "true" || rows[1][13] != "0.2" {
		t.Fatalf("unexpected row: %v", rows[1])
	}
	columns := make(map[string]string, len(rows[0]))
	for index, header := range rows[0] {
		columns[header] = rows[1][index]
	}
	if columns["slo_externality"] != "0.3" || columns["capacity_queue_decode"] != "2e+06" ||
		columns["capacity_demand_decode"] != "50000" || columns["score"] != "-0.13" {
		t.Fatalf("new policy columns missing or misplaced: %v", columns)
	}
}
