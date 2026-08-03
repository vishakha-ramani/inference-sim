package cmd

import "testing"

func TestEffectiveEDPPRule_JointCausalVarForcesVar(t *testing.T) {
	if got := effectiveEDPPRule("dpp", true); got != "var" {
		t.Fatalf("effective rule = %q, want var", got)
	}
	if got := effectiveEDPPRule("least-ttft", false); got != "least-ttft" {
		t.Fatalf("legacy effective rule = %q, want unchanged", got)
	}
}
