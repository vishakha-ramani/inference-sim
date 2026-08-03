package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestEDPPRule_Var_AcceptedAndBadMetricRejected is the durable behavioral guard on the
// --edpp-rule var / --edpp-var-metric CLI wiring (cmd/root.go, sim/cluster). It verifies:
//
//  1. --edpp-rule var --edpp-var-metric flip threads through to a completed run under the
//     reduced EDPP path AND emits the loud DIAGNOSTIC-ORACLE / UPPER-BOUND warning (§4);
//  2. an unknown --edpp-var-metric is rejected (non-zero exit) at construction (R3 guard);
//  3. --edpp-rule var-prefill reaches a completed run through the real CLI/config wiring.
//
// Both scenarios re-exec `blis run` as a child process (BLIS_EDPPVAR_CHILD=1) because a
// fatal/panic calls os.Exit and would otherwise kill the whole test binary.
func TestEDPPRule_Var_AcceptedAndBadMetricRejected(t *testing.T) {
	if os.Getenv("BLIS_EDPPVAR_CHILD") == "1" {
		runEDPPVarChild(t)
		return
	}
	if _, err := os.Stat(frozenH100CoeffsPath); os.IsNotExist(err) {
		t.Skipf("frozen coeffs file absent (%s), skipping", frozenH100CoeffsPath)
	}

	// Case 1: --edpp-rule var --edpp-var-metric flip completes and produces metrics + warning.
	out, err := runEDPPVarChildProcess(t, "accept")
	if err != nil {
		t.Fatalf("--edpp-rule var --edpp-var-metric flip run failed (expected success): %v\noutput:\n%s", err, out)
	}
	if !edppRuleCompletedRE.Match(out) {
		t.Fatalf("var run produced no completed_requests metrics in stdout:\n%s", out)
	}
	const wantWarn = "DIAGNOSTIC ORACLE"
	if !strings.Contains(string(out), wantWarn) {
		t.Errorf("var run should emit the %q upper-bound warning (INV-9 gate), got:\n%s", wantWarn, out)
	}

	// Case 2: an unknown --edpp-var-metric must fail (non-zero exit; validate() panics, R3).
	out, err = runEDPPVarChildProcess(t, "badmetric")
	if err == nil {
		t.Fatalf("expected non-zero exit for --edpp-var-metric bogus, got exit 0; output:\n%s", out)
	}
	const wantMsg = "VarMetric"
	if !strings.Contains(string(out), wantMsg) {
		t.Errorf("badmetric scenario: failure message should mention %q, got:\n%s", wantMsg, out)
	}

	// Case 3: the simplified VaR + prefill-stability rule is accepted and
	// announces the isolated objective.
	out, err = runEDPPVarChildProcess(t, "varprefill")
	if err != nil {
		t.Fatalf("--edpp-rule var-prefill run failed (expected success): %v\noutput:\n%s", err, out)
	}
	if !edppRuleCompletedRE.Match(out) {
		t.Fatalf("var-prefill run produced no completed_requests metrics in stdout:\n%s", out)
	}
	const wantMode = "PREFILL-STABLE"
	if !strings.Contains(string(out), wantMode) {
		t.Errorf("var-prefill run should announce %q mode, got:\n%s", wantMode, out)
	}
}

func runEDPPVarChildProcess(t *testing.T, scenario string) ([]byte, error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestEDPPRule_Var_AcceptedAndBadMetricRejected")
	cmd.Env = append(os.Environ(), "BLIS_EDPPVAR_CHILD=1", "BLIS_EDPPVAR_SCENARIO="+scenario)
	return cmd.CombinedOutput()
}

// runEDPPVarChild is the child-process body: a tiny 1P2D EDPP config with --edpp-rule var
// (plus a bogus --edpp-var-metric in the "badmetric" scenario) executing the real runCmd.
func runEDPPVarChild(t *testing.T) {
	scenario := os.Getenv("BLIS_EDPPVAR_SCENARIO")
	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)

	metric := "flip"
	if scenario == "badmetric" {
		metric = "bogus"
	}
	rule := "var"
	logLevel := "warning"
	if scenario == "varprefill" {
		rule = "var-prefill"
		logLevel = "info"
	}
	args := []string{
		"--model", "test-model",
		"--latency-model", "trained-physics",
		"--hardware", "H100",
		"--tp", "1",
		"--model-config-folder", mcFolder,
		"--hardware-config", hwPath,
		"--total-kv-blocks", "2000",
		"--block-size-in-tokens", "16",
		"--num-instances", "3",
		"--prefill-instances", "1",
		"--decode-instances", "2",
		"--seed", "42",
		"--workload", "distribution",
		"--rate", "20",
		"--num-requests", "40",
		"--prompt-tokens", "64",
		"--output-tokens", "32",
		"--max-num-running-reqs", "4",
		"--horizon", "60000000",
		"--defaults-filepath", "../defaults.yaml",
		"--log", logLevel, // capture the selected mode's construction log
		"--pd-decider", "edpp",
		"--edpp-coeffs", frozenH100CoeffsPath,
		"--edpp-tau-ttft", "10s",
		"--edpp-tau-itl", "50ms",
		"--edpp-tau-e2e", "30s",
		"--edpp-rule", rule,
		"--edpp-var-metric", metric,
	}
	if scenario == "varprefill" {
		args = append(args, "--edpp-var-deployable")
	}
	if err := runCmd.ParseFlags(args); err != nil {
		fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
		os.Exit(2)
	}
	runCmd.Run(runCmd, nil)
	os.Exit(0)
}
