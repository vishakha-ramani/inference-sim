package cluster

import (
	"math"
	"testing"
)

// TestV2SaturationAnalyzerAnalyze verifies the V2 token-based saturation analyzer
// contract: capacity in tokens via min(k1_memory, k2_compute), demand as
// tokensInUse + queueLength*avgInputTokens, and model-level RequiredCapacity/SpareCapacity signals.
func TestV2SaturationAnalyzerAnalyze(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  0.8,  // k1 = totalKvCapTokens * 0.8
		ScaleUpThreshold:  0.8,  // RequiredCapacity when demand/supply > 1/0.8
		ScaleDownBoundary: 0.4,  // SpareCapacity when demand/supply < 0.4
		AvgInputTokens:    512,  // for queue-to-demand conversion
	}
	analyzer := NewV2SaturationAnalyzer(cfg)

	if analyzer.Name() != "v2-saturation" {
		t.Fatalf("Name() = %q, want %q", analyzer.Name(), "v2-saturation")
	}

	tests := []struct {
		name string
		input ModelSignals
		// Expected outputs — use -1 to skip check
		wantRequiredPositive bool
		wantSparePositive    bool
		wantTotalSupplyZero  bool
		wantTotalDemandZero  bool
		// Invariant checks
		checkAggregation bool // sum(vc.Supply)==TotalSupply, sum(vc.Demand)==TotalDemand
	}{
		{
			name:                "zero replicas — all-zero output",
			input:               ModelSignals{ModelID: "m1", Replicas: nil},
			wantRequiredPositive: false,
			wantSparePositive:    false,
			wantTotalSupplyZero:  true,
			wantTotalDemandZero:  true,
		},
		{
			name: "all replicas saturated — RequiredCapacity > 0",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         9000,  // 90% used
						QueueDepth:            10,     // additional demand: 10 * 512 = 5120 tokens
						CostPerHour:           10.0,
					},
					{
						InstanceID:            "i2",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         8500,
						QueueDepth:            8,
						CostPerHour:           10.0,
					},
				},
			},
			wantRequiredPositive: true,
			wantSparePositive:    false,
			checkAggregation:     true,
		},
		{
			name: "all replicas idle with headroom — SpareCapacity > 0",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         500,  // 5% used — very idle
						QueueDepth:            0,
						CostPerHour:           10.0,
					},
					{
						InstanceID:            "i2",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         600,
						QueueDepth:            0,
						CostPerHour:           10.0,
					},
					{
						InstanceID:            "i3",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         400,
						QueueDepth:            0,
						CostPerHour:           10.0,
					},
				},
			},
			wantRequiredPositive: false,
			wantSparePositive:    true,
			checkAggregation:     true,
		},
		{
			name: "single replica near saturation — SpareCapacity must be zero",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         7500,
						QueueDepth:            5,
						CostPerHour:           10.0,
					},
				},
			},
			wantRequiredPositive: false, // not necessarily saturated
			wantSparePositive:    false, // cannot scale below 1 replica
		},
		{
			name: "mixed variants — aggregation invariant holds",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         5000,
						QueueDepth:            2,
						CostPerHour:           10.0,
					},
					{
						InstanceID:            "i2",
						Variant:               NewVariantSpec("H100", 2),
						TotalKvCapacityTokens: 20000,
						KvTokensInUse:         8000,
						QueueDepth:            3,
						CostPerHour:           20.0,
					},
				},
			},
			checkAggregation: true,
		},
		{
			name: "mutual exclusivity — RequiredCapacity > 0 implies SpareCapacity == 0",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         9500,
						QueueDepth:            20,
						CostPerHour:           10.0,
					},
				},
			},
			wantRequiredPositive: true,
			wantSparePositive:    false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := analyzer.Analyze(tc.input)

			// ModelID propagation
			if result.ModelID != tc.input.ModelID {
				t.Errorf("ModelID = %q, want %q", result.ModelID, tc.input.ModelID)
			}

			// RequiredCapacity / SpareCapacity checks
			if tc.wantRequiredPositive && result.RequiredCapacity <= 0 {
				t.Errorf("RequiredCapacity = %f, want > 0", result.RequiredCapacity)
			}
			if !tc.wantRequiredPositive && result.RequiredCapacity > 0 {
				// Only check if we explicitly expect it to be zero
				if tc.wantTotalSupplyZero || tc.wantSparePositive {
					t.Errorf("RequiredCapacity = %f, want 0", result.RequiredCapacity)
				}
			}
			if tc.wantSparePositive && result.SpareCapacity <= 0 {
				t.Errorf("SpareCapacity = %f, want > 0", result.SpareCapacity)
			}
			if !tc.wantSparePositive && result.SpareCapacity > 0 {
				t.Errorf("SpareCapacity = %f, want 0", result.SpareCapacity)
			}

			// Mutual exclusivity: never both positive
			if result.RequiredCapacity > 0 && result.SpareCapacity > 0 {
				t.Errorf("mutual exclusivity violated: RequiredCapacity=%f, SpareCapacity=%f",
					result.RequiredCapacity, result.SpareCapacity)
			}

			// Zero-supply guard
			if tc.wantTotalSupplyZero && result.TotalSupply != 0 {
				t.Errorf("TotalSupply = %f, want 0", result.TotalSupply)
			}
			if tc.wantTotalDemandZero && result.TotalDemand != 0 {
				t.Errorf("TotalDemand = %f, want 0", result.TotalDemand)
			}

			// Utilization guard: no NaN or Inf
			if math.IsNaN(result.Utilization) || math.IsInf(result.Utilization, 0) {
				t.Errorf("Utilization = %f, must not be NaN or Inf", result.Utilization)
			}

			// Aggregation invariant: sum(vc.Supply)==TotalSupply, sum(vc.Demand)==TotalDemand
			if tc.checkAggregation {
				var sumSupply, sumDemand float64
				for _, vc := range result.VariantCapacities {
					sumSupply += vc.Supply
					sumDemand += vc.Demand
				}
				if math.Abs(sumSupply-result.TotalSupply) > 1e-6 {
					t.Errorf("sum(vc.Supply)=%f != TotalSupply=%f", sumSupply, result.TotalSupply)
				}
				if math.Abs(sumDemand-result.TotalDemand) > 1e-6 {
					t.Errorf("sum(vc.Demand)=%f != TotalDemand=%f", sumDemand, result.TotalDemand)
				}
			}
		})
	}
}

// TestV2SaturationAnalyzerK1K2 verifies that effective capacity uses min(k1, k2).
func TestV2SaturationAnalyzerK1K2(t *testing.T) {
	// When k1 < k2 (memory is bottleneck), effective capacity should be k1.
	// We verify this indirectly: with low demand relative to k1, SpareCapacity should be
	// positive even if k2 would be higher.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  0.5,  // k1 = 10000 * 0.5 = 5000 tokens
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    512,
	}
	analyzer := NewV2SaturationAnalyzer(cfg)

	result := analyzer.Analyze(ModelSignals{
		ModelID: "m1",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "i1",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000, // k1 = 10000 * 0.5 = 5000
				KvTokensInUse:         100,   // very low demand
				QueueDepth:            0,
				CostPerHour:           10.0,
			},
			{
				InstanceID:            "i2",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000,
				KvTokensInUse:         100,
				QueueDepth:            0,
				CostPerHour:           10.0,
			},
			{
				InstanceID:            "i3",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000,
				KvTokensInUse:         100,
				QueueDepth:            0,
				CostPerHour:           10.0,
			},
		},
	})

	// Supply should be based on k1 (5000 tokens per replica × 3 replicas = 15000)
	// Demand = 300 tokens total. SpareCapacity should be positive.
	if result.TotalSupply <= 0 {
		t.Fatalf("TotalSupply = %f, want > 0", result.TotalSupply)
	}
	if result.SpareCapacity <= 0 {
		t.Errorf("SpareCapacity = %f, want > 0 (supply >> demand with 3 replicas)", result.SpareCapacity)
	}
	if result.RequiredCapacity > 0 {
		t.Errorf("RequiredCapacity = %f, want 0", result.RequiredCapacity)
	}
}

// TestV2SaturationAnalyzerN1MixedVariants verifies that the N-1 redistribution check
// uses the highest-capacity variant (conservative) to match Engine's most-expensive-first
// scale-down selection. Regression test for the min→max fix.
func TestV2SaturationAnalyzerN1MixedVariants(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  0.8,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.4,
		AvgInputTokens:    512,
	}
	analyzer := NewV2SaturationAnalyzer(cfg)

	// Scenario from review: A100 (cheap, low capacity) + H100 (expensive, high capacity).
	// Engine will remove H100 (most expensive). N-1 check must simulate removing the
	// highest-capacity replica (H100), not the lowest (A100).
	//
	// A100: 1 replica, totalKvCap=10000, k1=8000, demand=2000, supply=8000
	// H100: 1 replica, totalKvCap=25000, k1=20000, demand=2000, supply=20000
	// Total supply=28000, demand=4000
	// SpareCapacity = 28000 - 4000/0.4 = 28000 - 10000 = 18000 > 0
	//
	// N-1 with max (correct): remove H100's 20000 → supply=8000, 8000 > 10000? NO → no spare
	// N-1 with min (bug):     remove A100's 8000  → supply=20000, 20000 > 10000? YES → spare approved
	//
	// If the bug existed, Engine would remove H100, leaving only A100 (supply=8000)
	// which violates ScaleDownBoundary (need 10000).
	result := analyzer.Analyze(ModelSignals{
		ModelID: "m1",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "i1",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000,
				KvTokensInUse:         2000,
				QueueDepth:            0,
				CostPerHour:           10.0,
			},
			{
				InstanceID:            "i2",
				Variant:               NewVariantSpec("H100", 2),
				TotalKvCapacityTokens: 25000,
				KvTokensInUse:         2000,
				QueueDepth:            0,
				CostPerHour:           20.0,
			},
		},
	})

	// With conservative N-1 check (removing H100's capacity), scale-down should be blocked
	if result.SpareCapacity > 0 {
		t.Errorf("SpareCapacity = %f, want 0 (N-1 check should block: removing H100 leaves insufficient supply)", result.SpareCapacity)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_LoadingOnlyModel verifies that when all instances
// are Loading (Replicas == nil), the analyzer correctly returns all-zero output.
// A loading-only model has no routable replicas → no demand signal → RequiredCapacity = 0 is correct.
func TestV2SaturationAnalyzer_PendingSupply_LoadingOnlyModel(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID:                      "test-model",
		Replicas:                     nil, // no Active/WarmingUp instances
		PendingReplicaCount:          2,
		PendingTotalKvCapacityTokens: 20000,
	}

	result := a.Analyze(metrics)

	// No routable replicas → no demand can be measured → all-zero result is correct.
	if result.RequiredCapacity != 0 {
		t.Errorf("RequiredCapacity = %g, want 0 (no demand signal with zero routable replicas)", result.RequiredCapacity)
	}
	if result.TotalSupply != 0 {
		t.Errorf("TotalSupply = %g, want 0 (ready-replica-only)", result.TotalSupply)
	}
	if result.TotalDemand != 0 {
		t.Errorf("TotalDemand = %g, want 0 (no routable replicas to measure demand from)", result.TotalDemand)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_ThresholdApplied verifies that KvCacheThreshold
// is applied to pending supply (not just to ready-replica supply).
func TestV2SaturationAnalyzer_PendingSupply_ThresholdApplied(t *testing.T) {
	// KvCacheThreshold=0.8: effective pending supply = 10000 * 0.8 = 8000 (not 10000).
	// Active: capacity=10000, k1=8000. Demand=9000 → demand/threshold(0.8)=11250.
	// Without pending: requiredCapacity = 11250 - 8000 = 3250 > 0 → scale-up.
	// With pending (threshold applied correctly): pendingSupply = 10000 * 0.8 = 8000.
	//   totalSupplyForScaleUp = 8000 + 8000 = 16000 → 11250 < 16000 → no scale-up.
	// If threshold were NOT applied to pending: pendingSupply = 10000.
	//   totalSupplyForScaleUp = 8000 + 10000 = 18000 → 11250 < 18000 → also no scale-up (indistinguishable).
	// So we use a case where threshold-correct pending is just barely enough:
	//   Active: capacity=10000, k1=8000. Demand=12000 → demand/threshold(0.8)=15000.
	//   pendingSupply (threshold=0.8) = 10000*0.8 = 8000 → total=16000 > 15000 → suppressed.
	//   pendingSupply (threshold=1.0, wrong) = 10000 → total=18000 > 15000 → also suppressed.
	// Use threshold=0.5 to make the difference observable:
	//   Active: capacity=10000, k1=5000. Demand=5000 → demand/threshold(0.8)=6250.
	//   pendingSupply (threshold=0.5) = 10000*0.5 = 5000 → total=10000 > 6250 → suppressed.
	//   pendingSupply (threshold=1.0, wrong) = 10000 → total=15000 > 6250 → also suppressed.
	// To make it observable: set pending capacity so that threshold*pending < gap but 1.0*pending > gap.
	//   Active: capacity=10000, k1=5000. Demand=4500 → demand/threshold(0.8)=5625.
	//   gap = 5625 - 5000 = 625.
	//   pendingKvCap=1000 → pendingSupply(0.5)=500 < 625 → scale-up still fires.
	//   pendingKvCap=1000 → pendingSupply(1.0, wrong)=1000 > 625 → no scale-up (wrong behavior).
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  0.5,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         4500,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000, // k1 = 10000 * 0.5 = 5000
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 1000, // pendingSupply(0.5)=500; pendingSupply(1.0)=1000
	}

	result := a.Analyze(metrics)

	// With KvCacheThreshold=0.5 applied to pending: pendingSupply=500 → gap=625 not covered → scale-up.
	if result.RequiredCapacity <= 0 {
		t.Errorf("RequiredCapacity = %g, want > 0 (threshold correctly applied to pending: 500 < gap 625)", result.RequiredCapacity)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_SuppressesScaleUp verifies the core fix for #1109:
// when a Loading instance's capacity covers the demand gap, no further scale-up is emitted.
func TestV2SaturationAnalyzer_PendingSupply_SuppressesScaleUp(t *testing.T) {
	// Configuration: ScaleUpThreshold=0.8 means scale-up fires when demand > 0.8 * supply.
	// Active replica: capacity=10000, current demand=9000 → demand/threshold=11250 > 10000 → scale-up.
	// After fix: 1 Loading replica adds 10000 pending capacity.
	// totalSupplyForScaleUp = 10000 + 10000 = 20000 → demand/threshold=11250 < 20000 → no scale-up.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         9000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 10000,
	}

	result := a.Analyze(metrics)

	if result.RequiredCapacity != 0 {
		t.Errorf("RequiredCapacity = %g, want 0 (loading replica covers demand gap)", result.RequiredCapacity)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_ZeroCapacityLoading verifies the analyzer's > 0 guard
// on PendingTotalKvCapacityTokens: a ModelSignals with PendingReplicaCount=1 but
// PendingTotalKvCapacityTokens=0 must contribute no pending supply.
// Note: DefaultCollector now skips zero-capacity loading snapshots entirely, so this state is
// only reachable via direct ModelSignals construction (e.g., future Collector implementations).
// The test pins the analyzer's own guard to prevent a regression where PendingReplicaCount > 0
// is checked instead of PendingTotalKvCapacityTokens > 0.
func TestV2SaturationAnalyzer_PendingSupply_ZeroCapacityLoading(t *testing.T) {
	// Active replica: capacity=10000, demand=9000 → demand/threshold(0.8)=11250 > 10000 → scale-up needed.
	// Loading replica: PendingReplicaCount=1 but PendingTotalKvCapacityTokens=0 (zero-KV node).
	// pendingSupply must be 0 → RequiredCapacity = 11250 - 10000 = 1250 > 0.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         9000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 0, // zero-KV loading instance — contributes no pending supply
	}

	result := a.Analyze(metrics)

	if result.RequiredCapacity <= 0 {
		t.Errorf("RequiredCapacity = %g, want > 0 (zero-KV loading instance must not suppress scale-up)", result.RequiredCapacity)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_StillScalesUpForDelta verifies that when
// demand has grown beyond what the Loading instance covers, scale-up still fires for the delta.
func TestV2SaturationAnalyzer_PendingSupply_StillScalesUpForDelta(t *testing.T) {
	// Active: capacity=10000, demand=18000 → demand/threshold(0.8)=22500.
	// 1 Loading replica: pending capacity=10000.
	// totalSupplyForScaleUp = 10000 + 10000 = 20000 → requiredCapacity = 22500-20000 = 2500 > 0.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         18000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 10000,
	}

	result := a.Analyze(metrics)

	// demand/threshold = 18000/0.8 = 22500; totalSupplyForScaleUp = 10000+10000 = 20000; delta = 2500.
	if result.RequiredCapacity != 2500 {
		t.Errorf("RequiredCapacity = %g, want 2500 (demand exceeds ready+pending supply by exactly 2500)", result.RequiredCapacity)
	}
	// The ready-only supply (10000) must not be inflated by pending.
	if result.TotalSupply != 10000 {
		t.Errorf("TotalSupply = %g, want 10000 (ready supply only; pending does not inflate TotalSupply)", result.TotalSupply)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_DoesNotAffectScaleDown verifies that pending
// supply does NOT inflate TotalSupply used for SpareCapacity (no premature scale-down).
func TestV2SaturationAnalyzer_PendingSupply_DoesNotAffectScaleDown(t *testing.T) {
	// 2 Active replicas with low utilization → spare capacity signal.
	// 1 Loading replica: pending capacity=10000.
	// TotalSupply must be the ready-only value (20000), not 30000.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         1000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
			{
				InstanceID:            "active-2",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         1000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 10000,
	}

	result := a.Analyze(metrics)

	// TotalSupply must be ready-only (20000). Pending (10000) must not be included.
	if result.TotalSupply != 20000 {
		t.Errorf("TotalSupply = %g, want 20000 (ready-only; pending must not inflate)", result.TotalSupply)
	}
	// SpareCapacity must be based on ready-only supply (no premature scale-down risk from pending).
	// demand=2000, ScaleDownBoundary=0.3, supply=20000:
	// spareCapacity = 20000 - (2000/0.3) = 20000 - 6667 = 13333 > 0
	// N-1 check: supplyAfterRemoval=10000 > 2000/0.3=6667 → SpareCapacity is set.
	if result.SpareCapacity <= 0 {
		t.Errorf("SpareCapacity = %g, want > 0 (low utilization with 2 ready replicas)", result.SpareCapacity)
	}
}

// TestV2SaturationAnalyzer_DynamicAvgInputTokens verifies T1: when AvgInTokens > 0 on a
// replica the observed value is used for demand, not config.AvgInputTokens.
func TestV2SaturationAnalyzer_DynamicAvgInputTokens(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    512, // config fallback
	}

	// Observed average is much smaller than config (e.g. short-context workload).
	// With config=512 and queue=10: demand = 0 + 10*512 = 5120.
	// With observed=64:             demand = 0 + 10*64  = 640.
	// At supply=10000, config demand pushes utilization to 0.512 (no signal).
	// Observed demand of 640 gives utilization 0.064 (even more idle → spare capacity).
	// The key assertion: TotalDemand must match the observed value, not the config value.
	observedAvgIn := 64.0
	queue := 10

	a := NewV2SaturationAnalyzer(cfg)
	result := a.Analyze(ModelSignals{
		ModelID: "m1",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "i1",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000,
				KvTokensInUse:         0,
				QueueDepth:            queue,
				AvgInTokens:           observedAvgIn,
				CostPerHour:           10.0,
			},
		},
	})

	wantDemand := float64(queue) * observedAvgIn
	if math.Abs(result.TotalDemand-wantDemand) > 1e-6 {
		t.Errorf("TotalDemand = %f, want %f (observed AvgInTokens=%f should override config=%f)",
			result.TotalDemand, wantDemand, observedAvgIn, cfg.AvgInputTokens)
	}
}

// TestV2SaturationAnalyzer_AvgInputTokensFallback verifies that config.AvgInputTokens is
// used when AvgInTokens == 0 (cold start — no completed requests yet).
func TestV2SaturationAnalyzer_AvgInputTokensFallback(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    256,
	}
	queue := 5

	a := NewV2SaturationAnalyzer(cfg)
	result := a.Analyze(ModelSignals{
		ModelID: "m1",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "i1",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000,
				KvTokensInUse:         0,
				QueueDepth:            queue,
				AvgInTokens:           0, // cold start
				CostPerHour:           10.0,
			},
		},
	})

	wantDemand := float64(queue) * cfg.AvgInputTokens
	if math.Abs(result.TotalDemand-wantDemand) > 1e-6 {
		t.Errorf("TotalDemand = %f, want %f (should fall back to config AvgInputTokens=%f when AvgInTokens=0)",
			result.TotalDemand, wantDemand, cfg.AvgInputTokens)
	}
}

// TestV2SaturationAnalyzer_K2ComputeBound verifies T2: when MaxBatchSize, AvgInTokens, and
// AvgOutTokens are all populated, k2 is derived from the WVA formula and — when
// k2 < k1 — the effective capacity is capped at k2.
func TestV2SaturationAnalyzer_K2ComputeBound(t *testing.T) {
	// Scenario: short-context, many-output workload where compute saturates before KV.
	// I=64 (short prompt), O=512 (long output), B=32 (max batch).
	// nSteady = 32 * 512 / (64 + 512) = 16384 / 576 ≈ 28.44
	// k2 = 28.44 * (64 + 256) = 28.44 * 320 ≈ 9101
	// k1 = 100000 * 1.0 = 100000  (large KV — memory is NOT the bottleneck)
	// effectiveCapacity = min(100000, 9101) = 9101
	//
	// Without k2: supply = 100000, demand = 1000 → no scale-up.
	// With k2:    supply = 9101,   demand = 1000 → also no scale-up here, but supply is correctly bounded.
	// To test scale-up: set demand > k2 * ScaleUpThreshold.
	// demand > 9101 * 0.8 = 7281 → use KvTokensInUse=8000.
	I := 64.0
	O := 512.0
	B := 32.0
	nSteady := B * O / (I + O)
	k2Expected := nSteady * (I + O/2) // ≈ 9101

	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    64,
	}
	a := NewV2SaturationAnalyzer(cfg)

	// First call seeds the rolling history with one sample.
	result := a.Analyze(ModelSignals{
		ModelID: "m1",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "i1",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 100000, // k1 >> k2 — memory is not the bottleneck
				KvTokensInUse:         8000,   // demand > k2*threshold → should trigger scale-up
				QueueDepth:            0,
				AvgInTokens:           I,
				AvgOutTokens:          O,
				MaxBatchSize:          B,
				CostPerHour:           10.0,
			},
		},
	})

	// Supply must be bounded by k2 (≈ 9101), not k1 (100000).
	// A single sample rolling average equals the derived value exactly.
	if math.Abs(result.TotalSupply-k2Expected) > 1.0 {
		t.Errorf("TotalSupply = %.1f, want ≈%.1f (k2 from WVA formula should bound supply, not k1=100000)",
			result.TotalSupply, k2Expected)
	}

	// With supply ≈ 9101 and KvTokensInUse=8000:
	// demand/ScaleUpThreshold = 8000/0.8 = 10000 > supply(9101) → RequiredCapacity > 0.
	if result.RequiredCapacity <= 0 {
		t.Errorf("RequiredCapacity = %f, want > 0 (demand 8000 exceeds k2-bounded supply %.1f * threshold 0.8)",
			result.RequiredCapacity, k2Expected)
	}
}

// TestV2SaturationAnalyzer_K2FallbackToK1 verifies that when batch parameters are absent
// (cold start: AvgInTokens=0 or AvgOutTokens=0 or MaxBatchSize=0), k2 falls back to k1
// and the effective capacity is unchanged from the memory-bound value.
func TestV2SaturationAnalyzer_K2FallbackToK1(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  0.8,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    512,
	}
	a := NewV2SaturationAnalyzer(cfg)

	// No AvgInTokens, AvgOutTokens, or MaxBatchSize → k2 = k1.
	// k1 = 10000 * 0.8 = 8000. With demand=1000, supply should be 8000.
	result := a.Analyze(ModelSignals{
		ModelID: "m1",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "i1",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000,
				KvTokensInUse:         1000,
				QueueDepth:            0,
				AvgInTokens:           0, // no data
				AvgOutTokens:          0,
				MaxBatchSize:          0,
				CostPerHour:           10.0,
			},
		},
	})

	wantSupply := 10000.0 * 0.8 // k1 = k2 fallback
	if math.Abs(result.TotalSupply-wantSupply) > 1e-6 {
		t.Errorf("TotalSupply = %f, want %f (k2 should fall back to k1 when batch params absent)",
			result.TotalSupply, wantSupply)
	}
}

// TestV2SaturationAnalyzer_K2RollingAverage verifies that k2 converges toward the rolling
// average over multiple Analyze calls rather than using only the most recent derived value.
func TestV2SaturationAnalyzer_K2RollingAverage(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    512,
	}
	a := NewV2SaturationAnalyzer(cfg)

	// Seed with 5 calls at k2≈1000, then switch to k2≈5000.
	// After the switch, the rolling average should be between 1000 and 5000
	// (not immediately jump to 5000).

	// k2 = nSteady*(I+O/2) where nSteady = B*O/(I+O).
	// Call with I=100, O=100, B=40: nSteady=20, k2=20*(100+50)=3000 → seed value.
	replica := func(avgIn, avgOut, maxBatch float64) ReplicaMetrics {
		return ReplicaMetrics{
			InstanceID:            "i1",
			Variant:               NewVariantSpec("A100", 1),
			TotalKvCapacityTokens: 1000000, // huge — k1 never binds
			KvTokensInUse:         100,
			QueueDepth:            0,
			AvgInTokens:           avgIn,
			AvgOutTokens:          avgOut,
			MaxBatchSize:          maxBatch,
			CostPerHour:           10.0,
		}
	}

	// I=100, O=100, B=40: nSteady=40*100/200=20, k2=20*150=3000
	for i := 0; i < 5; i++ {
		a.Analyze(ModelSignals{ModelID: "m1", Replicas: []ReplicaMetrics{replica(100, 100, 40)}})
	}

	// Switch to I=10, O=900, B=40: nSteady=40*900/910≈39.6, k2=39.6*(10+450)≈18200
	// (much larger k2 — long output, compute is less of a bottleneck)
	result := a.Analyze(ModelSignals{ModelID: "m1", Replicas: []ReplicaMetrics{replica(10, 900, 40)}})

	// After 5 samples of ≈3000 and 1 sample of ≈18200, rolling average should be between 3000 and 18200.
	if result.TotalSupply <= 3000 {
		t.Errorf("TotalSupply = %.1f: rolling average should be > seed value 3000 after high-k2 sample", result.TotalSupply)
	}
	if result.TotalSupply >= 18200 {
		t.Errorf("TotalSupply = %.1f: rolling average should be < single high-k2 sample 18200 (history smoothing)", result.TotalSupply)
	}
}

// TestV2SaturationAnalyzerConfigValidation verifies constructor rejects invalid configs.
func TestV2SaturationAnalyzerConfigValidation(t *testing.T) {
	tests := []struct {
		name string
		cfg  V2SaturationAnalyzerConfig
	}{
		{"zero KvCacheThreshold", V2SaturationAnalyzerConfig{KvCacheThreshold: 0, ScaleUpThreshold: 0.8, ScaleDownBoundary: 0.4, AvgInputTokens: 512}},
		{"KvCacheThreshold > 1.0", V2SaturationAnalyzerConfig{KvCacheThreshold: 1.5, ScaleUpThreshold: 0.8, ScaleDownBoundary: 0.4, AvgInputTokens: 512}},
		{"negative ScaleUpThreshold", V2SaturationAnalyzerConfig{KvCacheThreshold: 0.8, ScaleUpThreshold: -1, ScaleDownBoundary: 0.4, AvgInputTokens: 512}},
		{"NaN ScaleDownBoundary", V2SaturationAnalyzerConfig{KvCacheThreshold: 0.8, ScaleUpThreshold: 0.8, ScaleDownBoundary: math.NaN(), AvgInputTokens: 512}},
		{"zero AvgInputTokens", V2SaturationAnalyzerConfig{KvCacheThreshold: 0.8, ScaleUpThreshold: 0.8, ScaleDownBoundary: 0.4, AvgInputTokens: 0}},
		{"ScaleDownBoundary >= ScaleUpThreshold", V2SaturationAnalyzerConfig{KvCacheThreshold: 0.8, ScaleUpThreshold: 0.4, ScaleDownBoundary: 0.8, AvgInputTokens: 512}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic for invalid config, got none")
				}
			}()
			NewV2SaturationAnalyzer(tc.cfg)
		})
	}
}
