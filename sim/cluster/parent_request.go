package cluster

import "github.com/inference-sim/inference-sim/sim"

// ParentRequest tracks the disaggregated lifecycle of a request that was split
// into prefill and decode sub-requests. Owned by ClusterSimulator.
type ParentRequest struct {
	ID              string       // Original request ID
	OriginalRequest *sim.Request // Pointer to the original request (for metadata)
	PrefillSubReqID string
	DecodeSubReqID  string
	DecodeSubReq    *sim.Request // Pointer to the decode sub-request. Set by KVTransferStartedEvent on successful ReserveTransferredKV (issue #1343); KVTransferCompletedEvent promotes it from StateWaitingForRemoteKVs to StateQueued and enqueues it on the decode instance. Nil after a late-drop (decode pod unroutable at transfer complete) when reserved KV is released.
	PrefillSubReq   *sim.Request // Pointer to the prefill sub-request, retained at routing so a prefill timeout can be detected and the decode in-flight reservation (taken at selection) released. See detectPrefillTimeouts.
	NumKVBlocks     int64        // KV blocks to transfer (ceil(inputLen / blockSize))

	// Phase timestamps (microseconds). Zero means phase not yet reached.
	ArrivalTime          int64
	PrefillEnqueueTime   int64
	PrefillCompleteTime  int64
	TransferStartTime    int64
	TransferCompleteTime int64
	DecodeEnqueueTime    int64
	// Schedule instants (microseconds): when each sub-request first entered the
	// running batch (waiting → running). Zero = not yet scheduled. Populated via
	// recordAdmissionTime from the OnAdmit hook, only when --pd-outcome-trace is set.
	// Used to compute realized admission delay T_adm = schedule − enqueue (§3.8).
	PrefillScheduleTime int64
	DecodeScheduleTime  int64
	// CompletionTime has four meanings depending on outcome:
	//   - Successful decode: set by detectDecodeCompletions to
	//     clusterClock + decodeInstance.PostDecodeFixedOverhead() when the decode
	//     sub-request finishes its last step. Includes PostDecodeFixedOverhead so
	//     that projectPDMetrics() computes the same client-visible E2E as non-PD
	//     recordRequestCompletion (issue #846). For roofline (overhead=0), equals
	//     the raw cluster clock tick.
	//   - Dropped at transfer start (issue #1343): set to KVTransferStartedEvent
	//     time when ReserveTransferredKV fails or the decode pod is non-routable.
	//     Signature: TransferStartTime > 0 && CompletionTime == TransferStartTime
	//     && TransferCompleteTime == 0. No KVTransferCompletedEvent is scheduled.
	//   - Dropped at transfer complete: set to KVTransferCompletedEvent time when
	//     the decode pod transitioned non-routable during the transfer window.
	//     Reserved KV is released before the drop. Signature:
	//     TransferCompleteTime > 0 && DecodeSubReq == nil.
	//   - Decode timeout: set by detectDecodeCompletions to the cluster clock at
	//     timeout detection time. The session is cancelled via sessionCallback.
	CompletionTime int64

	// Instance assignment.
	// DecodeInstanceID is set upfront at executeDisaggregatedRouting time (decode-first routing),
	// before prefill routing begins. PrefillInstanceID is set by PrefillRoutingEvent.
	PrefillInstanceID InstanceID
	DecodeInstanceID  InstanceID

	// EncodeInstanceID is set by the encode routing stage in executeDisaggregatedRouting
	// when an encode decider approves. Zero value (empty) = encode did not fire for this
	// request. GAP-4, issue #1264.
	EncodeInstanceID InstanceID

	// PrefillPodHint is a forced prefill instance supplied by a joint/fixed-plan
	// decider (DisaggregationDecision.PrefillPodHint). When non-empty,
	// PrefillRoutingEvent routes the prefill sub-request to this instance instead
	// of invoking the scorer. Empty ⇒ normal scorer routing.
	PrefillPodHint string
}

// NewParentRequest creates a ParentRequest from the original request.
// blockSizeTokens must be > 0 (validated by KVCacheConfig constructor).
func NewParentRequest(req *sim.Request, blockSizeTokens int64) *ParentRequest {
	if blockSizeTokens <= 0 {
		panic("NewParentRequest: blockSizeTokens must be > 0")
	}
	inputLen := req.InputLen()
	numBlocks := (inputLen + blockSizeTokens - 1) / blockSizeTokens
	return &ParentRequest{
		ID:              req.ID,
		OriginalRequest: req,
		PrefillSubReqID: req.ID + "_prefill",
		DecodeSubReqID:  req.ID + "_decode",
		NumKVBlocks:     numBlocks,
		ArrivalTime:     req.ArrivalTime,
	}
}
