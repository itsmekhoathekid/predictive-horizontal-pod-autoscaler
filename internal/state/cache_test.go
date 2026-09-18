/*
Copyright 2026 The Predictive Horizontal Pod Autoscaler Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package state

import (
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/types"

	jamiethompsonmev1alpha1 "github.com/jthomperoo/predictive-horizontal-pod-autoscaler/api/v1alpha1"
)

func TestCacheMigratesAndCheckpointsLegacyState(t *testing.T) {
	cache := NewCache()
	key := types.NamespacedName{Namespace: "default", Name: "example"}
	persisted := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData{
		ModelHistories: map[string]jamiethompsonmev1alpha1.ModelHistory{"legacy": {Type: "Linear"}},
	}

	loaded, force := cache.Load(key, persisted)
	if !force || loaded.SchemaVersion != SchemaVersion {
		t.Fatalf("legacy state was not migrated: force=%t data=%+v", force, loaded)
	}
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	if !cache.CheckpointDue(key, now, time.Minute, false) {
		t.Fatal("state without a checkpoint should be due")
	}
	checkpointed := cache.MarkCheckpoint(key, loaded, now)
	if checkpointed.LastCheckpointTime == nil {
		t.Fatal("checkpoint timestamp was not stored")
	}
	if cache.CheckpointDue(key, now.Add(59*time.Second), time.Minute, false) {
		t.Fatal("checkpoint became due too early")
	}
	if !cache.CheckpointDue(key, now.Add(time.Minute), time.Minute, false) {
		t.Fatal("checkpoint did not become due at the interval")
	}
}

func TestCacheReturnsCopies(t *testing.T) {
	cache := NewCache()
	key := types.NamespacedName{Name: "example"}
	persisted := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData{
		SchemaVersion:  SchemaVersion,
		ModelHistories: map[string]jamiethompsonmev1alpha1.ModelHistory{},
	}
	first, _ := cache.Load(key, persisted)
	first.ModelHistories["mutated"] = jamiethompsonmev1alpha1.ModelHistory{}
	second, _ := cache.Load(key, persisted)
	if _, exists := second.ModelHistories["mutated"]; exists {
		t.Fatal("caller mutation leaked into cached state")
	}
}

func TestCacheResetsUnsupportedFutureSchema(t *testing.T) {
	cache := NewCache()
	key := types.NamespacedName{Namespace: "default", Name: "example"}
	persisted := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData{
		SchemaVersion: SchemaVersion + 1,
		ModelHistories: map[string]jamiethompsonmev1alpha1.ModelHistory{
			"unsafe": {Type: jamiethompsonmev1alpha1.TypeOnlineLinear},
		},
	}

	loaded, force := cache.Load(key, persisted)
	if !force {
		t.Fatal("future schema should force a replacement checkpoint")
	}
	if loaded.SchemaVersion != SchemaVersion || len(loaded.ModelHistories) != 0 {
		t.Fatalf("future schema was not reset safely: %+v", loaded)
	}
}

func TestCacheRestoresCheckpointInNewProcess(t *testing.T) {
	key := types.NamespacedName{Namespace: "default", Name: "example"}
	persisted := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData{
		SchemaVersion: SchemaVersion,
		ModelHistories: map[string]jamiethompsonmev1alpha1.ModelHistory{
			"online": {
				Type: jamiethompsonmev1alpha1.TypeOnlineLinear,
				OnlineLinearState: &jamiethompsonmev1alpha1.OnlineLinearState{
					SamplesSeen:    12,
					UpdatesApplied: 2,
					PendingSamples: []jamiethompsonmev1alpha1.OnlineLinearSample{{Replicas: 3}},
				},
			},
		},
	}

	restored, force := NewCache().Load(key, persisted)
	state := restored.ModelHistories["online"].OnlineLinearState
	if force || state == nil || state.SamplesSeen != 12 || state.UpdatesApplied != 2 ||
		len(state.PendingSamples) != 1 {
		t.Fatalf("checkpoint state was not restored: force=%t state=%+v", force, state)
	}
}
