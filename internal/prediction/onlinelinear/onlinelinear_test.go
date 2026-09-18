/*
Copyright 2026 The Predictive Horizontal Pod Autoscaler Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package onlinelinear

import (
	"math"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	jamiethompsonmev1alpha1 "github.com/jthomperoo/predictive-horizontal-pod-autoscaler/api/v1alpha1"
)

func TestPredictDatapointWarmupAndForecastError(t *testing.T) {
	learningRate := 0.5
	warmup := int64(2)
	model := onlineModel(jamiethompsonmev1alpha1.OnlineUpdateDatapoint, 1, learningRate, warmup)
	predictor := &Predict{}
	history := newHistory()
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	first, err := predictor.Process(model, history, observation(start, 1), true)
	if err != nil {
		t.Fatal(err)
	}
	if first.Ready || first.Prediction != nil {
		t.Fatalf("first sample should be warming up: %+v", first)
	}

	second, err := predictor.Process(model, first.History, observation(start.Add(10*time.Second), 2), true)
	if err != nil {
		t.Fatal(err)
	}
	if !second.Ready || second.Prediction == nil || *second.Prediction != 3 {
		t.Fatalf("second sample should predict 3 replicas: %+v", second)
	}

	third, err := predictor.Process(model, second.History, observation(start.Add(20*time.Second), 3), true)
	if err != nil {
		t.Fatal(err)
	}
	if third.MeanAbsoluteError == nil || *third.MeanAbsoluteError != 0 {
		t.Fatalf("expected zero prequential MAE, got %+v", third.MeanAbsoluteError)
	}
	if third.History.OnlineLinearState.SamplesSeen != 3 ||
		third.History.OnlineLinearState.UpdatesApplied != 3 {
		t.Fatalf("unexpected counters: %+v", third.History.OnlineLinearState)
	}
}

func TestDefaults(t *testing.T) {
	model := &jamiethompsonmev1alpha1.Model{
		OnlineLinear: &jamiethompsonmev1alpha1.OnlineLinear{LookAhead: 10000},
	}
	config, err := Defaults(model)
	if err != nil {
		t.Fatal(err)
	}
	if config.UpdateMode != jamiethompsonmev1alpha1.OnlineUpdateDatapoint || config.BatchSize != 1 ||
		config.LearningRate != 0.01 || config.WarmupSamples != 6 || config.CheckpointInterval != time.Minute ||
		config.Mode != jamiethompsonmev1alpha1.OnlineModeActive {
		t.Fatalf("unexpected datapoint defaults: %+v", config)
	}

	minibatch := jamiethompsonmev1alpha1.OnlineUpdateMinibatch
	model.OnlineLinear.UpdateMode = &minibatch
	config, err = Defaults(model)
	if err != nil {
		t.Fatal(err)
	}
	if config.BatchSize != 6 {
		t.Fatalf("unexpected minibatch default size: %+v", config)
	}
}

func TestPredictMinibatchOnlyUpdatesAtBoundary(t *testing.T) {
	model := onlineModel(jamiethompsonmev1alpha1.OnlineUpdateMinibatch, 3, 0.1, 3)
	predictor := &Predict{}
	history := newHistory()
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	for i := 0; i < 2; i++ {
		result, err := predictor.Process(model, history, observation(start.Add(time.Duration(i)*10*time.Second), int32(i+1)), true)
		if err != nil {
			t.Fatal(err)
		}
		history = result.History
		if result.Updated || history.OnlineLinearState.UpdatesApplied != 0 {
			t.Fatalf("minibatch updated before boundary: %+v", history.OnlineLinearState)
		}
	}

	result, err := predictor.Process(model, history, observation(start.Add(20*time.Second), 3), true)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Updated || !result.Ready || result.History.OnlineLinearState.UpdatesApplied != 1 ||
		len(result.History.OnlineLinearState.PendingSamples) != 0 {
		t.Fatalf("minibatch did not update at boundary: %+v", result.History.OnlineLinearState)
	}
}

func TestPredictConstantDemandDoesNotCreatePhantomReplica(t *testing.T) {
	model := onlineModel(jamiethompsonmev1alpha1.OnlineUpdateDatapoint, 1, 0.1, 2)
	predictor := &Predict{}
	history := newHistory()
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	var result Result
	for i := 0; i < 40; i++ {
		var err error
		result, err = predictor.Process(model, history, observation(start.Add(time.Duration(i)*10*time.Second), 1), true)
		if err != nil {
			t.Fatal(err)
		}
		history = result.History
	}
	if result.Prediction == nil || *result.Prediction != 1 {
		t.Fatalf("constant demand predicted %+v", result.Prediction)
	}
}

func TestPredictFallingDemand(t *testing.T) {
	model := onlineModel(jamiethompsonmev1alpha1.OnlineUpdateDatapoint, 1, 0.5, 2)
	predictor := &Predict{}
	history := newHistory()
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	var result Result
	for i, replicas := range []int32{5, 4, 3} {
		var err error
		result, err = predictor.Process(model, history,
			observation(start.Add(time.Duration(i)*10*time.Second), replicas), true)
		if err != nil {
			t.Fatal(err)
		}
		history = result.History
	}
	if result.Prediction == nil || *result.Prediction >= 3 {
		t.Fatalf("falling demand did not produce a lower forecast: %+v", result.Prediction)
	}
}

func TestPredictionRoundsUpAndClampsNegativeValues(t *testing.T) {
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	state := &jamiethompsonmev1alpha1.OnlineLinearState{
		OriginTime: &metav1.Time{Time: start},
		Intercept:  1.01,
	}
	rounded, err := predict(state, start, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if rounded != 2 {
		t.Fatalf("prediction was not rounded up: %d", rounded)
	}

	state.Intercept = -2
	clamped, err := predict(state, start, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if clamped != 0 {
		t.Fatalf("negative prediction was not clamped: %d", clamped)
	}
}

func TestRebasePreservesPrediction(t *testing.T) {
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	state := &jamiethompsonmev1alpha1.OnlineLinearState{
		OriginTime:  &metav1.Time{Time: start},
		Intercept:   2,
		Coefficient: 0.25,
	}
	target := start.Add(400 * time.Second)
	before, err := predict(state, target, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	rebaseIfNeeded(state, target, 10*time.Second)
	after, err := predict(state, target, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("rebase changed prediction from %d to %d", before, after)
	}
}

func TestInvalidOrChangedStateResets(t *testing.T) {
	model := onlineModel(jamiethompsonmev1alpha1.OnlineUpdateDatapoint, 1, 0.1, 2)
	history := newHistory()
	history.OnlineLinearState = &jamiethompsonmev1alpha1.OnlineLinearState{
		Generation:  "old",
		Intercept:   math.NaN(),
		SamplesSeen: 99,
	}

	result, err := (&Predict{}).Process(model, history,
		observation(time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), 4), false)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Reset || !result.ForceCheckpoint || result.History.OnlineLinearState.SamplesSeen != 1 {
		t.Fatalf("invalid state was not reset: %+v", result)
	}
}

func onlineModel(mode string, batchSize int, learningRate float64, warmup int64) *jamiethompsonmev1alpha1.Model {
	active := jamiethompsonmev1alpha1.OnlineModeActive
	return &jamiethompsonmev1alpha1.Model{
		Type: jamiethompsonmev1alpha1.TypeOnlineLinear,
		Name: "test-online",
		OnlineLinear: &jamiethompsonmev1alpha1.OnlineLinear{
			LookAhead:     10000,
			UpdateMode:    &mode,
			BatchSize:     &batchSize,
			LearningRate:  &learningRate,
			WarmupSamples: &warmup,
			Mode:          &active,
		},
	}
}

func newHistory() jamiethompsonmev1alpha1.ModelHistory {
	return jamiethompsonmev1alpha1.ModelHistory{
		Type:              jamiethompsonmev1alpha1.TypeOnlineLinear,
		SyncPeriodsPassed: 1,
	}
}

func observation(timestamp time.Time, replicas int32) jamiethompsonmev1alpha1.TimestampedReplicas {
	return jamiethompsonmev1alpha1.TimestampedReplicas{
		Time:     &metav1.Time{Time: timestamp},
		Replicas: replicas,
	}
}
