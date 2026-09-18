/*
Copyright 2026 The Predictive Horizontal Pod Autoscaler Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

// Package onlinelinear implements an incrementally trained linear replica forecaster.
package onlinelinear

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	jamiethompsonmev1alpha1 "github.com/jthomperoo/predictive-horizontal-pod-autoscaler/api/v1alpha1"
)

const (
	defaultLearningRate     = 0.01
	defaultWarmupSamples    = int64(6)
	defaultMinibatchSize    = 6
	defaultCheckpoint       = time.Minute
	maximumNormalizedTime   = 32.0
	maximumGradientNorm     = 100.0
	maximumPendingForecasts = 1024
)

// Config is the fully defaulted OnlineLinear configuration.
type Config struct {
	LookAhead          time.Duration
	UpdateMode         string
	BatchSize          int
	LearningRate       float64
	WarmupSamples      int64
	CheckpointInterval time.Duration
	Mode               string
}

// Result contains an updated history and an optional prediction.
type Result struct {
	History           jamiethompsonmev1alpha1.ModelHistory
	Prediction        *int32
	Ready             bool
	Updated           bool
	Reset             bool
	ForceCheckpoint   bool
	MeanAbsoluteError *float64
}

// Processor updates and evaluates an OnlineLinear model.
type Processor interface {
	Process(
		model *jamiethompsonmev1alpha1.Model,
		history jamiethompsonmev1alpha1.ModelHistory,
		observation jamiethompsonmev1alpha1.TimestampedReplicas,
		shouldPredict bool,
	) (Result, error)
}

// Predict implements Processor.
type Predict struct{}

// Defaults resolves optional configuration values.
func Defaults(model *jamiethompsonmev1alpha1.Model) (Config, error) {
	if model == nil || model.OnlineLinear == nil {
		return Config{}, errors.New("no OnlineLinear configuration provided for model")
	}

	online := model.OnlineLinear
	config := Config{
		LookAhead:          time.Duration(online.LookAhead) * time.Millisecond,
		UpdateMode:         jamiethompsonmev1alpha1.OnlineUpdateDatapoint,
		BatchSize:          1,
		LearningRate:       defaultLearningRate,
		WarmupSamples:      defaultWarmupSamples,
		CheckpointInterval: defaultCheckpoint,
		Mode:               jamiethompsonmev1alpha1.OnlineModeActive,
	}

	if online.UpdateMode != nil {
		config.UpdateMode = *online.UpdateMode
	}
	if config.UpdateMode == jamiethompsonmev1alpha1.OnlineUpdateMinibatch {
		config.BatchSize = defaultMinibatchSize
	}
	if online.BatchSize != nil {
		config.BatchSize = *online.BatchSize
	}
	if online.LearningRate != nil {
		config.LearningRate = *online.LearningRate
	}
	if online.WarmupSamples != nil {
		config.WarmupSamples = *online.WarmupSamples
	}
	if online.CheckpointInterval != nil {
		config.CheckpointInterval = online.CheckpointInterval.Duration
	}
	if online.Mode != nil {
		config.Mode = *online.Mode
	}

	return config, nil
}

// Generation returns a stable fingerprint for settings that change learned parameter semantics.
func Generation(config Config) string {
	payload, err := json.Marshal(struct {
		LookAhead    int64   `json:"lookAhead"`
		UpdateMode   string  `json:"updateMode"`
		BatchSize    int     `json:"batchSize"`
		LearningRate float64 `json:"learningRate"`
	}{
		LookAhead:    config.LookAhead.Milliseconds(),
		UpdateMode:   config.UpdateMode,
		BatchSize:    config.BatchSize,
		LearningRate: config.LearningRate,
	})
	if err != nil {
		panic(err)
	}
	hash := sha256.Sum256(payload)
	return hex.EncodeToString(hash[:8])
}

// Process evaluates due forecasts, trains with an observation, and optionally forecasts future demand.
func (p *Predict) Process(
	model *jamiethompsonmev1alpha1.Model,
	history jamiethompsonmev1alpha1.ModelHistory,
	observation jamiethompsonmev1alpha1.TimestampedReplicas,
	shouldPredict bool,
) (Result, error) {
	config, err := Defaults(model)
	if err != nil {
		return Result{}, err
	}
	if observation.Time == nil {
		return Result{}, errors.New("OnlineLinear observation has no timestamp")
	}

	generation := Generation(config)
	state := history.OnlineLinearState
	reset := state == nil || state.Generation != generation || !validState(state)
	if reset {
		state = &jamiethompsonmev1alpha1.OnlineLinearState{Generation: generation}
		history.ReplicaHistory = nil
	}

	if model.ResetDuration != nil && state.LastObservationTime != nil &&
		observation.Time.Sub(state.LastObservationTime.Time) > model.ResetDuration.Duration {
		state = &jamiethompsonmev1alpha1.OnlineLinearState{Generation: generation}
		reset = true
	}

	evaluateForecasts(state, observation)
	state.SamplesSeen++
	state.LastObservationTime = observation.Time.DeepCopy()

	if state.OriginTime == nil {
		state.OriginTime = observation.Time.DeepCopy()
		state.Intercept = float64(observation.Replicas)
	}
	rebaseIfNeeded(state, observation.Time.Time, config.LookAhead)

	updated := false
	sample := jamiethompsonmev1alpha1.OnlineLinearSample{
		Time:     observation.Time.DeepCopy(),
		Replicas: observation.Replicas,
	}
	if config.UpdateMode == jamiethompsonmev1alpha1.OnlineUpdateMinibatch {
		state.PendingSamples = append(state.PendingSamples, sample)
		if len(state.PendingSamples) >= config.BatchSize {
			batch := append([]jamiethompsonmev1alpha1.OnlineLinearSample(nil), state.PendingSamples[:config.BatchSize]...)
			if err := applyBatch(state, batch, config); err != nil {
				return Result{}, err
			}
			state.PendingSamples = append([]jamiethompsonmev1alpha1.OnlineLinearSample(nil), state.PendingSamples[config.BatchSize:]...)
			updated = true
		}
	} else {
		if err := applyBatch(state, []jamiethompsonmev1alpha1.OnlineLinearSample{sample}, config); err != nil {
			return Result{}, err
		}
		updated = true
	}
	if updated {
		state.LastUpdateTime = observation.Time.DeepCopy()
	}

	ready := state.TrainedSamples >= config.WarmupSamples
	var prediction *int32
	if ready && shouldPredict {
		value, err := predict(state, observation.Time.Add(config.LookAhead), config.LookAhead)
		if err != nil {
			return Result{}, err
		}
		prediction = &value
		state.LastPrediction = int32Ptr(value)
		state.PendingForecasts = append(state.PendingForecasts, jamiethompsonmev1alpha1.OnlineLinearForecast{
			TargetTime: &metav1.Time{Time: observation.Time.Add(config.LookAhead)},
			Replicas:   value,
		})
		if len(state.PendingForecasts) > maximumPendingForecasts {
			state.PendingForecasts = append([]jamiethompsonmev1alpha1.OnlineLinearForecast(nil),
				state.PendingForecasts[len(state.PendingForecasts)-maximumPendingForecasts:]...)
		}
	}

	history.OnlineLinearState = state
	var meanAbsoluteError *float64
	if state.EvaluatedForecasts > 0 {
		value := state.AbsoluteErrorSum / float64(state.EvaluatedForecasts)
		meanAbsoluteError = &value
	}

	return Result{
		History:           history,
		Prediction:        prediction,
		Ready:             ready,
		Updated:           updated,
		Reset:             reset,
		ForceCheckpoint:   reset,
		MeanAbsoluteError: meanAbsoluteError,
	}, nil
}

func applyBatch(
	state *jamiethompsonmev1alpha1.OnlineLinearState,
	samples []jamiethompsonmev1alpha1.OnlineLinearSample,
	config Config,
) error {
	if len(samples) == 0 {
		return nil
	}
	oldIntercept := state.Intercept
	oldCoefficient := state.Coefficient
	if state.TrainedSamples == 0 {
		state.Intercept = float64(samples[0].Replicas)
	}

	var interceptGradient float64
	var coefficientGradient float64
	for _, sample := range samples {
		if sample.Time == nil {
			return errors.New("OnlineLinear minibatch sample has no timestamp")
		}
		x := normalizedTime(state, sample.Time.Time, config.LookAhead)
		errorValue := float64(sample.Replicas) - (state.Intercept + state.Coefficient*x)
		interceptGradient += errorValue
		coefficientGradient += errorValue * x
	}
	interceptGradient /= float64(len(samples))
	coefficientGradient /= float64(len(samples))
	interceptGradient, coefficientGradient = clipGradient(interceptGradient, coefficientGradient)

	state.Intercept += config.LearningRate * interceptGradient
	state.Coefficient += config.LearningRate * coefficientGradient
	if !finite(state.Intercept) || !finite(state.Coefficient) {
		state.Intercept = oldIntercept
		state.Coefficient = oldCoefficient
		return errors.New("OnlineLinear SGD update produced non-finite parameters")
	}
	state.TrainedSamples += int64(len(samples))
	state.UpdatesApplied++
	return nil
}

func evaluateForecasts(
	state *jamiethompsonmev1alpha1.OnlineLinearState,
	observation jamiethompsonmev1alpha1.TimestampedReplicas,
) {
	remaining := make([]jamiethompsonmev1alpha1.OnlineLinearForecast, 0, len(state.PendingForecasts))
	for _, forecast := range state.PendingForecasts {
		if forecast.TargetTime == nil || forecast.TargetTime.After(observation.Time.Time) {
			remaining = append(remaining, forecast)
			continue
		}
		state.AbsoluteErrorSum += math.Abs(float64(forecast.Replicas - observation.Replicas))
		state.EvaluatedForecasts++
	}
	state.PendingForecasts = remaining
}

func rebaseIfNeeded(
	state *jamiethompsonmev1alpha1.OnlineLinearState,
	now time.Time,
	lookAhead time.Duration,
) {
	x := normalizedTime(state, now, lookAhead)
	if math.Abs(x) <= maximumNormalizedTime {
		return
	}
	state.Intercept += state.Coefficient * x
	state.OriginTime = &metav1.Time{Time: now}
}

func predict(
	state *jamiethompsonmev1alpha1.OnlineLinearState,
	target time.Time,
	lookAhead time.Duration,
) (int32, error) {
	x := normalizedTime(state, target, lookAhead)
	value := state.Intercept + state.Coefficient*x
	if !finite(value) {
		return 0, errors.New("OnlineLinear prediction is not finite")
	}
	value = math.Max(0, math.Ceil(value))
	if value > math.MaxInt32 {
		return 0, fmt.Errorf("OnlineLinear prediction %.0f exceeds int32", value)
	}
	return int32(value), nil
}

func normalizedTime(
	state *jamiethompsonmev1alpha1.OnlineLinearState,
	timestamp time.Time,
	lookAhead time.Duration,
) float64 {
	return float64(timestamp.Sub(state.OriginTime.Time)) / float64(lookAhead)
}

func clipGradient(intercept, coefficient float64) (float64, float64) {
	norm := math.Hypot(intercept, coefficient)
	if norm <= maximumGradientNorm || norm == 0 {
		return intercept, coefficient
	}
	scale := maximumGradientNorm / norm
	return intercept * scale, coefficient * scale
}

func validState(state *jamiethompsonmev1alpha1.OnlineLinearState) bool {
	return state != nil && finite(state.Intercept) && finite(state.Coefficient) &&
		finite(state.AbsoluteErrorSum) && state.SamplesSeen >= 0 && state.TrainedSamples >= 0 &&
		state.UpdatesApplied >= 0 && state.EvaluatedForecasts >= 0
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func int32Ptr(value int32) *int32 {
	return &value
}
