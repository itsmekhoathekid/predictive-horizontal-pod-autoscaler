/*
Copyright 2026 The Predictive Horizontal Pod Autoscaler Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

// Package observability owns metrics for incremental model operation.
package observability

import (
	"github.com/prometheus/client_golang/prometheus"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

var (
	// OnlineSamples counts observations ingested by OnlineLinear models.
	OnlineSamples = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "phpa_online_linear_samples_total",
		Help: "Number of observations ingested by an OnlineLinear model.",
	}, []string{"namespace", "phpa", "model"})
	// OnlineUpdates counts SGD updates applied by OnlineLinear models.
	OnlineUpdates = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "phpa_online_linear_updates_total",
		Help: "Number of SGD updates applied by an OnlineLinear model.",
	}, []string{"namespace", "phpa", "model"})
	// OnlinePrediction exposes the most recent predicted replica count.
	OnlinePrediction = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "phpa_online_linear_prediction",
		Help: "Most recent OnlineLinear replica prediction.",
	}, []string{"namespace", "phpa", "model"})
	// OnlineMAE exposes prequential mean absolute error.
	OnlineMAE = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "phpa_online_linear_mean_absolute_error",
		Help: "Prequential mean absolute error of due OnlineLinear forecasts.",
	}, []string{"namespace", "phpa", "model"})
	// OnlineFallbacks counts OnlineLinear failures that fell back to reactive HPA output.
	OnlineFallbacks = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "phpa_online_linear_fallbacks_total",
		Help: "Number of OnlineLinear processing failures that used reactive scaling only.",
	}, []string{"namespace", "phpa", "model"})
	// OnlineResets counts model state resets.
	OnlineResets = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "phpa_online_linear_resets_total",
		Help: "Number of OnlineLinear state resets.",
	}, []string{"namespace", "phpa", "model"})
	// StateCheckpoints counts durable ConfigMap checkpoints.
	StateCheckpoints = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "phpa_model_state_checkpoints_total",
		Help: "Number of successful PHPA model state checkpoints.",
	}, []string{"namespace", "phpa"})
)

func init() {
	ctrlmetrics.Registry.MustRegister(
		OnlineSamples,
		OnlineUpdates,
		OnlinePrediction,
		OnlineMAE,
		OnlineFallbacks,
		OnlineResets,
		StateCheckpoints,
	)
}
