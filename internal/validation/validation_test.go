/*
Copyright 2026 The Predictive Horizontal Pod Autoscaler Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package validation

import (
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	jamiethompsonmev1alpha1 "github.com/jthomperoo/predictive-horizontal-pod-autoscaler/api/v1alpha1"
)

func TestValidateOnlineLinearCrossFieldRules(t *testing.T) {
	datapoint := jamiethompsonmev1alpha1.OnlineUpdateDatapoint
	minibatch := jamiethompsonmev1alpha1.OnlineUpdateMinibatch
	one := 1
	two := 2
	zero := 0.0
	oneWarmup := int64(1)
	invalidMode := "invalid"
	invalidUpdateMode := "invalid"
	shortCheckpoint := metav1.Duration{Duration: 5 * time.Second}

	tests := []struct {
		name        string
		model       jamiethompsonmev1alpha1.Model
		wantErrPart string
	}{
		{
			name:  "valid defaults",
			model: onlineValidationModel(&datapoint, &one, nil, nil),
		},
		{
			name:        "datapoint rejects batch",
			model:       onlineValidationModel(&datapoint, &two, nil, nil),
			wantErrPart: "batchSize must be 1",
		},
		{
			name:        "minibatch rejects one",
			model:       onlineValidationModel(&minibatch, &one, nil, nil),
			wantErrPart: "batchSize must be at least 2",
		},
		{
			name:        "rejects zero learning rate",
			model:       onlineValidationModel(&datapoint, &one, &zero, nil),
			wantErrPart: "learningRate",
		},
		{
			name:        "rejects checkpoint shorter than sync",
			model:       onlineValidationModel(&datapoint, &one, nil, &shortCheckpoint),
			wantErrPart: "checkpointInterval",
		},
		{
			name:        "rejects unsupported update mode",
			model:       onlineValidationModel(&invalidUpdateMode, &one, nil, nil),
			wantErrPart: "updateMode",
		},
		{
			name: "rejects warmup below two",
			model: func() jamiethompsonmev1alpha1.Model {
				model := onlineValidationModel(&datapoint, &one, nil, nil)
				model.OnlineLinear.WarmupSamples = &oneWarmup
				return model
			}(),
			wantErrPart: "warmupSamples",
		},
		{
			name: "rejects unsupported mode",
			model: func() jamiethompsonmev1alpha1.Model {
				model := onlineValidationModel(&datapoint, &one, nil, nil)
				model.OnlineLinear.Mode = &invalidMode
				return model
			}(),
			wantErrPart: "mode",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			instance := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscaler{}
			instance.Spec.MaxReplicas = 10
			instance.Spec.Models = []jamiethompsonmev1alpha1.Model{test.model}
			err := Validate(instance)
			if test.wantErrPart == "" && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if test.wantErrPart != "" && (err == nil || !strings.Contains(err.Error(), test.wantErrPart)) {
				t.Fatalf("got error %v, want containing %q", err, test.wantErrPart)
			}
		})
	}
}

func onlineValidationModel(
	updateMode *string,
	batchSize *int,
	learningRate *float64,
	checkpoint *metav1.Duration,
) jamiethompsonmev1alpha1.Model {
	return jamiethompsonmev1alpha1.Model{
		Type: jamiethompsonmev1alpha1.TypeOnlineLinear,
		Name: "online",
		OnlineLinear: &jamiethompsonmev1alpha1.OnlineLinear{
			LookAhead:          10000,
			UpdateMode:         updateMode,
			BatchSize:          batchSize,
			LearningRate:       learningRate,
			CheckpointInterval: checkpoint,
		},
	}
}
