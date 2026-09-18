/*
Copyright 2026 The Predictive Horizontal Pod Autoscaler Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package controllers

import (
	"context"
	"errors"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	jamiethompsonmev1alpha1 "github.com/jthomperoo/predictive-horizontal-pod-autoscaler/api/v1alpha1"
	"github.com/jthomperoo/predictive-horizontal-pod-autoscaler/internal/prediction/onlinelinear"
)

type failingOnlineProcessor struct{}

type conflictOnceClient struct {
	client.Client
	conflicts int
}

func (c *conflictOnceClient) Update(
	ctx context.Context,
	obj client.Object,
	opts ...client.UpdateOption,
) error {
	if _, ok := obj.(*corev1.ConfigMap); ok && c.conflicts == 0 {
		c.conflicts++
		return k8serrors.NewConflict(schema.GroupResource{Resource: "configmaps"}, obj.GetName(),
			errors.New("simulated conflict"))
	}
	return c.Client.Update(ctx, obj, opts...)
}

func (failingOnlineProcessor) Process(
	_ *jamiethompsonmev1alpha1.Model,
	_ jamiethompsonmev1alpha1.ModelHistory,
	_ jamiethompsonmev1alpha1.TimestampedReplicas,
	_ bool,
) (onlinelinear.Result, error) {
	return onlinelinear.Result{}, errors.New("training failed")
}

func TestProcessModelsOnlineLinearActiveAndObserve(t *testing.T) {
	tests := []struct {
		name                string
		mode                string
		expectedPredictions int
		expectedReason      string
	}{
		{
			name:                "active contributes prediction",
			mode:                jamiethompsonmev1alpha1.OnlineModeActive,
			expectedPredictions: 2,
			expectedReason:      "Ready",
		},
		{
			name:                "observe records without contributing",
			mode:                jamiethompsonmev1alpha1.OnlineModeObserve,
			expectedPredictions: 1,
			expectedReason:      "ObserveOnly",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			updateMode := jamiethompsonmev1alpha1.OnlineUpdateDatapoint
			batchSize := 1
			learningRate := 0.5
			warmup := int64(2)
			instance := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscaler{}
			instance.Namespace = "test"
			instance.Name = "phpa"
			instance.Spec.Models = []jamiethompsonmev1alpha1.Model{{
				Type: jamiethompsonmev1alpha1.TypeOnlineLinear,
				Name: "online",
				OnlineLinear: &jamiethompsonmev1alpha1.OnlineLinear{
					LookAhead:     10000,
					UpdateMode:    &updateMode,
					BatchSize:     &batchSize,
					LearningRate:  &learningRate,
					WarmupSamples: &warmup,
					Mode:          &test.mode,
				},
			}}
			reconciler := &PredictiveHorizontalPodAutoscalerReconciler{
				OnlineLinearPredicter: &onlinelinear.Predict{},
			}
			data := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData{
				ModelHistories: map[string]jamiethompsonmev1alpha1.ModelHistory{},
			}
			start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

			_, data, _, force, interval := reconciler.processModels(context.Background(), instance, data, start, 1, 1)
			if !force || interval != time.Minute {
				t.Fatalf("unexpected checkpoint decision: force=%t interval=%s", force, interval)
			}
			predictions, _, statuses, _, _ := reconciler.processModels(
				context.Background(), instance, data, start.Add(10*time.Second), 1, 2)
			if len(predictions) != test.expectedPredictions {
				t.Fatalf("got %d predictions, want %d: %v", len(predictions), test.expectedPredictions, predictions)
			}
			if len(statuses) != 1 || statuses[0].Reason != test.expectedReason || !statuses[0].Ready {
				t.Fatalf("unexpected status: %+v", statuses)
			}
		})
	}
}

func TestProcessModelsOnlineLinearFailureFallsBackToReactive(t *testing.T) {
	instance := onlineControllerInstance(jamiethompsonmev1alpha1.OnlineModeActive, 1)
	reconciler := &PredictiveHorizontalPodAutoscalerReconciler{
		OnlineLinearPredicter: failingOnlineProcessor{},
	}
	data := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData{
		ModelHistories: map[string]jamiethompsonmev1alpha1.ModelHistory{},
	}

	predictions, _, statuses, _, _ := reconciler.processModels(
		context.Background(), instance, data, time.Now().UTC(), 2, 4)
	if len(predictions) != 1 || predictions[0] != 4 {
		t.Fatalf("model failure did not retain reactive target: %v", predictions)
	}
	if len(statuses) != 1 || statuses[0].Reason != "ProcessingError" {
		t.Fatalf("model failure status was not reported: %+v", statuses)
	}
}

func TestUpdateConfigMapDataRetriesConflict(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "state", Namespace: "default"},
		Data:       map[string]string{configMapDataKey: `{}`},
	}
	baseClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(configMap).Build()
	conflictingClient := &conflictOnceClient{Client: baseClient}
	reconciler := &PredictiveHorizontalPodAutoscalerReconciler{Client: conflictingClient}
	data := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData{
		SchemaVersion:  2,
		ModelHistories: map[string]jamiethompsonmev1alpha1.ModelHistory{},
	}

	if err := reconciler.updateConfigMapData(context.Background(), configMap, data); err != nil {
		t.Fatal(err)
	}
	if conflictingClient.conflicts != 1 {
		t.Fatalf("expected exactly one simulated conflict, got %d", conflictingClient.conflicts)
	}
	updated := &corev1.ConfigMap{}
	if err := baseClient.Get(context.Background(), client.ObjectKeyFromObject(configMap), updated); err != nil {
		t.Fatal(err)
	}
	if updated.Data[configMapDataKey] != `{"schemaVersion":2,"modelHistories":{}}` {
		t.Fatalf("unexpected checkpoint data: %s", updated.Data[configMapDataKey])
	}
}

func TestProcessModelsOnlineLinearTrainsEverySyncButPredictsAtConfiguredPeriod(t *testing.T) {
	instance := onlineControllerInstance(jamiethompsonmev1alpha1.OnlineModeActive, 3)
	reconciler := &PredictiveHorizontalPodAutoscalerReconciler{
		OnlineLinearPredicter: &onlinelinear.Predict{},
	}
	data := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData{
		ModelHistories: map[string]jamiethompsonmev1alpha1.ModelHistory{},
	}
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	for i := 0; i < 3; i++ {
		predictions, updated, _, _, _ := reconciler.processModels(
			context.Background(), instance, data, start.Add(time.Duration(i)*10*time.Second), 1, int32(i+1))
		data = updated
		if i < 2 && len(predictions) != 1 {
			t.Fatalf("prediction emitted too early on sync %d: %v", i+1, predictions)
		}
		if i == 2 && len(predictions) != 2 {
			t.Fatalf("prediction missing at configured period: %v", predictions)
		}
	}
	state := data.ModelHistories["online"].OnlineLinearState
	if state == nil || state.SamplesSeen != 3 || state.UpdatesApplied != 3 {
		t.Fatalf("model did not train on every sync: %+v", state)
	}
}

func onlineControllerInstance(mode string, perSyncPeriod int) *jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscaler {
	updateMode := jamiethompsonmev1alpha1.OnlineUpdateDatapoint
	batchSize := 1
	learningRate := 0.5
	warmup := int64(2)
	instance := &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscaler{}
	instance.Namespace = "test"
	instance.Name = "phpa"
	instance.Spec.Models = []jamiethompsonmev1alpha1.Model{{
		Type:          jamiethompsonmev1alpha1.TypeOnlineLinear,
		Name:          "online",
		PerSyncPeriod: &perSyncPeriod,
		OnlineLinear: &jamiethompsonmev1alpha1.OnlineLinear{
			LookAhead:     10000,
			UpdateMode:    &updateMode,
			BatchSize:     &batchSize,
			LearningRate:  &learningRate,
			WarmupSamples: &warmup,
			Mode:          &mode,
		},
	}}
	return instance
}
