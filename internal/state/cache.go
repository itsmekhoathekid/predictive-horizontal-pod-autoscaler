/*
Copyright 2026 The Predictive Horizontal Pod Autoscaler Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

// Package state provides the write-back cache used for PHPA model state.
package state

import (
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	jamiethompsonmev1alpha1 "github.com/jthomperoo/predictive-horizontal-pod-autoscaler/api/v1alpha1"
)

// SchemaVersion is the current ConfigMap state format.
const SchemaVersion = 2

type entry struct {
	data           *jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData
	lastCheckpoint time.Time
}

// Cache stores dirty state between durable ConfigMap checkpoints.
type Cache struct {
	mutex   sync.RWMutex
	entries map[types.NamespacedName]entry
}

// NewCache creates an empty state cache.
func NewCache() *Cache {
	return &Cache{entries: map[types.NamespacedName]entry{}}
}

// Load returns cached state or seeds the cache from durable state.
func (c *Cache) Load(
	key types.NamespacedName,
	persisted *jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData,
) (*jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData, bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if cached, exists := c.entries[key]; exists {
		return cached.data.DeepCopy(), false
	}

	data := persisted.DeepCopy()
	forceCheckpoint := data.SchemaVersion != SchemaVersion
	if data.SchemaVersion > SchemaVersion {
		data = &jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData{
			SchemaVersion:  SchemaVersion,
			ModelHistories: map[string]jamiethompsonmev1alpha1.ModelHistory{},
		}
	}
	data.SchemaVersion = SchemaVersion
	if data.ModelHistories == nil {
		data.ModelHistories = map[string]jamiethompsonmev1alpha1.ModelHistory{}
	}
	lastCheckpoint := time.Time{}
	if data.LastCheckpointTime != nil {
		lastCheckpoint = data.LastCheckpointTime.Time
	}
	c.entries[key] = entry{data: data.DeepCopy(), lastCheckpoint: lastCheckpoint}
	return data, forceCheckpoint
}

// Put replaces cached state without marking it durable.
func (c *Cache) Put(
	key types.NamespacedName,
	data *jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData,
) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	cached := c.entries[key]
	cached.data = data.DeepCopy()
	c.entries[key] = cached
}

// CheckpointDue reports whether the cache must be persisted.
func (c *Cache) CheckpointDue(
	key types.NamespacedName,
	now time.Time,
	interval time.Duration,
	force bool,
) bool {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	if force {
		return true
	}
	cached, exists := c.entries[key]
	return !exists || cached.lastCheckpoint.IsZero() || !now.Before(cached.lastCheckpoint.Add(interval))
}

// MarkCheckpoint records a successful durable write and returns state with its timestamp updated.
func (c *Cache) MarkCheckpoint(
	key types.NamespacedName,
	data *jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData,
	now time.Time,
) *jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	checkpointed := data.DeepCopy()
	checkpointed.SchemaVersion = SchemaVersion
	checkpointed.LastCheckpointTime = &metav1.Time{Time: now}
	c.entries[key] = entry{data: checkpointed.DeepCopy(), lastCheckpoint: now}
	return checkpointed
}

// Delete removes cached state for a deleted PHPA.
func (c *Cache) Delete(key types.NamespacedName) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	delete(c.entries, key)
}

// Snapshot returns deep copies of all cached PHPA state for a best-effort shutdown flush.
func (c *Cache) Snapshot() map[types.NamespacedName]*jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	result := make(map[types.NamespacedName]*jamiethompsonmev1alpha1.PredictiveHorizontalPodAutoscalerData, len(c.entries))
	for key, cached := range c.entries {
		result[key] = cached.data.DeepCopy()
	}
	return result
}
