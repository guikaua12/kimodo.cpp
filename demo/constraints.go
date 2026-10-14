package main

// Structural validation of upstream Kimodo constraints.json documents posted
// with /api/generate. The native engine performs the full semantic validation
// (FK, joint conversion, merge rules); the server only rejects documents whose
// shape could never be valid so the user gets an immediate 400 instead of a
// failed animation.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// constraintTypes lists the seven upstream TYPE_TO_CLASS names in the order
// used by error messages.
var constraintTypes = []string{"root2d", "fullbody", "left-hand", "right-hand", "left-foot", "right-foot", "end-effector"}

// endEffectorJointNames are the base names a generic end-effector constraint
// may list (upstream skeleton/base.py expand_joint_names).
var endEffectorJointNames = []string{"LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"}

// soma77Joints is the joint count of upstream SOMA constraint files, which a
// SOMA 30 control skeleton accepts besides its own 30 joints.
const soma77Joints = 77

type constraintSummary struct {
	Sets  int      // constraint objects in the document
	Count int      // constrained frames over all sets (sum of len(frame_indices))
	Types []string // distinct types in first-occurrence order
}

// validateConstraints checks a constraints array against the model skeleton
// and the total number of generated frames. An absent, null or empty array is
// valid and means "no constraints".
func validateConstraints(raw json.RawMessage, skeleton skeletonDefinition, totalFrames int) (constraintSummary, error) {
	summary := constraintSummary{Types: []string{}}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return summary, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	decoder.UseNumber()
	var document any
	if err := decoder.Decode(&document); err != nil {
		return summary, fmt.Errorf("constraints must be valid JSON: %v", err)
	}
	sets, ok := document.([]any)
	if !ok {
		return summary, fmt.Errorf("constraints must be a JSON array of constraint objects")
	}
	joints := len(skeleton.parents)
	for index, element := range sets {
		set, ok := element.(map[string]any)
		if !ok {
			return summary, fmt.Errorf("constraints[%d] must be an object", index)
		}
		kind, _ := set["type"].(string)
		if _, present := set["type"]; !present {
			return summary, fmt.Errorf("constraints[%d]: missing type (expected one of %s)", index, strings.Join(constraintTypes, ", "))
		}
		if !slices.Contains(constraintTypes, kind) {
			return summary, fmt.Errorf("constraints[%d]: unknown type %s (expected one of %s)", index, describeJSON(set["type"]), strings.Join(constraintTypes, ", "))
		}
		where := fmt.Sprintf("constraints[%d] (%s)", index, kind)
		frames, err := frameIndices(set["frame_indices"], totalFrames)
		if err != nil {
			return summary, fmt.Errorf("%s: %v", where, err)
		}
		// requireShape checks a mandatory (or, when optional, a present and
		// non-null) number array against the expected dimensions; -1 in want
		// stands for the joint count, which may be J or (SOMA only) 77.
		requireShape := func(key string, optional bool, want ...int) error {
			value, present := set[key]
			if !present || value == nil {
				if optional {
					return nil
				}
				return fmt.Errorf("%s: missing %s", where, key)
			}
			shape, ok := numericShape(value, len(want))
			matches := ok
			for axis := 0; matches && axis < len(want); axis++ {
				expected := want[axis]
				if expected == -1 {
					matches = shape[axis] == joints || (skeleton.key == "soma30" && shape[axis] == soma77Joints)
				} else {
					matches = shape[axis] == expected
				}
			}
			if matches {
				return nil
			}
			dims := make([]string, len(want))
			for axis, expected := range want {
				dims[axis] = strconv.Itoa(expected)
				if axis == 0 {
					dims[axis] = "K"
				} else if expected == -1 {
					dims[axis] = strconv.Itoa(joints)
					if skeleton.key == "soma30" {
						dims[axis] += " or 77"
					}
				}
			}
			got := "not a rectangular number array"
			if ok {
				got = fmt.Sprint(shape)
			}
			return fmt.Errorf("%s: %s must be a [%s] number array with K = len(frame_indices) = %d, got %s", where, key, strings.Join(dims, "]["), frames, got)
		}
		if kind == "root2d" {
			if err := requireShape("smooth_root_2d", false, frames, 2); err != nil {
				return summary, err
			}
			if err := requireShape("global_root_heading", true, frames, 2); err != nil {
				return summary, err
			}
		} else {
			if err := requireShape("local_joints_rot", false, frames, -1, 3); err != nil {
				return summary, err
			}
			if err := requireShape("root_positions", false, frames, 3); err != nil {
				return summary, err
			}
			if err := requireShape("smooth_root_2d", true, frames, 2); err != nil {
				return summary, err
			}
		}
		if kind == "end-effector" {
			names, ok := set["joint_names"].([]any)
			if !ok || len(names) == 0 {
				return summary, fmt.Errorf("%s: joint_names must be a non-empty array of %s", where, strings.Join(endEffectorJointNames, ", "))
			}
			for _, name := range names {
				if text, ok := name.(string); !ok || !slices.Contains(endEffectorJointNames, text) {
					return summary, fmt.Errorf("%s: unknown joint name %s (expected %s)", where, describeJSON(name), strings.Join(endEffectorJointNames, ", "))
				}
			}
		}
		summary.Sets++
		summary.Count += frames
		if !slices.Contains(summary.Types, kind) {
			summary.Types = append(summary.Types, kind)
		}
	}
	return summary, nil
}

// frameIndices validates a non-empty array of integer frame indices within
// [0,totalFrames) and returns its length. Like the engine, integral floats
// (3.0, 1e2, e.g. from a NumPy float array) count as integers. Upstream keeps
// the order as given (not necessarily sorted) and tolerates repeats, so
// neither is checked.
func frameIndices(value any, totalFrames int) (int, error) {
	items, ok := value.([]any)
	if !ok {
		return 0, fmt.Errorf("frame_indices must be an array of integers")
	}
	if len(items) == 0 {
		return 0, fmt.Errorf("frame_indices must not be empty")
	}
	for position, item := range items {
		number, ok := item.(json.Number)
		if !ok {
			return 0, fmt.Errorf("frame_indices[%d] must be an integer, got %s", position, describeJSON(item))
		}
		frame, err := number.Float64()
		if err != nil || math.IsNaN(frame) || math.IsInf(frame, 0) || frame != math.Trunc(frame) {
			return 0, fmt.Errorf("frame_indices[%d] must be an integer, got %s", position, number)
		}
		if frame < 0 || frame >= float64(totalFrames) {
			return 0, fmt.Errorf("frame_indices[%d] = %s is outside the %d generated frames [0,%d)", position, number, totalFrames, totalFrames)
		}
	}
	return len(items), nil
}

// numericShape returns the dimensions of value when it is a rectangular nested
// array of JSON numbers with exactly rank levels (decoded with UseNumber).
func numericShape(value any, rank int) ([]int, bool) {
	if rank == 0 {
		// Out-of-range literals such as 1e400 are rejected like non-numbers.
		number, ok := value.(json.Number)
		if !ok {
			return nil, false
		}
		parsed, err := number.Float64()
		return nil, err == nil && !math.IsInf(parsed, 0)
	}
	items, ok := value.([]any)
	if !ok {
		return nil, false
	}
	inner := make([]int, rank-1)
	for index, item := range items {
		shape, ok := numericShape(item, rank-1)
		if !ok {
			return nil, false
		}
		if index == 0 {
			inner = shape
		} else if !slices.Equal(inner, shape) {
			return nil, false
		}
	}
	return append([]int{len(items)}, inner...), true
}

// describeJSON renders a decoded JSON value for error messages, truncated so a
// large nested array cannot flood the response.
func describeJSON(value any) string {
	if value == nil {
		return "null"
	}
	b, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf("%T", value)
	}
	if len(b) > 40 {
		return string(b[:40]) + "..."
	}
	return string(b)
}
