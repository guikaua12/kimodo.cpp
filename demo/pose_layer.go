package main

// Pose presets (demo/editor/poses.js) ride along a generate request as
// `pose`: the editor-side choice (id, label, params) and an optional layer
// the server applies to the generated rotations once the engine returns.
//
// The layer exists because some poses cannot be asked of the model: Kimodo
// never generates a muzzle-forward rifle low ready, and pinning the hands
// there makes it lean and bow its head (docs/motion-stage-research/
// 09-experiment-log.md, "Pose presets"). So the editor pins Kimodo's own
// carry and the layer moves the arms onto the authored one afterwards. The engine's raw output is kept as
// local_rotations_xyzw.model.f32 and the posed rotations are always derived
// from it, so applying a layer is idempotent.
//
// Mode "hold", per listed joint j over the T frames:
//   mean_j  = normalize(sum_t s_t q_t)  (s_t = sign(q_t . q_0))
//   r       = inv(mean_j) * q_t, its angle clamped to max_sway_deg and scaled by sway
//   out     = slerp(q_t, target_j * r, strength)
// i.e. the clip-mean local rotation becomes the authored one and the model's
// own sway around its mean stays on top. applyPoseLayer in poses.js is the
// JavaScript twin (editor previews); both are checked against
// editor/tests/fixtures/pose_layer.json.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
)

const (
	poseFile          = "pose.json"
	modelRotationFile = "local_rotations_xyzw.model.f32"
	defaultMaxSwayDeg = 15.0
)

var poseIDPattern = regexp.MustCompile(`^[a-z0-9_]{1,64}$`)

// poseRequest is the optional /api/generate `pose` object.
type poseRequest struct {
	ID     string          `json:"id"`
	Label  string          `json:"label,omitempty"`
	Params json.RawMessage `json:"params,omitempty"` // editor parameters, echoed back for restore
	Layer  *poseLayer      `json:"layer,omitempty"`
}

type poseLayer struct {
	Mode       string     `json:"mode"`
	Joints     []string   `json:"joints"`
	Rotations  []quatJSON `json:"rotations"` // xyzw targets, one per joint
	Strength   *float64   `json:"strength,omitempty"`
	Sway       *float64   `json:"sway,omitempty"`
	MaxSwayDeg *float64   `json:"max_sway_deg,omitempty"`
}

// quatJSON is an xyzw quaternion that must be exactly four numbers in JSON
// (a plain [4]float64 would zero-fill a short array, e.g. an axis-angle triple).
type quatJSON [4]float64

func (q *quatJSON) UnmarshalJSON(b []byte) error {
	var values []float64
	if err := json.Unmarshal(b, &values); err != nil {
		return fmt.Errorf("pose layer rotation must be an array of 4 numbers (xyzw)")
	}
	if len(values) != 4 {
		return fmt.Errorf("pose layer rotation must have 4 components (xyzw), got %d", len(values))
	}
	copy(q[:], values)
	return nil
}

// poseSummary is what a gallery record keeps of the pose (the layer itself
// is in <id>/pose.json).
type poseSummary struct {
	ID      string          `json:"id"`
	Label   string          `json:"label,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Layered bool            `json:"layered,omitempty"`
}

func (p *poseRequest) summary() *poseSummary {
	return &poseSummary{ID: p.ID, Label: p.Label, Params: p.Params, Layered: p.Layer != nil}
}

// validate checks the pose against the request's skeleton.
func (p *poseRequest) validate(names []string) error {
	if !poseIDPattern.MatchString(p.ID) {
		return fmt.Errorf("pose id must be 1..64 characters of a-z, 0-9 and _")
	}
	if len(p.Label) > 200 {
		return fmt.Errorf("pose label is longer than 200 bytes")
	}
	if len(p.Params) > 4096 {
		return fmt.Errorf("pose params exceed 4 KiB")
	}
	if len(p.Params) > 0 {
		var params map[string]any
		if err := json.Unmarshal(p.Params, &params); err != nil {
			return fmt.Errorf("pose params must be a JSON object")
		}
	}
	layer := p.Layer
	if layer == nil {
		return nil
	}
	if layer.Mode != "hold" {
		return fmt.Errorf("unknown pose layer mode %q (expected hold)", layer.Mode)
	}
	if len(layer.Joints) == 0 || len(layer.Joints) != len(layer.Rotations) {
		return fmt.Errorf("pose layer needs one rotation per joint")
	}
	seen := map[string]bool{}
	for index, name := range layer.Joints {
		if seen[name] {
			return fmt.Errorf("pose layer joint %q is listed twice", name)
		}
		seen[name] = true
		if jointIndexOf(names, name) < 0 {
			return fmt.Errorf("pose layer joint %q is not in this skeleton", name)
		}
		q := layer.Rotations[index]
		norm := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
		if math.IsNaN(norm) || math.IsInf(norm, 0) || norm < 1e-6 {
			return fmt.Errorf("pose layer rotation %d is not a usable quaternion", index)
		}
	}
	unit := func(value *float64, what string) error {
		if value != nil && (math.IsNaN(*value) || *value < 0 || *value > 1) {
			return fmt.Errorf("pose layer %s must be 0..1", what)
		}
		return nil
	}
	if err := unit(layer.Strength, "strength"); err != nil {
		return err
	}
	if err := unit(layer.Sway, "sway"); err != nil {
		return err
	}
	if layer.MaxSwayDeg != nil && (math.IsNaN(*layer.MaxSwayDeg) || *layer.MaxSwayDeg < 0 || *layer.MaxSwayDeg > 180) {
		return fmt.Errorf("pose layer max_sway_deg must be 0..180")
	}
	return nil
}

func jointIndexOf(names []string, name string) int {
	for index, candidate := range names {
		if candidate == name {
			return index
		}
	}
	return -1
}

// ---------------------------------------------------------------- quaternions (xyzw, float64)

type quat [4]float64

func (a quat) mul(b quat) quat {
	return quat{
		a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
		a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
		a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
		a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
	}
}

func (a quat) dot(b quat) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] }

// normalized returns q / |q|; a zero or non-finite q becomes the identity.
func (a quat) normalized() quat {
	n := math.Sqrt(a.dot(a))
	if !(n > 0) || math.IsInf(n, 0) {
		return quat{0, 0, 0, 1}
	}
	return quat{a[0] / n, a[1] / n, a[2] / n, a[3] / n}
}

// inverse of a unit quaternion.
func (a quat) inverse() quat { return quat{-a[0], -a[1], -a[2], a[3]} }

// slerp along the shortest arc (math.js quatSlerp: normalised lerp when nearly parallel).
func slerp(a, b quat, t float64) quat {
	cos := a.dot(b)
	if cos < 0 {
		cos = -cos
		b = quat{-b[0], -b[1], -b[2], -b[3]}
	}
	ka, kb := 1-t, t
	if cos <= 0.9995 {
		theta := math.Acos(math.Min(1, cos))
		sin := math.Sin(theta)
		ka = math.Sin((1-t)*theta) / sin
		kb = math.Sin(t*theta) / sin
	}
	return quat{a[0]*ka + b[0]*kb, a[1]*ka + b[1]*kb, a[2]*ka + b[2]*kb, a[3]*ka + b[3]*kb}.normalized()
}

// clampScale returns q with its rotation angle min(angle, maxRad) * scale
// (canonical w >= 0 form).
func clampScale(q quat, maxRad, scale float64) quat {
	if q[3] < 0 {
		q = quat{-q[0], -q[1], -q[2], -q[3]}
	}
	s := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	if s < 1e-12 {
		return quat{0, 0, 0, 1}
	}
	angle := math.Min(2*math.Atan2(s, q[3]), maxRad) * scale
	k := math.Sin(angle/2) / s
	return quat{q[0] * k, q[1] * k, q[2] * k, math.Cos(angle / 2)}
}

// applyPoseLayer returns the [T][J] xyzw rotations with the layer applied.
func applyPoseLayer(rotations []float32, names []string, layer *poseLayer) ([]float32, error) {
	joints := len(names)
	if joints == 0 || len(rotations)%(joints*4) != 0 {
		return nil, fmt.Errorf("rotations do not hold %d-joint frames", joints)
	}
	out := append([]float32(nil), rotations...)
	frames := len(rotations) / (joints * 4)
	if layer == nil || frames == 0 {
		return out, nil
	}
	if layer.Mode != "hold" {
		return nil, fmt.Errorf("unknown pose layer mode %q", layer.Mode)
	}
	clamp01 := func(value *float64) float64 {
		if value == nil {
			return 1
		}
		return math.Min(1, math.Max(0, *value))
	}
	strength, sway := clamp01(layer.Strength), clamp01(layer.Sway)
	maxRad := defaultMaxSwayDeg * math.Pi / 180
	if layer.MaxSwayDeg != nil {
		maxRad = math.Max(0, *layer.MaxSwayDeg) * math.Pi / 180
	}
	at := func(frame, joint int) quat {
		i := (frame*joints + joint) * 4
		return quat{float64(rotations[i]), float64(rotations[i+1]), float64(rotations[i+2]), float64(rotations[i+3])}
	}
	for k, name := range layer.Joints {
		j := jointIndexOf(names, name)
		if j < 0 {
			return nil, fmt.Errorf("pose layer joint %q is not in this skeleton", name)
		}
		target := quat(layer.Rotations[k]).normalized()
		q0 := at(0, j)
		var sum quat
		for t := 0; t < frames; t++ {
			q := at(t, j)
			sign := 1.0
			if q.dot(q0) < 0 {
				sign = -1
			}
			for c := range sum {
				sum[c] += sign * q[c]
			}
		}
		mean := q0.normalized()
		if math.Sqrt(sum.dot(sum)) > 1e-9 {
			mean = sum.normalized()
		}
		inverse := mean.inverse()
		for t := 0; t < frames; t++ {
			q := at(t, j).normalized()
			r := clampScale(inverse.mul(q), maxRad, sway)
			s := slerp(q, target.mul(r), strength).normalized()
			i := (t*joints + j) * 4
			for c := 0; c < 4; c++ {
				out[i+c] = float32(s[c])
			}
		}
	}
	return out, nil
}

// applyPoseFile applies <dir>/pose.json's layer to the generated rotations:
// the engine's output is first copied to local_rotations_xyzw.model.f32 (or
// read back from it when it already exists), then the posed rotations replace
// local_rotations_xyzw.f32. Without pose.json or a layer it does nothing.
func applyPoseFile(dir string, names []string) error {
	b, err := os.ReadFile(filepath.Join(dir, poseFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var pose poseRequest
	if err := json.Unmarshal(b, &pose); err != nil {
		return fmt.Errorf("read %s: %w", poseFile, err)
	}
	if pose.Layer == nil {
		return nil
	}
	if err := pose.validate(names); err != nil {
		return err
	}
	modelPath := filepath.Join(dir, modelRotationFile)
	posedPath := filepath.Join(dir, "local_rotations_xyzw.f32")
	raw, err := readF32(modelPath)
	if errors.Is(err, os.ErrNotExist) {
		if raw, err = readF32(posedPath); err == nil {
			err = writeF32(modelPath, raw)
		}
	}
	if err != nil {
		return err
	}
	posed, err := applyPoseLayer(raw, names, pose.Layer)
	if err != nil {
		return err
	}
	return writeF32(posedPath, posed)
}
