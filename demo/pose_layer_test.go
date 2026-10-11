package main

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The JavaScript twin (editor/poses.js applyPoseLayer) wrote the expected
// outputs (editor/tests/fixtures/gen_pose_layer.mjs).
func TestApplyPoseLayerFixture(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("editor", "tests", "fixtures", "pose_layer.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Names     []string  `json:"names"`
		Frames    int       `json:"frames"`
		Rotations []float32 `json:"rotations"`
		Cases     []struct {
			Layer    poseLayer `json:"layer"`
			Expected []float32 `json:"expected"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 || len(fixture.Rotations) != fixture.Frames*len(fixture.Names)*4 {
		t.Fatalf("bad fixture")
	}
	for index, c := range fixture.Cases {
		got, err := applyPoseLayer(fixture.Rotations, fixture.Names, &c.Layer)
		if err != nil {
			t.Fatal(err)
		}
		worst := 0.0
		for i := range got {
			worst = math.Max(worst, math.Abs(float64(got[i]-c.Expected[i])))
		}
		if worst > 2e-6 {
			t.Fatalf("case %d: max difference to the JavaScript layer %g", index, worst)
		}
	}
}

func TestApplyPoseLayerHold(t *testing.T) {
	names := []string{"Hips", "LeftArm", "RightArm"}
	const frames = 5
	// LeftArm sways about +-6 degrees around x; Hips and RightArm are identity.
	rotations := make([]float32, frames*3*4)
	for frame := 0; frame < frames; frame++ {
		for joint := 0; joint < 3; joint++ {
			rotations[(frame*3+joint)*4+3] = 1
		}
		angle := (float64(frame) - 2) * 3 * math.Pi / 180
		i := (frame*3 + 1) * 4
		rotations[i], rotations[i+3] = float32(math.Sin(angle/2)), float32(math.Cos(angle/2))
	}
	target := quat{0, 0, math.Sin(math.Pi / 8), math.Cos(math.Pi / 8)} // 45 degrees about z
	at := func(values []float32, frame, joint int) quat {
		i := (frame*3 + joint) * 4
		return quat{float64(values[i]), float64(values[i+1]), float64(values[i+2]), float64(values[i+3])}
	}
	// Angle of a^-1 b via atan2 (acos of a float32 dot product is too coarse near 0).
	angle := func(a, b quat) float64 {
		d := a.normalized().inverse().mul(b.normalized())
		return 2 * math.Atan2(math.Sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]), math.Abs(d[3])) * 180 / math.Pi
	}

	rigid, err := applyPoseLayer(rotations, names, &poseLayer{Mode: "hold", Joints: []string{"LeftArm"}, Rotations: []quatJSON{quatJSON(target)}, Sway: new(0.0)})
	if err != nil {
		t.Fatal(err)
	}
	swaying, err := applyPoseLayer(rotations, names, &poseLayer{Mode: "hold", Joints: []string{"LeftArm"}, Rotations: []quatJSON{quatJSON(target)}})
	if err != nil {
		t.Fatal(err)
	}
	for frame := 0; frame < frames; frame++ {
		if d := angle(at(rigid, frame, 1), target); d > 1e-3 {
			t.Fatalf("sway 0, frame %d: %.4f degrees from the target", frame, d)
		}
		// The sway around the clip mean (frame 2) survives on top of the target.
		want := math.Abs(float64(frame)-2) * 3
		if d := angle(at(swaying, frame, 1), target); math.Abs(d-want) > 1e-3 {
			t.Fatalf("sway 1, frame %d: %.4f degrees from the target, want %.1f", frame, d, want)
		}
		for _, joint := range []int{0, 2} {
			if at(swaying, frame, joint) != at(rotations, frame, joint) {
				t.Fatalf("unlisted joint %d changed", joint)
			}
		}
	}
	half, _ := applyPoseLayer(rotations, names, &poseLayer{Mode: "hold", Joints: []string{"LeftArm"}, Rotations: []quatJSON{quatJSON(target)}, Sway: new(0.0), Strength: new(.5)})
	if d := angle(at(half, 2, 1), target); math.Abs(d-22.5) > 1e-3 {
		t.Fatalf("strength 0.5: %.4f degrees from the target, want 22.5", d)
	}
	// The sway is clamped per joint.
	clamped, _ := applyPoseLayer(rotations, names, &poseLayer{Mode: "hold", Joints: []string{"LeftArm"}, Rotations: []quatJSON{quatJSON(target)}, MaxSwayDeg: new(2.0)})
	if d := angle(at(clamped, 0, 1), target); math.Abs(d-2) > 1e-3 {
		t.Fatalf("clamped sway: %.4f degrees, want 2", d)
	}
	if _, err := applyPoseLayer(rotations, names, &poseLayer{Mode: "hold", Joints: []string{"Nope"}, Rotations: []quatJSON{quatJSON(target)}}); err == nil {
		t.Fatal("unknown joint accepted")
	}
}

func TestPoseRequestValidate(t *testing.T) {
	names := skeletonDefinitions["soma30"].names
	good := func() *poseRequest {
		return &poseRequest{ID: "rifle_low_ready", Label: "Rifle: low ready", Params: json.RawMessage(`{"hold":100}`),
			Layer: &poseLayer{Mode: "hold", Joints: []string{"LeftArm", "RightArm"}, Rotations: []quatJSON{{0, 0, 0, 1}, {0, 1, 0, 0}}}}
	}
	if err := good().validate(names); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		mutate func(*poseRequest)
		err    string
	}{
		{"id", func(p *poseRequest) { p.ID = "Rifle!" }, "pose id"},
		{"params", func(p *poseRequest) { p.Params = json.RawMessage(`[1]`) }, "JSON object"},
		{"mode", func(p *poseRequest) { p.Layer.Mode = "add" }, "unknown pose layer mode"},
		{"count", func(p *poseRequest) { p.Layer.Rotations = p.Layer.Rotations[:1] }, "one rotation per joint"},
		{"joint", func(p *poseRequest) { p.Layer.Joints[1] = "Tail" }, "not in this skeleton"},
		{"duplicate", func(p *poseRequest) { p.Layer.Joints[1] = "LeftArm" }, "listed twice"},
		{"zero quaternion", func(p *poseRequest) { p.Layer.Rotations[0] = quatJSON{} }, "usable quaternion"},
		{"strength", func(p *poseRequest) { p.Layer.Strength = new(1.5) }, "strength must be 0..1"},
		{"sway", func(p *poseRequest) { p.Layer.Sway = new(-0.1) }, "sway must be 0..1"},
		{"max sway", func(p *poseRequest) { p.Layer.MaxSwayDeg = new(200.0) }, "max_sway_deg"},
	} {
		t.Run(test.name, func(t *testing.T) {
			p := good()
			test.mutate(p)
			if err := p.validate(names); err == nil || !strings.Contains(err.Error(), test.err) {
				t.Fatalf("error %v, want %q", err, test.err)
			}
		})
	}
	if err := (&poseRequest{ID: "crouch"}).validate(names); err != nil {
		t.Fatalf("a pose without a layer: %v", err)
	}
}

func TestApplyPoseFile(t *testing.T) {
	dir := t.TempDir()
	names := []string{"Hips", "LeftArm"}
	rotations := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}
	if err := writeF32(filepath.Join(dir, "local_rotations_xyzw.f32"), rotations); err != nil {
		t.Fatal(err)
	}
	if err := applyPoseFile(dir, names); err != nil { // no pose.json: nothing to do
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, modelRotationFile)); !os.IsNotExist(err) {
		t.Fatalf("model rotations written without a pose (%v)", err)
	}
	pose := `{"id":"test","layer":{"mode":"hold","joints":["LeftArm"],"rotations":[[0,0,1,0]]}}`
	if err := os.WriteFile(filepath.Join(dir, poseFile), []byte(pose), 0600); err != nil {
		t.Fatal(err)
	}
	for pass := 0; pass < 2; pass++ { // idempotent: always derived from the model output
		if err := applyPoseFile(dir, names); err != nil {
			t.Fatal(err)
		}
		posed, _ := readF32(filepath.Join(dir, "local_rotations_xyzw.f32"))
		model, _ := readF32(filepath.Join(dir, modelRotationFile))
		if !equalF32(model, rotations) || !equalF32(posed, []float32{0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0}) {
			t.Fatalf("pass %d: posed %v model %v", pass, posed, model)
		}
	}
}

func equalF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-6 {
			return false
		}
	}
	return true
}

func TestWorkerPoseLayer(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("KIMODO_DEMO_FAKE_GENERATOR", "1")
	t.Setenv("KIMODO_FAKE_JOINTS", "30")
	g := newTestGallery(t)
	g.generator = executable
	server := httptest.NewServer(g.routes(t.TempDir(), embeddedWebFiles()))
	defer server.Close()
	done := make(chan struct{})
	go func() {
		g.worker()
		close(done)
	}()
	defer func() {
		close(g.queue)
		<-done
	}()
	wait := func(id string) *animation {
		t.Helper()
		deadline := time.Now().Add(30 * time.Second)
		for time.Now().Before(deadline) {
			g.mu.RLock()
			current := *g.items[id]
			g.mu.RUnlock()
			if current.Status == "ready" || current.Status == "failed" {
				return &current
			}
			time.Sleep(20 * time.Millisecond)
		}
		t.Fatalf("animation %s did not finish", id)
		return nil
	}
	target := [4]float64{0, 0, math.Sin(math.Pi / 8), math.Cos(math.Pi / 8)}
	for _, rotation := range []string{`[0.3,-0.2,-0.1]`, `[0,0,0,1,7]`, `["x",0,0,1]`} {
		status, text := postJSON(t, server, []byte(`{"prompt":"walk","model":"soma-rp-v1.1","pose":{"id":"x","layer":{"mode":"hold","joints":["LeftArm"],"rotations":[`+rotation+`]}}}`))
		if status != http.StatusBadRequest {
			t.Fatalf("rotation %s: %d %s", rotation, status, text)
		}
	}
	body := mustJSON(t, map[string]any{"prompt": "walk", "frames": 60, "model": "soma-rp-v1.1",
		"pose": map[string]any{"id": "rifle_collapsed_low_ready", "label": "Rifle: collapsed low ready", "params": map[string]any{"hold": 100, "sway": 50},
			"layer": map[string]any{"mode": "hold", "joints": []string{"LeftArm", "RightForeArm"}, "rotations": [][4]float64{target, target}, "sway": .5}}})
	status, text := postJSON(t, server, body)
	if status != http.StatusAccepted || !strings.Contains(text, `"pose":{"id":"rifle_collapsed_low_ready","label":"Rifle: collapsed low ready","params":{"hold":100,"sway":50},"layered":true}`) {
		t.Fatalf("status %d: %s", status, text)
	}
	var accepted animation
	_ = json.Unmarshal([]byte(text), &accepted)
	// pose.json is served while the animation is queued or running, too.
	if response, served := get(t, server, "/api/animations/"+accepted.ID+"/pose.json"); response.StatusCode != 200 || !bytes.Contains(served, []byte(`"layer":{"mode":"hold"`)) {
		t.Fatalf("pose.json: %d %s", response.StatusCode, served)
	}
	a := wait(accepted.ID)
	if a.Status != "ready" {
		t.Fatalf("posed generation failed: %s", a.Error)
	}
	// The fake generator writes identity rotations: the held joints end on the target.
	_, posed := get(t, server, "/api/animations/"+a.ID+"/rotations.f32")
	_, model := get(t, server, "/api/animations/"+a.ID+"/model_rotations.f32")
	if len(posed) != 60*30*16 || len(model) != len(posed) {
		t.Fatalf("rotations %d bytes, model %d bytes", len(posed), len(model))
	}
	values := func(b []byte) []float32 {
		path := filepath.Join(t.TempDir(), "x.f32")
		_ = os.WriteFile(path, b, 0600)
		v, _ := readF32(path)
		return v
	}
	p, m := values(posed), values(model)
	for frame := 0; frame < 60; frame++ {
		for joint := 0; joint < 30; joint++ {
			i := (frame*30 + joint) * 4
			want := []float32{0, 0, 0, 1}
			if joint == 11 || joint == 18 {
				want = []float32{0, 0, float32(target[2]), float32(target[3])}
			}
			if !equalF32(p[i:i+4], want) || !equalF32(m[i:i+4], []float32{0, 0, 0, 1}) {
				t.Fatalf("frame %d joint %d: posed %v model %v", frame, joint, p[i:i+4], m[i:i+4])
			}
		}
	}
	record, _ := os.ReadFile(filepath.Join(g.output, a.ID+".json"))
	if decoded, err := decodeAnimation(record); err != nil || decoded.Pose == nil || decoded.Pose.ID != "rifle_collapsed_low_ready" || !decoded.Pose.Layered {
		t.Fatalf("record pose = %+v (%v)", decoded, err)
	}
	// Without a layer nothing is rewritten and no model copy is kept.
	status, text = postJSON(t, server, []byte(`{"prompt":"walk","frames":60,"model":"soma-rp-v1.1","pose":{"id":"crouch"}}`))
	_ = json.Unmarshal([]byte(text), &accepted)
	if a := wait(accepted.ID); status != http.StatusAccepted || a.Status != "ready" || a.Pose == nil || a.Pose.Layered {
		t.Fatalf("crouch: %d %s", status, text)
	}
	if _, err := os.Stat(filepath.Join(g.output, accepted.ID, modelRotationFile)); !os.IsNotExist(err) {
		t.Fatalf("model rotations kept without a layer (%v)", err)
	}
	before := len(g.items)
	status, text = postJSON(t, server, []byte(`{"prompt":"walk","model":"soma-rp-v1.1","pose":{"id":"x","layer":{"mode":"hold","joints":["Wing"],"rotations":[[0,0,0,1]]}}}`))
	if status != http.StatusBadRequest || !strings.Contains(text, "invalid pose") || len(g.items) != before {
		t.Fatalf("bad pose: %d %s", status, text)
	}
}
