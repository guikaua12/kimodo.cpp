package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestMain doubles as a fake `kmd-generate --server` when the worker test
// launches this test binary as its generator.
func TestMain(m *testing.M) {
	if os.Getenv("KIMODO_DEMO_FAKE_GENERATOR") == "1" {
		os.Exit(runFakeGenerator())
	}
	os.Exit(m.Run())
}

// runFakeGenerator imitates the native server: it accepts only v2 JSON request
// lines (unknown keys rejected), records each request beside its output and
// writes zero-motion files of the requested length.
func runFakeGenerator() int {
	joints, _ := strconv.Atoi(os.Getenv("KIMODO_FAKE_JOINTS"))
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		line := scanner.Bytes()
		var request generatorRequest
		decoder := json.NewDecoder(bytes.NewReader(line))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&request); err != nil {
			fmt.Printf("ERR\tnot a v2 request: %s\n", strings.ReplaceAll(err.Error(), "\t", " "))
			continue
		}
		frames, failure := 0, ""
		for _, segment := range request.Segments {
			prompt, err := os.ReadFile(segment.PromptFile)
			if err != nil {
				failure = "cannot read prompt"
			} else if strings.Contains(string(prompt), "please fail") {
				failure = "fake failure"
			}
			frames += segment.Frames
		}
		if request.ConstraintsFile != "" {
			if _, err := os.Stat(request.ConstraintsFile); err != nil {
				failure = "cannot read constraints"
			}
		}
		if failure != "" {
			fmt.Printf("ERR\t%s\n", failure)
			continue
		}
		rotations := make([]float32, frames*joints*4)
		for index := 3; index < len(rotations); index += 4 {
			rotations[index] = 1
		}
		write := func(name string, values []float32) {
			if err := writeF32(filepath.Join(request.Output, name), values); err != nil && failure == "" {
				failure = err.Error()
			}
		}
		write("root_positions.f32", make([]float32, frames*3))
		write("local_rotations_xyzw.f32", rotations)
		write("foot_contacts.f32", make([]float32, frames*4))
		write("smooth_root.f32", make([]float32, frames*3))
		_ = os.WriteFile(filepath.Join(request.Output, "request.json"), append(slices.Clone(line), '\n'), 0600)
		_ = os.WriteFile(filepath.Join(request.Output, "generator.pid"), []byte(strconv.Itoa(os.Getpid())), 0600)
		if failure != "" {
			fmt.Printf("ERR\t%s\n", failure)
			continue
		}
		fmt.Printf("OK\t%d\t%d\n", frames, joints)
	}
	return 0
}

func TestWriteSkeletonGLB(t *testing.T) {
	for _, skeletonKey := range []string{"smplx22", "soma30", "g1skel34"} {
		t.Run(skeletonKey, func(t *testing.T) {
			skeleton := skeletonDefinitions[skeletonKey]
			path := filepath.Join(t.TempDir(), "animation.glb")
			roots := []float32{0, 0, 0, 1, 0, 0}
			rotations := make([]float32, 2*len(skeleton.parents)*4)
			for frame := 0; frame < 2; frame++ {
				for joint := range skeleton.parents {
					rotations[(frame*len(skeleton.parents)+joint)*4+3] = 1
				}
			}
			if err := writeSkeletonGLB(path, roots, rotations, skeleton); err != nil {
				t.Fatal(err)
			}
			assertSkeletonGLB(t, path, len(skeleton.parents))

			// A second export of the same motion must not rewrite the file.
			old := time.Now().Add(-time.Hour).Truncate(time.Second)
			if err := os.Chtimes(path, old, old); err != nil {
				t.Fatal(err)
			}
			if err := writeSkeletonGLB(path, roots, rotations, skeleton); err != nil {
				t.Fatal(err)
			}
			if info, err := os.Stat(path); err != nil || !info.ModTime().Equal(old) {
				t.Fatalf("identical GLB was rewritten: %v %v", info.ModTime(), err)
			}
			// A changed motion is written.
			roots[3] = 2
			if err := writeSkeletonGLB(path, roots, rotations, skeleton); err != nil {
				t.Fatal(err)
			}
			if info, err := os.Stat(path); err != nil || info.ModTime().Equal(old) {
				t.Fatalf("changed GLB was not rewritten: %v", err)
			}
		})
	}
}

func TestSafePathPart(t *testing.T) {
	for _, value := range []string{"q8_0", "run-jump.seed42", "BF16"} {
		if !safePathPart(value) {
			t.Fatalf("expected safe path part: %q", value)
		}
	}
	for _, value := range []string{"", ".", "..", "../escape", "a/b", "space here"} {
		if safePathPart(value) {
			t.Fatalf("expected unsafe path part: %q", value)
		}
	}
}

func assertSkeletonGLB(t *testing.T, path string, expectedJoints int) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) < 20 || binary.LittleEndian.Uint32(b) != 0x46546c67 || binary.LittleEndian.Uint32(b[4:]) != 2 || int(binary.LittleEndian.Uint32(b[8:])) != len(b) {
		t.Fatalf("invalid GLB header")
	}
	jsonLength := int(binary.LittleEndian.Uint32(b[12:]))
	if binary.LittleEndian.Uint32(b[16:]) != 0x4e4f534a || 20+jsonLength > len(b) {
		t.Fatalf("invalid GLB JSON chunk")
	}
	var document struct {
		Asset      map[string]string `json:"asset"`
		Nodes      []json.RawMessage `json:"nodes"`
		Animations []json.RawMessage `json:"animations"`
	}
	if err := json.Unmarshal(b[20:20+jsonLength], &document); err != nil {
		t.Fatal(err)
	}
	if document.Asset["version"] != "2.0" || len(document.Nodes) != expectedJoints || len(document.Animations) != 1 {
		t.Fatalf("unexpected GLB document: %s", b[20:20+jsonLength])
	}
}

// --- constraint fixtures -----------------------------------------------------

func rows(count, width int, value float64) [][]float64 {
	result := make([][]float64, count)
	for index := range result {
		result[index] = make([]float64, width)
		for axis := range result[index] {
			result[index][axis] = value
		}
	}
	return result
}

func root2dSet(frames ...int) map[string]any {
	return map[string]any{"type": "root2d", "frame_indices": append([]int{}, frames...), "smooth_root_2d": rows(len(frames), 2, .5)}
}

// poseSet builds a fullbody / limb / end-effector set with joints axis-angle
// rotations per frame.
func poseSet(kind string, joints int, frames ...int) map[string]any {
	rotations := make([][][]float64, len(frames))
	for index := range rotations {
		rotations[index] = rows(joints, 3, .1)
	}
	set := map[string]any{"type": kind, "frame_indices": frames, "local_joints_rot": rotations, "root_positions": rows(len(frames), 3, .9), "smooth_root_2d": rows(len(frames), 2, .2)}
	if kind == "end-effector" {
		set["joint_names"] = []string{"LeftFoot", "Hips"}
	}
	return set
}

func with(set map[string]any, key string, value any) map[string]any {
	clone := map[string]any{}
	for k, v := range set {
		clone[k] = v
	}
	if value == nil {
		delete(clone, key)
	} else {
		clone[key] = value
	}
	return clone
}

func mustJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestValidateConstraints(t *testing.T) {
	soma, smplx, g1 := skeletonDefinitions["soma30"], skeletonDefinitions["smplx22"], skeletonDefinitions["g1skel34"]
	ragged := poseSet("fullbody", 30, 1, 2)
	ragged["local_joints_rot"] = [][][]float64{rows(30, 3, 0), rows(29, 3, 0)}
	cases := []struct {
		name     string
		skeleton skeletonDefinition
		raw      string // used when value is nil
		value    any
		err      string // expected substring; empty = valid
		count    int
		types    []string
	}{
		{name: "absent", skeleton: soma},
		{name: "null", skeleton: soma, raw: "null"},
		{name: "empty array", skeleton: soma, raw: "[]"},
		{name: "mixed soma30", skeleton: soma, value: []any{
			poseSet("fullbody", 30, 0, 149), with(root2dSet(10, 11, 12), "global_root_heading", rows(3, 2, .7)),
			poseSet("left-hand", 30, 5), poseSet("end-effector", 30, 7), root2dSet(20)},
			count: 8, types: []string{"fullbody", "root2d", "left-hand", "end-effector"}},
		{name: "all limb types", skeleton: soma, value: []any{poseSet("left-hand", 30, 1), poseSet("right-hand", 30, 1), poseSet("left-foot", 30, 2), poseSet("right-foot", 30, 2)},
			count: 4, types: []string{"left-hand", "right-hand", "left-foot", "right-foot"}},
		{name: "soma30 accepts 77 joint files", skeleton: soma, value: []any{poseSet("fullbody", 77, 3)}, count: 1, types: []string{"fullbody"}},
		{name: "g1 34 joints", skeleton: g1, value: []any{poseSet("right-foot", 34, 3, 3, 0)}, count: 3, types: []string{"right-foot"}},
		{name: "unsorted repeated frames", skeleton: smplx, value: []any{root2dSet(129, 93, 0, 93)}, count: 4, types: []string{"root2d"}},
		{name: "null heading is absent", skeleton: soma, value: []any{with(root2dSet(1), "global_root_heading", json.RawMessage("null"))}, count: 1, types: []string{"root2d"}},
		{name: "smooth root optional for poses", skeleton: soma, value: []any{with(poseSet("fullbody", 30, 1), "smooth_root_2d", nil)}, count: 1, types: []string{"fullbody"}},
		{name: "smplx rejects 77 joints", skeleton: smplx, value: []any{poseSet("fullbody", 77, 3)}, err: "local_joints_rot must be a [K][22][3]"},
		{name: "wrong joint count", skeleton: soma, value: []any{poseSet("fullbody", 22, 3)}, err: "[K][30 or 77][3]"},
		{name: "object instead of array", skeleton: soma, value: root2dSet(1), err: "JSON array"},
		{name: "element not object", skeleton: soma, raw: "[1]", err: "constraints[0] must be an object"},
		{name: "missing type", skeleton: soma, value: []any{with(root2dSet(1), "type", nil)}, err: "missing type"},
		{name: "unknown type", skeleton: soma, value: []any{with(root2dSet(1), "type", "end-effectors")}, err: `unknown type "end-effectors"`},
		{name: "non-string type", skeleton: soma, value: []any{with(root2dSet(1), "type", 3)}, err: "unknown type 3"},
		{name: "frame equal to total", skeleton: soma, value: []any{root2dSet(150)}, err: "frame_indices[0] = 150 is outside"},
		{name: "negative frame", skeleton: soma, value: []any{root2dSet(1, -1)}, err: "frame_indices[1] = -1 is outside"},
		{name: "fractional frame", skeleton: soma, raw: `[{"type":"root2d","frame_indices":[1.5],"smooth_root_2d":[[0,0]]}]`, err: "must be an integer"},
		{name: "float-typed integer frames", skeleton: soma, raw: `[{"type":"root2d","frame_indices":[3.0,1e2,0.0],"smooth_root_2d":[[0,0],[0,0],[0,0]]}]`, count: 3, types: []string{"root2d"}},
		{name: "fractional exponent frame", skeleton: soma, raw: `[{"type":"root2d","frame_indices":[2.5e0],"smooth_root_2d":[[0,0]]}]`, err: "must be an integer"},
		{name: "float frame equal to total", skeleton: soma, raw: `[{"type":"root2d","frame_indices":[1.5e2],"smooth_root_2d":[[0,0]]}]`, err: "frame_indices[0] = 1.5e2 is outside"},
		{name: "huge frame", skeleton: soma, raw: `[{"type":"root2d","frame_indices":[1e400],"smooth_root_2d":[[0,0]]}]`, err: "must be an integer"},
		{name: "number out of range", skeleton: soma, raw: `[{"type":"root2d","frame_indices":[1],"smooth_root_2d":[[1e400,0]]}]`, err: "smooth_root_2d"},
		{name: "string frame", skeleton: soma, raw: `[{"type":"root2d","frame_indices":["3"],"smooth_root_2d":[[0,0]]}]`, err: "must be an integer"},
		{name: "empty frames", skeleton: soma, value: []any{root2dSet()}, err: "frame_indices must not be empty"},
		{name: "missing frames", skeleton: soma, value: []any{with(root2dSet(1), "frame_indices", nil)}, err: "frame_indices must be an array"},
		{name: "missing smooth root", skeleton: soma, value: []any{with(root2dSet(1), "smooth_root_2d", nil)}, err: "missing smooth_root_2d"},
		{name: "smooth root length", skeleton: soma, value: []any{with(root2dSet(1, 2), "smooth_root_2d", rows(1, 2, 0))}, err: "smooth_root_2d must be a [K][2] number array with K = len(frame_indices) = 2, got [1 2]"},
		{name: "smooth root width", skeleton: soma, value: []any{with(root2dSet(1), "smooth_root_2d", rows(1, 3, 0))}, err: "smooth_root_2d"},
		{name: "heading width", skeleton: soma, value: []any{with(root2dSet(1), "global_root_heading", rows(1, 1, 0))}, err: "global_root_heading"},
		{name: "missing root positions", skeleton: soma, value: []any{with(poseSet("fullbody", 30, 1), "root_positions", nil)}, err: "missing root_positions"},
		{name: "root positions length", skeleton: soma, value: []any{with(poseSet("left-foot", 30, 1, 2), "root_positions", rows(3, 3, 0))}, err: "root_positions"},
		{name: "missing rotations", skeleton: soma, value: []any{with(poseSet("fullbody", 30, 1), "local_joints_rot", nil)}, err: "missing local_joints_rot"},
		{name: "ragged rotations", skeleton: soma, value: []any{ragged}, err: "not a rectangular number array"},
		{name: "null number", skeleton: soma, raw: `[{"type":"root2d","frame_indices":[1],"smooth_root_2d":[[null,0]]}]`, err: "smooth_root_2d"},
		{name: "end-effector without names", skeleton: soma, value: []any{with(poseSet("end-effector", 30, 1), "joint_names", nil)}, err: "joint_names must be a non-empty array"},
		{name: "end-effector bad name", skeleton: soma, value: []any{with(poseSet("end-effector", 30, 1), "joint_names", []string{"left_hand"})}, err: `unknown joint name "left_hand"`},
		{name: "invalid JSON", skeleton: soma, raw: `[{"type":`, err: "valid JSON"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			raw := json.RawMessage(test.raw)
			if test.value != nil {
				raw = mustJSON(t, test.value)
			}
			summary, err := validateConstraints(raw, test.skeleton, 150)
			if test.err != "" {
				if err == nil || !strings.Contains(err.Error(), test.err) {
					t.Fatalf("error = %v, want substring %q", err, test.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if summary.Count != test.count || !slices.Equal(summary.Types, append([]string{}, test.types...)) {
				t.Fatalf("summary = %+v, want count %d types %v", summary, test.count, test.types)
			}
		})
	}
}

func TestGenerateRequestNormalize(t *testing.T) {
	decode := func(t *testing.T, body string) (generateRequest, error) {
		t.Helper()
		var request generateRequest
		if err := json.Unmarshal([]byte(body), &request); err != nil {
			t.Fatal(err)
		}
		return request, request.normalize()
	}
	t.Run("legacy body gets defaults", func(t *testing.T) {
		request, err := decode(t, `{"prompt":"  walk forward ","frames":150,"steps":100,"seed":0}`)
		if err != nil {
			t.Fatal(err)
		}
		if len(request.Segments) != 1 || request.Segments[0] != (promptSegment{Prompt: "walk forward", Frames: 150}) || request.Steps != 100 || request.TransitionFrames != 5 ||
			*request.TextCFG != 2 || *request.ConstraintCFG != 2 || *request.PostProcessing || *request.RootMargin != .04 ||
			request.Model != "smplx-rp-v1" || request.TextQuantization != "q8_0" || len(request.Constraints) != 0 {
			t.Fatalf("unexpected defaults: %+v", request)
		}
	})
	t.Run("minimal body", func(t *testing.T) {
		request, err := decode(t, `{"prompt":"walk"}`)
		if err != nil || request.Segments[0].Frames != 150 || request.Steps != 100 || request.totalFrames() != 150 {
			t.Fatalf("unexpected: %+v %v", request, err)
		}
	})
	t.Run("explicit zeros are kept", func(t *testing.T) {
		request, err := decode(t, `{"prompt":"walk","text_cfg":0,"constraint_cfg":0,"root_margin":0,"post_processing":true}`)
		if err != nil || *request.TextCFG != 0 || *request.ConstraintCFG != 0 || *request.RootMargin != 0 || !*request.PostProcessing {
			t.Fatalf("unexpected: %+v %v", request, err)
		}
	})
	t.Run("new fields", func(t *testing.T) {
		request, err := decode(t, `{"segments":[{"prompt":"a","frames":300},{"prompt":"b","frames":60}],"text_cfg":3.5,"constraint_cfg":1.25,"post_processing":false,"root_margin":0.1,"constraints":[]}`)
		if err != nil || request.totalFrames() != 360 || *request.TextCFG != 3.5 || *request.ConstraintCFG != 1.25 || *request.PostProcessing || *request.RootMargin != .1 || string(request.Constraints) != "[]" {
			t.Fatalf("unexpected: %+v %v", request, err)
		}
	})
	seventeen := strings.Repeat(`{"prompt":"a","frames":60},`, 17)
	for _, test := range []struct{ name, body, err string }{
		{"segment too long", `{"segments":[{"prompt":"a","frames":301}]}`, "60..300 frames"},
		{"segment too short", `{"segments":[{"prompt":"a","frames":59}]}`, "60..300 frames"},
		{"empty prompt", `{"segments":[{"prompt":"  ","frames":90}]}`, "1..4096 bytes"},
		{"too many segments", `{"segments":[` + strings.TrimSuffix(seventeen, ",") + `]}`, "at most 16"},
		{"steps", `{"prompt":"a","steps":1001}`, "steps 1..1000"},
		{"transition", `{"prompt":"a","transition_frames":61}`, "transition frames"},
		{"negative text cfg", `{"prompt":"a","text_cfg":-0.5}`, "text_cfg and constraint_cfg"},
		{"large constraint cfg", `{"prompt":"a","constraint_cfg":10.5}`, "text_cfg and constraint_cfg"},
		{"root margin", `{"prompt":"a","root_margin":1.5}`, "root_margin"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := decode(t, test.body); err == nil || !strings.Contains(err.Error(), test.err) {
				t.Fatalf("error = %v, want substring %q", err, test.err)
			}
		})
	}
	// Any uint64 seed is accepted (the protocol line carries it as a string).
	if request, err := decode(t, `{"prompt":"a","seed":18446744073709551615}`); err != nil || request.Seed != 18446744073709551615 {
		t.Fatalf("largest seed: %d (%v)", request.Seed, err)
	}
}

func TestGeneratorRequestLine(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "abc123")
	item := &animation{Seed: 42, DiffusionSteps: 50, TransitionFrames: 7, TextCFG: new(2.5), ConstraintCFG: new(1.5), PostProcessing: new(true), RootMargin: new(.04)}
	segments := []promptSegment{{Prompt: "walk", Frames: 150}, {Prompt: "run", Frames: 90}}
	prompts := []string{filepath.Join(dir, "segment-01.txt"), filepath.Join(dir, "segment-02.txt")}
	slash := func(path string) string {
		absolute, err := filepath.Abs(path)
		if err != nil {
			t.Fatal(err)
		}
		return filepath.ToSlash(absolute)
	}
	line, err := generatorRequestLine(item, dir, segments, prompts, filepath.Join(dir, "constraints.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(line, []byte("{")) || !bytes.HasSuffix(line, []byte("}\n")) || bytes.Count(line, []byte("\n")) != 1 || bytes.Contains(line, []byte(`\\`)) {
		t.Fatalf("not a single forward-slash JSON line: %q", line)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(line, &fields); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	want := []string{"constraint_cfg", "constraints_file", "output", "post_processing", "root_margin", "seed", "segments", "steps", "text_cfg", "transition_frames"}
	if !slices.Equal(keys, want) {
		t.Fatalf("keys = %v, want %v", keys, want)
	}
	var request generatorRequest
	if err := json.Unmarshal(line, &request); err != nil {
		t.Fatal(err)
	}
	expected := generatorRequest{Output: slash(dir), Seed: 42, Steps: 50, TransitionFrames: 7, TextCFG: 2.5, ConstraintCFG: 1.5, PostProcessing: true, RootMargin: .04, ConstraintsFile: slash(filepath.Join(dir, "constraints.json")),
		Segments: []generatorSegment{{Frames: 150, PromptFile: slash(prompts[0])}, {Frames: 90, PromptFile: slash(prompts[1])}}}
	if fmt.Sprint(request) != fmt.Sprint(expected) {
		t.Fatalf("request = %+v\nwant      %+v", request, expected)
	}
	for _, literal := range []string{`"text_cfg":2.5`, `"constraint_cfg":1.5`, `"post_processing":true`, `"root_margin":0.04`, `"seed":"42"`} {
		if !bytes.Contains(line, []byte(literal)) {
			t.Fatalf("line lacks %s: %s", literal, line)
		}
	}
	// A record without the generation parameters (written before the
	// constraint editor) gets the legacy values; a seed above 2^53 stays exact.
	line, err = generatorRequestLine(&animation{Seed: 18446744073709551615, DiffusionSteps: 100, TransitionFrames: 5}, dir, segments[:1], prompts[:1], "")
	if err != nil || bytes.Contains(line, []byte("constraints_file")) || !bytes.Contains(line, []byte(`"seed":"18446744073709551615"`)) ||
		!bytes.Contains(line, []byte(`"text_cfg":2,"constraint_cfg":2,`)) || !bytes.Contains(line, []byte(`"post_processing":false,"root_margin":0.04}`)) {
		t.Fatalf("unconstrained line = %s (%v)", line, err)
	}
	if _, err := generatorRequestLine(item, dir, segments, prompts[:1], ""); err == nil {
		t.Fatal("mismatched prompt files accepted")
	}
}

func TestDecodeAnimation(t *testing.T) {
	legacy := `{"id":"7da8b15e05c02c91","prompt":"walk","frames":120,"diffusion_steps":50,"seed":301,"created_at":"2026-10-05T22:55:44Z","status":"ready","kind":"generated","model":"soma-rp-v1.1","text_quantization":"q8_0","segments":[{"prompt":"walk","frames":120}],"transition_frames":10}`
	a, err := decodeAnimation([]byte(legacy))
	if err != nil {
		t.Fatal(err)
	}
	// The generation parameters of a record written before the constraint
	// editor stay absent, so the editor keeps its own settings.
	if a.TextCFG != nil || a.ConstraintCFG != nil || a.PostProcessing != nil || a.RootMargin != nil || a.ConstraintCount != 0 || a.ConstraintTypes == nil || len(a.ConstraintTypes) != 0 || a.Seed != 301 || a.Model != "soma-rp-v1.1" {
		t.Fatalf("legacy record = %+v", a)
	}
	encoded := mustJSON(t, a)
	for _, literal := range []string{`"constraint_count":0`, `"constraint_types":[]`} {
		if !bytes.Contains(encoded, []byte(literal)) {
			t.Fatalf("encoded legacy record lacks %s: %s", literal, encoded)
		}
	}
	for _, key := range []string{`"text_cfg"`, `"constraint_cfg"`, `"post_processing"`, `"root_margin"`} {
		if bytes.Contains(encoded, []byte(key)) {
			t.Fatalf("encoded legacy record has %s: %s", key, encoded)
		}
	}
	a, err = decodeAnimation([]byte(`{"id":"x","status":"ready","model":"g1-rp-v1","text_cfg":0,"constraint_cfg":0,"post_processing":false,"root_margin":0,"constraint_count":3,"constraint_types":["root2d"]}`))
	if err != nil || a.TextCFG == nil || *a.TextCFG != 0 || *a.ConstraintCFG != 0 || a.PostProcessing == nil || *a.PostProcessing || *a.RootMargin != 0 ||
		a.ConstraintCount != 3 || !slices.Equal(a.ConstraintTypes, []string{"root2d"}) || a.TextQuantization != "bf16" || a.Model != "g1-rp-v1" {
		t.Fatalf("new record = %+v (%v)", a, err)
	}
	if encoded := mustJSON(t, a); !bytes.Contains(encoded, []byte(`"text_cfg":0,"constraint_cfg":0,"post_processing":false,"root_margin":0,`)) {
		t.Fatalf("explicit zero parameters dropped: %s", encoded)
	}
	// The first records (before the model selector) are SMPL-X RP v1.
	a, err = decodeAnimation([]byte(`{"id":"x","prompt":"walk","frames":150,"diffusion_steps":100,"seed":0,"status":"ready","kind":"generated"}`))
	if err != nil || a.Model != "smplx-rp-v1" || a.TextQuantization != "bf16" {
		t.Fatalf("first-day record = %+v (%v)", a, err)
	}
}

// TestGalleryLoad checks the startup scan: records a stopped server left
// queued or running become failed (saved, constraints still served) and are
// never re-queued; ready records are untouched.
func TestGalleryLoad(t *testing.T) {
	g := newTestGallery(t)
	records := map[string]string{
		"aaaaaaaaaaaaaaaa": `{"id":"aaaaaaaaaaaaaaaa","prompt":"walk","frames":150,"status":"running","progress":"Generating 1 segment","model":"soma-rp-v1.1","constraint_count":2,"constraint_types":["root2d"],"text_cfg":2,"constraint_cfg":2,"post_processing":true,"root_margin":0.04}`,
		"bbbbbbbbbbbbbbbb": `{"id":"bbbbbbbbbbbbbbbb","prompt":"jump","frames":90,"status":"queued","model":"soma-rp-v1.1"}`,
		"cccccccccccccccc": `{"id":"cccccccccccccccc","prompt":"run","frames":60,"status":"ready","model":"soma-rp-v1.1"}`,
		"misnamed":         `{"id":"dddddddddddddddd","prompt":"sit","frames":60,"status":"running","model":"soma-rp-v1.1"}`,
	}
	for name, record := range records {
		if err := os.WriteFile(filepath.Join(g.output, name+".json"), []byte(record), 0644); err != nil {
			t.Fatal(err)
		}
	}
	constraints := []byte(`[{"type":"root2d","frame_indices":[0,149],"smooth_root_2d":[[0,0],[1,2]]}]`)
	if err := os.MkdirAll(filepath.Join(g.output, "aaaaaaaaaaaaaaaa"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(g.output, "aaaaaaaaaaaaaaaa", "constraints.json"), constraints, 0600); err != nil {
		t.Fatal(err)
	}
	g.load()
	if len(g.queue) != 0 {
		t.Fatalf("load queued %d animations", len(g.queue))
	}
	for _, id := range []string{"aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "dddddddddddddddd"} {
		a := g.items[id]
		if a == nil || a.Status != "failed" || a.Error != interruptedError || a.Progress != "" {
			t.Fatalf("%s after load = %+v", id, a)
		}
	}
	if a := g.items["cccccccccccccccc"]; a == nil || a.Status != "ready" || a.Error != "" {
		t.Fatalf("ready record after load = %+v", a)
	}
	saved, err := os.ReadFile(filepath.Join(g.output, "aaaaaaaaaaaaaaaa.json"))
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := decodeAnimation(saved)
	if err != nil || reloaded.Status != "failed" || reloaded.Progress != "" || reloaded.PostProcessing == nil || !*reloaded.PostProcessing || reloaded.ConstraintCount != 2 {
		t.Fatalf("saved record = %s (%v)", saved, err)
	}
	// A record stored under another name is marked failed in memory only.
	if misnamed, _ := os.ReadFile(filepath.Join(g.output, "misnamed.json")); string(misnamed) != records["misnamed"] {
		t.Fatalf("misnamed record rewritten: %s", misnamed)
	}
	server := httptest.NewServer(g.routes(t.TempDir(), embeddedWebFiles()))
	defer server.Close()
	if response, body := get(t, server, "/api/animations/aaaaaaaaaaaaaaaa/constraints.json"); response.StatusCode != 200 || !bytes.Equal(body, constraints) {
		t.Fatalf("constraints.json of the interrupted animation: %d %s", response.StatusCode, body)
	}
}

func TestSkeletonControlMetadata(t *testing.T) {
	expected := map[string]struct {
		hips   [2]int
		chains map[string][]int
	}{
		"smplx22":  {[2]int{2, 1}, map[string][]int{"LeftFoot": {7, 10}, "RightFoot": {8, 11}, "LeftHand": {20}, "RightHand": {21}}},
		"soma30":   {[2]int{26, 22}, map[string][]int{"LeftFoot": {24, 25}, "RightFoot": {28, 29}, "LeftHand": {13, 15}, "RightHand": {19, 21}}},
		"g1skel34": {[2]int{8, 1}, map[string][]int{"LeftFoot": {6, 7}, "RightFoot": {13, 14}, "LeftHand": {24, 25}, "RightHand": {32, 33}}},
	}
	for key, want := range expected {
		skeleton := skeletonDefinitions[key]
		model := newMotionModel("id", "label", "skeleton", key, "", "", "", filepath.Join(t.TempDir(), "missing.gguf"), true)
		if model.Hips != want.hips || model.RootIndex != 0 || skeleton.parents[model.RootIndex] != -1 || !slices.Equal(model.Names, skeleton.names) || len(model.Names) != len(model.Parents) {
			t.Fatalf("%s: hips %v root %d names %d parents %d", key, model.Hips, model.RootIndex, len(model.Names), len(model.Parents))
		}
		if len(model.Chains) != len(want.chains) {
			t.Fatalf("%s: chains %v", key, model.Chains)
		}
		for name, indices := range want.chains {
			got := make([]int, 0, len(model.Chains[name]))
			for _, joint := range model.Chains[name] {
				got = append(got, skeleton.jointIndex(joint))
			}
			if !slices.Equal(got, indices) {
				t.Fatalf("%s %s: chain %v -> %v, want %v", key, name, model.Chains[name], got, indices)
			}
		}
	}
}

// --- HTTP ----------------------------------------------------------------

// newTestGallery builds a gallery over a temp output directory with every
// model and the q8_0 text bundle available (dummy files), except smplx-rp-v1.
func newTestGallery(t *testing.T) *gallery {
	t.Helper()
	root := t.TempDir()
	dummy := filepath.Join(root, "dummy.gguf")
	if err := os.WriteFile(dummy, []byte("gguf"), 0600); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(root, "demo-output")
	if err := os.MkdirAll(output, 0755); err != nil {
		t.Fatal(err)
	}
	models := map[string]motionModel{
		"smplx-rp-v1":  newMotionModel("smplx-rp-v1", "SMPL-X RP v1", "SMPL-X 22 joints", "smplx22", "", "", "", filepath.Join(root, "missing.gguf"), false),
		"soma-rp-v1.1": newMotionModel("soma-rp-v1.1", "SOMA RP v1.1", "SOMA 30", "soma30", "", "", "", dummy, true),
		"g1-rp-v1":     newMotionModel("g1-rp-v1", "G1 RP v1", "G1 34", "g1skel34", "", "", "", dummy, true),
	}
	textBundles := map[string]textBundle{
		"q8_0": {ID: "q8_0", Label: "Q8_0", Available: true, Path: dummy},
		"bf16": {ID: "bf16", Label: "BF16", Reason: "missing"},
	}
	return &gallery{items: map[string]*animation{}, output: output, queue: make(chan string, 32), models: models, textBundles: textBundles}
}

func postJSON(t *testing.T, server *httptest.Server, body []byte) (int, string) {
	t.Helper()
	response, err := http.Post(server.URL+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	b, _ := io.ReadAll(response.Body)
	return response.StatusCode, string(b)
}

func get(t *testing.T, server *httptest.Server, path string) (*http.Response, []byte) {
	t.Helper()
	response, err := http.Get(server.URL + path)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	b, _ := io.ReadAll(response.Body)
	return response, b
}

func TestGenerateEndpoint(t *testing.T) {
	g := newTestGallery(t)
	server := httptest.NewServer(g.routes(t.TempDir(), embeddedWebFiles()))
	defer server.Close()

	// A dense full-body interval (150 frames x 30 joints) is far above the old
	// 32 KiB body limit.
	frames := make([]int, 150)
	for index := range frames {
		frames[index] = index
	}
	constraints := mustJSON(t, []any{poseSet("fullbody", 30, frames...), root2dSet(10, 11, 12), poseSet("left-hand", 30, 40)})
	body := mustJSON(t, map[string]any{"segments": []promptSegment{{Prompt: "walk", Frames: 150}, {Prompt: "wave", Frames: 60}}, "model": "soma-rp-v1.1", "seed": 3, "steps": 20,
		"text_cfg": 2.5, "constraint_cfg": 1.5, "post_processing": true, "root_margin": 0.05, "constraints": constraints})
	if len(body) < 32<<10 {
		t.Fatalf("fixture too small to exercise the raised limit: %d bytes", len(body))
	}
	status, text := postJSON(t, server, body)
	if status != http.StatusAccepted {
		t.Fatalf("status %d: %s", status, text)
	}
	var a animation
	if err := json.Unmarshal([]byte(text), &a); err != nil {
		t.Fatal(err)
	}
	if a.Status != "queued" || a.Frames != 210 || *a.TextCFG != 2.5 || *a.ConstraintCFG != 1.5 || !*a.PostProcessing || *a.RootMargin != .05 || a.ConstraintCount != 154 || !slices.Equal(a.ConstraintTypes, []string{"fullbody", "root2d", "left-hand"}) {
		t.Fatalf("animation = %+v", a)
	}
	if queued := <-g.queue; queued != a.ID {
		t.Fatalf("queued %q, want %q", queued, a.ID)
	}
	saved, err := os.ReadFile(filepath.Join(g.output, a.ID, "constraints.json"))
	if err != nil || !bytes.Equal(saved, constraints) {
		t.Fatalf("constraints.json not saved verbatim (%v)", err)
	}
	record, err := os.ReadFile(filepath.Join(g.output, a.ID+".json"))
	if err != nil || !bytes.Contains(record, []byte(`"constraint_count": 154`)) || !bytes.Contains(record, []byte(`"post_processing": true`)) {
		t.Fatalf("record = %s (%v)", record, err)
	}
	// constraints.json is served before the animation is ready ...
	response, served := get(t, server, "/api/animations/"+a.ID+"/constraints.json")
	if response.StatusCode != 200 || !bytes.Equal(served, constraints) || response.Header.Get("Cache-Control") != "no-store" || response.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("constraints.json: %d %v", response.StatusCode, response.Header)
	}
	// ... motion files are not.
	if response, _ := get(t, server, "/api/animations/"+a.ID+"/contacts.f32"); response.StatusCode != 404 {
		t.Fatalf("contacts.f32 of a queued animation: %d", response.StatusCode)
	}

	t.Run("legacy body", func(t *testing.T) {
		status, text := postJSON(t, server, []byte(`{"prompt":"walk","frames":150,"steps":100,"seed":0,"model":"soma-rp-v1.1"}`))
		if status != http.StatusAccepted {
			t.Fatalf("status %d: %s", status, text)
		}
		var legacy animation
		_ = json.Unmarshal([]byte(text), &legacy)
		<-g.queue
		// An old client's request is stored with the defaults it was generated with.
		if legacy.TextCFG == nil || *legacy.TextCFG != 2 || *legacy.ConstraintCFG != 2 || *legacy.PostProcessing || *legacy.RootMargin != .04 || legacy.ConstraintCount != 0 || legacy.ConstraintTypes == nil ||
			!strings.Contains(text, `"text_cfg":2,"constraint_cfg":2,"post_processing":false,"root_margin":0.04,`) || !strings.Contains(text, `"constraint_types":[]`) {
			t.Fatalf("legacy animation = %s", text)
		}
		if _, err := os.Stat(filepath.Join(g.output, legacy.ID, "constraints.json")); !os.IsNotExist(err) {
			t.Fatalf("unconstrained request wrote constraints.json (%v)", err)
		}
		if response, _ := get(t, server, "/api/animations/"+legacy.ID+"/constraints.json"); response.StatusCode != 404 {
			t.Fatalf("absent constraints.json: %d", response.StatusCode)
		}
	})
	t.Run("g1 forces post-processing off", func(t *testing.T) {
		status, text := postJSON(t, server, []byte(`{"prompt":"walk","model":"g1-rp-v1","post_processing":true}`))
		<-g.queue
		if status != http.StatusAccepted || !strings.Contains(text, `"post_processing":false`) {
			t.Fatalf("status %d: %s", status, text)
		}
	})
	before := len(g.items)
	for _, test := range []struct {
		name   string
		body   string
		status int
		err    string
	}{
		{"unknown constraint type", `{"prompt":"walk","model":"soma-rp-v1.1","constraints":[{"type":"hand","frame_indices":[1]}]}`, 400, "invalid constraints: constraints[0]: unknown type"},
		{"frame out of range", `{"prompt":"walk","frames":60,"model":"soma-rp-v1.1","constraints":[{"type":"root2d","frame_indices":[60],"smooth_root_2d":[[0,0]]}]}`, 400, "outside the 60 generated frames"},
		{"constraints not an array", `{"prompt":"walk","model":"soma-rp-v1.1","constraints":{"type":"root2d"}}`, 400, "JSON array"},
		{"length mismatch", `{"prompt":"walk","model":"soma-rp-v1.1","constraints":[{"type":"root2d","frame_indices":[1,2],"smooth_root_2d":[[0,0]]}]}`, 400, "smooth_root_2d"},
		{"segment frames", `{"segments":[{"prompt":"walk","frames":301}],"model":"soma-rp-v1.1"}`, 400, "60..300"},
		{"unknown model", `{"prompt":"walk","model":"nope"}`, 400, "unknown motion model"},
		{"validation before availability", `{"prompt":"walk","constraints":[{"type":"bogus","frame_indices":[0]}]}`, 400, "invalid constraints"},
		{"unavailable model", `{"prompt":"walk"}`, 409, "not available"},
		{"unavailable text", `{"prompt":"walk","model":"soma-rp-v1.1","text_quantization":"bf16"}`, 409, "not available"},
		{"invalid JSON", `{"prompt":`, 400, "invalid JSON"},
	} {
		t.Run(test.name, func(t *testing.T) {
			status, text := postJSON(t, server, []byte(test.body))
			if status != test.status || !strings.Contains(text, test.err) {
				t.Fatalf("status %d %q, want %d %q", status, text, test.status, test.err)
			}
		})
	}
	if len(g.items) != before {
		t.Fatalf("rejected requests created animations")
	}
	t.Run("body limit", func(t *testing.T) {
		// Served in-process: over a real connection the client may see the
		// early close before it finishes uploading.
		huge := append([]byte(`{"prompt":"walk","pad":"`), bytes.Repeat([]byte("x"), maxGenerateBody)...)
		huge = append(huge, `"}`...)
		recorder := httptest.NewRecorder()
		g.routes(t.TempDir(), embeddedWebFiles()).ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(huge)))
		if recorder.Code != http.StatusRequestEntityTooLarge {
			t.Fatalf("status %d: %s", recorder.Code, recorder.Body)
		}
	})
}

func TestAnimationFileEndpoints(t *testing.T) {
	g := newTestGallery(t)
	server := httptest.NewServer(g.routes(t.TempDir(), embeddedWebFiles()))
	defer server.Close()
	legacy, err := decodeAnimation([]byte(`{"id":"0123456789abcdef","prompt":"walk","frames":2,"status":"ready","model":"soma-rp-v1.1"}`))
	if err != nil {
		t.Fatal(err)
	}
	full := &animation{ID: "fedcba9876543210", Prompt: "walk", Frames: 2, Status: "ready", Model: "soma-rp-v1.1", TextCFG: new(2.0), ConstraintCFG: new(2.0), PostProcessing: new(false), RootMargin: new(.04), ConstraintTypes: []string{}}
	g.items[legacy.ID], g.items[full.ID] = legacy, full
	files := map[string]int{"root_positions.f32": 6, "local_rotations_xyzw.f32": 2 * 30 * 4, "foot_contacts.f32": 8, "smooth_root.f32": 6}
	for _, a := range []*animation{legacy, full} {
		if err := os.MkdirAll(filepath.Join(g.output, a.ID), 0755); err != nil {
			t.Fatal(err)
		}
		for name, count := range files {
			if a == legacy && (name == "foot_contacts.f32" || name == "smooth_root.f32") {
				continue // generated before the engine wrote contacts / smooth root
			}
			if err := writeF32(filepath.Join(g.output, a.ID, name), make([]float32, count)); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, test := range []struct {
		path   string
		status int
		bytes  int
	}{
		{"/api/animations/" + full.ID + "/root.f32", 200, 24},
		{"/api/animations/" + full.ID + "/rotations.f32", 200, 960},
		{"/api/animations/" + full.ID + "/contacts.f32", 200, 32},
		{"/api/animations/" + full.ID + "/smooth_root.f32", 200, 24},
		{"/api/animations/" + full.ID + "/constraints.json", 404, -1},
		{"/api/animations/" + legacy.ID + "/root.f32", 200, 24},
		{"/api/animations/" + legacy.ID + "/contacts.f32", 404, -1},
		{"/api/animations/" + legacy.ID + "/smooth_root.f32", 404, -1},
		{"/api/animations/" + full.ID + "/foot_contacts.f32", 404, -1},
		{"/api/animations/" + full.ID + "/local_rotations_xyzw.f32", 404, -1},
		{"/api/animations/ffffffffffffffff/contacts.f32", 404, -1},
		{"/api/animations/" + full.ID + "/contacts.f32/extra", 404, -1},
	} {
		response, body := get(t, server, test.path)
		if response.StatusCode != test.status || (test.bytes >= 0 && len(body) != test.bytes) {
			t.Fatalf("%s: %d (%d bytes), want %d (%d bytes)", test.path, response.StatusCode, len(body), test.status, test.bytes)
		}
		if test.status == 200 && (response.Header.Get("Cache-Control") != "no-store" || response.Header.Get("Content-Type") != "application/octet-stream") {
			t.Fatalf("%s: headers %v", test.path, response.Header)
		}
	}
	// Old records still list; their generation parameters stay absent so the
	// editor does not override its own settings with them.
	_, body := get(t, server, "/api/animations")
	var listed []map[string]any
	if err := json.Unmarshal(body, &listed); err != nil || len(listed) != 2 {
		t.Fatalf("list = %s (%v)", body, err)
	}
	for _, entry := range listed {
		_, hasCFG := entry["text_cfg"]
		_, hasPost := entry["post_processing"]
		_, hasMargin := entry["root_margin"]
		isFull := entry["id"] == full.ID
		if entry["constraint_count"] != 0.0 || hasCFG != isFull || hasPost != isFull || hasMargin != isFull || (isFull && (entry["text_cfg"] != 2.0 || entry["post_processing"] != false)) {
			t.Fatalf("listed entry = %v", entry)
		}
	}
}

func TestModelsEndpoint(t *testing.T) {
	g := newTestGallery(t)
	server := httptest.NewServer(g.routes(t.TempDir(), embeddedWebFiles()))
	defer server.Close()
	_, body := get(t, server, "/api/models")
	var models []struct {
		ID          string              `json:"id"`
		SkeletonKey string              `json:"skeleton_key"`
		Available   bool                `json:"available"`
		Parents     []int               `json:"parents"`
		Names       []string            `json:"names"`
		RootIndex   *int                `json:"root_index"`
		Hips        []int               `json:"hips"`
		Chains      map[string][]string `json:"chains"`
	}
	if err := json.Unmarshal(body, &models); err != nil || len(models) != 3 {
		t.Fatalf("models = %s (%v)", body, err)
	}
	hips := map[string][]int{"smplx22": {2, 1}, "soma30": {26, 22}, "g1skel34": {8, 1}}
	for _, model := range models {
		if model.RootIndex == nil || *model.RootIndex != 0 || len(model.Names) != len(model.Parents) || !slices.Equal(model.Hips, hips[model.SkeletonKey]) {
			t.Fatalf("%s: %s", model.ID, body)
		}
		for _, base := range []string{"LeftFoot", "RightFoot", "LeftHand", "RightHand"} {
			chain := model.Chains[base]
			if len(chain) == 0 {
				t.Fatalf("%s: missing chain %s", model.ID, base)
			}
			for _, name := range chain {
				if !slices.Contains(model.Names, name) {
					t.Fatalf("%s: chain %s names unknown joint %s", model.ID, base, name)
				}
			}
		}
	}
	if !strings.Contains(string(body), `"chains":{"LeftFoot":["LeftFoot","LeftToeBase"],"LeftHand":["LeftHand","LeftHandMiddleEnd"],"RightFoot":["RightFoot","RightToeBase"],"RightHand":["RightHand","RightHandMiddleEnd"]}`) {
		t.Fatalf("soma chains not as specified: %s", body)
	}
}

func TestStaticRoutes(t *testing.T) {
	g := newTestGallery(t)
	for _, mode := range []struct {
		name string
		web  webFiles
	}{{"embedded", embeddedWebFiles()}, {"web-dir", directoryWebFiles(".")}} {
		t.Run(mode.name, func(t *testing.T) {
			server := httptest.NewServer(g.routes(t.TempDir(), mode.web))
			defer server.Close()
			three, err := os.ReadFile(filepath.Join("assets", "three", "three.module.min.js"))
			if err != nil {
				t.Fatal(err)
			}
			for _, test := range []struct {
				path, contentType string
				status            int
			}{
				{"/", "text/html; charset=utf-8", 200},
				{"/models.js", "application/javascript; charset=utf-8", 200},
				{"/compare", "text/html; charset=utf-8", 200},
				{"/localai.png", "image/png", 200},
				{"/api/comparisons", "application/json", 200},
				{"/api/text-quantizations", "application/json", 200},
				{"/editor/README.md", "text/markdown; charset=utf-8", 200},
				{"/assets/three/three.module.min.js", "text/javascript; charset=utf-8", 200},
				{"/assets/three/OrbitControls.js", "text/javascript; charset=utf-8", 200},
				{"/assets/three/TransformControls.js", "text/javascript; charset=utf-8", 200},
				{"/assets/three/CSS2DRenderer.js", "text/javascript; charset=utf-8", 200},
				{"/editor/", "", 404},
				{"/editor/missing.js", "", 404},
				{"/assets/three/", "", 404},
				{"/assets/three/missing.js", "", 404},
				{"/editor/..%5Cmain.go", "", 404},
				{"/editor/C:%5CWindows%5Cwin.ini", "", 404},
				{"/assets/three/..%2F..%2Fmain.go", "", 404},
			} {
				response, body := get(t, server, test.path)
				if response.StatusCode != test.status || (test.status == 200 && response.Header.Get("Content-Type") != test.contentType) {
					t.Fatalf("%s: %d %q", test.path, response.StatusCode, response.Header.Get("Content-Type"))
				}
				if test.path == "/assets/three/three.module.min.js" && !bytes.Equal(body, three) {
					t.Fatalf("three.module.min.js differs from the vendored file")
				}
			}
			// Requests that bypass ServeMux path cleaning must not escape.
			for _, path := range []string{"/editor/../main.go", "/editor/./README.md", "/assets/three/../../main.go", "/editor/sub//x.js"} {
				recorder := httptest.NewRecorder()
				request := httptest.NewRequest(http.MethodGet, "/", nil)
				request.URL.Path = path
				handler := serveStatic(mode.web.editor, "/editor/", "no-store")
				if strings.HasPrefix(path, "/assets/three/") {
					handler = serveStatic(mode.web.three, "/assets/three/", "no-store")
				}
				handler(recorder, request)
				if recorder.Code != 404 {
					t.Fatalf("%s: %d", path, recorder.Code)
				}
			}
		})
	}
	// Every editor file on disk is embedded and served with a usable type.
	server := httptest.NewServer(g.routes(t.TempDir(), embeddedWebFiles()))
	defer server.Close()
	err := fs.WalkDir(os.DirFS("editor"), ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() || strings.HasPrefix(entry.Name(), ".") || strings.HasPrefix(entry.Name(), "_") {
			return err
		}
		response, body := get(t, server, "/editor/"+path)
		disk, readErr := os.ReadFile(filepath.Join("editor", filepath.FromSlash(path)))
		if response.StatusCode != 200 || readErr != nil || !bytes.Equal(body, disk) || response.Header.Get("Cache-Control") != "no-store" {
			t.Errorf("/editor/%s: %d %v", path, response.StatusCode, readErr)
		}
		if strings.HasSuffix(path, ".js") && response.Header.Get("Content-Type") != "text/javascript; charset=utf-8" {
			t.Errorf("/editor/%s: content type %q", path, response.Header.Get("Content-Type"))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestStaticContentType(t *testing.T) {
	for name, want := range map[string]string{"app.js": "text/javascript; charset=utf-8", "x.MJS": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8", "LICENSE": "text/plain; charset=utf-8", "a.bin": "application/octet-stream"} {
		if got := staticContentType(name); got != want {
			t.Fatalf("%s: %q, want %q", name, got, want)
		}
	}
}

// TestWorkerProtocolV2 runs the real worker against this test binary acting
// as the native generator (see TestMain).
func TestWorkerProtocolV2(t *testing.T) {
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
		<-done // the worker closes the generator session on exit
	}()
	generate := func(body string) *animation {
		t.Helper()
		status, text := postJSON(t, server, []byte(body))
		if status != http.StatusAccepted {
			t.Fatalf("status %d: %s", status, text)
		}
		var accepted animation
		if err := json.Unmarshal([]byte(text), &accepted); err != nil {
			t.Fatal(err)
		}
		deadline := time.Now().Add(30 * time.Second)
		for time.Now().Before(deadline) {
			g.mu.RLock()
			current := *g.items[accepted.ID]
			g.mu.RUnlock()
			if current.Status == "ready" || current.Status == "failed" {
				return &current
			}
			time.Sleep(20 * time.Millisecond)
		}
		t.Fatalf("animation %s did not finish", accepted.ID)
		return nil
	}
	pid := func(a *animation) string {
		b, _ := os.ReadFile(filepath.Join(g.output, a.ID, "generator.pid"))
		return string(b)
	}
	constrained := generate(`{"segments":[{"prompt":"walk forward","frames":60},{"prompt":"turn left","frames":61}],"model":"soma-rp-v1.1","seed":9,"steps":12,"transition_frames":4,
		"text_cfg":3,"constraint_cfg":1,"post_processing":true,"root_margin":0.02,"constraints":[{"type":"root2d","frame_indices":[0,120],"smooth_root_2d":[[0,0],[1,2]]}]}`)
	if constrained.Status != "ready" {
		t.Fatalf("constrained generation failed: %s", constrained.Error)
	}
	dir := filepath.Join(g.output, constrained.ID)
	line, err := os.ReadFile(filepath.Join(dir, "request.json"))
	if err != nil {
		t.Fatal(err)
	}
	var request generatorRequest
	if err := json.Unmarshal(line, &request); err != nil {
		t.Fatal(err)
	}
	if request.Output != protocolPath(dir) || request.ConstraintsFile != protocolPath(filepath.Join(dir, "constraints.json")) || request.Seed != 9 || request.Steps != 12 || request.TransitionFrames != 4 ||
		request.TextCFG != 3 || request.ConstraintCFG != 1 || !request.PostProcessing || request.RootMargin != .02 || len(request.Segments) != 2 || request.Segments[1].Frames != 61 {
		t.Fatalf("request = %s", line)
	}
	if prompt, err := os.ReadFile(request.Segments[1].PromptFile); err != nil || string(prompt) != "turn left" {
		t.Fatalf("segment prompt = %q (%v)", prompt, err)
	}
	for path, size := range map[string]int{"contacts.f32": 121 * 4 * 4, "smooth_root.f32": 121 * 3 * 4, "rotations.f32": 121 * 30 * 4 * 4} {
		if response, body := get(t, server, "/api/animations/"+constrained.ID+"/"+path); response.StatusCode != 200 || len(body) != size {
			t.Fatalf("%s: %d (%d bytes)", path, response.StatusCode, len(body))
		}
	}
	assertSkeletonGLB(t, filepath.Join(dir, "animation.glb"), 30)

	plain := generate(`{"prompt":"jump","frames":90,"model":"soma-rp-v1.1"}`)
	line, _ = os.ReadFile(filepath.Join(g.output, plain.ID, "request.json"))
	if plain.Status != "ready" || bytes.Contains(line, []byte("constraints_file")) || !bytes.Contains(line, []byte(`"text_cfg":2,"constraint_cfg":2,`)) || !bytes.Contains(line, []byte(`"post_processing":false`)) {
		t.Fatalf("plain request = %s (%s)", line, plain.Error)
	}
	if pid(plain) == "" || pid(plain) != pid(constrained) {
		t.Fatalf("generator session was not reused: %q vs %q", pid(plain), pid(constrained))
	}
	failed := generate(`{"prompt":"please fail","model":"soma-rp-v1.1"}`)
	if failed.Status != "failed" || !strings.Contains(failed.Error, "fake failure") {
		t.Fatalf("failed animation = %+v", failed)
	}
	// A failed request restarts the native session.
	after := generate(`{"prompt":"walk","model":"soma-rp-v1.1"}`)
	if after.Status != "ready" || pid(after) == "" || pid(after) == pid(plain) {
		t.Fatalf("session after failure: %s pid %q (previous %q)", after.Status, pid(after), pid(plain))
	}
}
