package main

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

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
