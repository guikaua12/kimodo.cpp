package main

// /api/generate request handling and the JSON (v2) generator protocol spoken
// with `kmd-generate --server MOTION TEXT`.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
)

const (
	maxGenerateBody      = 16 << 20 // constraint documents carry per-frame poses
	defaultTextCFG       = 2.0
	defaultConstraintCFG = 2.0
	defaultRootMargin    = 0.04
	minSegmentFrames     = 60
	maxSegmentFrames     = 300
	maxSegments          = 16
	maxCFG               = 10.0
	maxRootMargin        = 1.0
)

// generateRequest is the /api/generate body. Every field added for the
// constraint editor is optional: old clients omit them and get the legacy
// behaviour (CFG 2/2, no constraints, no post-processing). Pointers tell an
// explicit 0 apart from an absent key.
type generateRequest struct {
	Prompt           string          `json:"prompt"`
	Segments         []promptSegment `json:"segments"`
	TransitionFrames int             `json:"transition_frames"`
	Frames           int             `json:"frames"`
	Steps            int             `json:"steps"`
	Seed             uint64          `json:"seed"`
	Model            string          `json:"model"`
	TextQuantization string          `json:"text_quantization"`
	TextCFG          *float64        `json:"text_cfg"`
	ConstraintCFG    *float64        `json:"constraint_cfg"`
	PostProcessing   *bool           `json:"post_processing"`
	RootMargin       *float64        `json:"root_margin"`
	Constraints      json.RawMessage `json:"constraints"`
}

// normalize fills the defaults for omitted fields and validates everything
// that does not depend on the selected model.
func (request *generateRequest) normalize() error {
	request.Prompt = strings.TrimSpace(request.Prompt)
	if len(request.Segments) == 0 {
		request.Segments = []promptSegment{{Prompt: request.Prompt, Frames: request.Frames}}
	}
	if len(request.Segments) > maxSegments {
		return fmt.Errorf("at most %d prompt segments", maxSegments)
	}
	if request.Frames == 0 {
		request.Frames = 150
	}
	if request.Steps == 0 {
		request.Steps = 100
	}
	for index := range request.Segments {
		segment := &request.Segments[index]
		segment.Prompt = strings.TrimSpace(segment.Prompt)
		if segment.Frames == 0 {
			segment.Frames = 150
		}
		if segment.Prompt == "" || len(segment.Prompt) > 4096 || segment.Frames < minSegmentFrames || segment.Frames > maxSegmentFrames {
			return fmt.Errorf("each prompt segment must be %d..%d frames and 1..4096 bytes", minSegmentFrames, maxSegmentFrames)
		}
	}
	if request.TransitionFrames == 0 {
		request.TransitionFrames = 5
	}
	if request.TransitionFrames < 1 || request.TransitionFrames > 60 || request.Steps < 1 || request.Steps > 1000 {
		return fmt.Errorf("transition frames must be 1..60 and steps 1..1000")
	}
	if request.TextCFG == nil {
		request.TextCFG = new(float64(defaultTextCFG))
	}
	if request.ConstraintCFG == nil {
		request.ConstraintCFG = new(float64(defaultConstraintCFG))
	}
	if request.PostProcessing == nil {
		request.PostProcessing = new(false)
	}
	if request.RootMargin == nil {
		request.RootMargin = new(float64(defaultRootMargin))
	}
	if *request.TextCFG < 0 || *request.TextCFG > maxCFG || *request.ConstraintCFG < 0 || *request.ConstraintCFG > maxCFG {
		return fmt.Errorf("text_cfg and constraint_cfg must be 0..%g", maxCFG)
	}
	if *request.RootMargin < 0 || *request.RootMargin > maxRootMargin {
		return fmt.Errorf("root_margin must be 0..%g metres", maxRootMargin)
	}
	if request.Model == "" {
		request.Model = "smplx-rp-v1"
	}
	if request.TextQuantization == "" {
		request.TextQuantization = "q8_0"
	}
	return nil
}

// totalFrames is the length of the generated timeline (segments are
// contiguous; transitions eat into the following segment).
func (request *generateRequest) totalFrames() int {
	total := 0
	for _, segment := range request.Segments {
		total += segment.Frames
	}
	return total
}

// decodeAnimation reads a persisted gallery record. Records written before the
// constraint editor lack the CFG, post-processing and root-margin keys; those
// stay absent (nil) so selecting such a record does not override the editor's
// current settings. The first records (before the model selector) have no
// "model": they are SMPL-X RP v1, the only model then, and the first text
// encoder was BF16.
func decodeAnimation(b []byte) (*animation, error) {
	var a animation
	if err := json.Unmarshal(b, &a); err != nil {
		return nil, err
	}
	if a.Model == "" {
		a.Model = "smplx-rp-v1"
	}
	if a.TextQuantization == "" {
		a.TextQuantization = "bf16"
	}
	if a.ConstraintTypes == nil {
		a.ConstraintTypes = []string{}
	}
	return &a, nil
}

type generatorSegment struct {
	Frames     int    `json:"frames"`
	PromptFile string `json:"prompt_file"`
}

// generatorRequest is one v2 request of the native server protocol
// (docs/IMPLEMENTATION.md, Constraints: Requests). The seed is written as a
// decimal string: the engine's JSON reader stores numbers as doubles, which
// are exact only up to 2^53, and it parses a string seed as a full uint64.
type generatorRequest struct {
	Output           string             `json:"output"`
	Seed             uint64             `json:"seed,string"`
	Steps            int                `json:"steps"`
	TransitionFrames int                `json:"transition_frames"`
	TextCFG          float64            `json:"text_cfg"`
	ConstraintCFG    float64            `json:"constraint_cfg"`
	Segments         []generatorSegment `json:"segments"`
	ConstraintsFile  string             `json:"constraints_file,omitempty"`
	PostProcessing   bool               `json:"post_processing"`
	RootMargin       float64            `json:"root_margin"`
}

// valueOr dereferences an optional record field.
func valueOr[T any](value *T, fallback T) T {
	if value == nil {
		return fallback
	}
	return *value
}

// protocolPath makes a path absolute (the worker may run with another working
// directory one day) and uses forward slashes, which Windows accepts and which
// need no JSON escaping.
func protocolPath(path string) string {
	if absolute, err := filepath.Abs(path); err == nil {
		path = absolute
	}
	return filepath.ToSlash(path)
}

// generatorRequestLine encodes the newline-terminated JSON request for one
// animation. constraintsFile is empty when the animation has no constraints.
func generatorRequestLine(item *animation, dir string, segments []promptSegment, promptPaths []string, constraintsFile string) ([]byte, error) {
	if len(segments) == 0 || len(segments) != len(promptPaths) {
		return nil, fmt.Errorf("generator request needs one prompt file per segment")
	}
	request := generatorRequest{
		Output:           protocolPath(dir),
		Seed:             item.Seed,
		Steps:            item.DiffusionSteps,
		TransitionFrames: item.TransitionFrames,
		TextCFG:          valueOr(item.TextCFG, defaultTextCFG),
		ConstraintCFG:    valueOr(item.ConstraintCFG, defaultConstraintCFG),
		Segments:         make([]generatorSegment, len(segments)),
		PostProcessing:   valueOr(item.PostProcessing, false),
		RootMargin:       valueOr(item.RootMargin, defaultRootMargin),
	}
	for index, segment := range segments {
		request.Segments[index] = generatorSegment{Frames: segment.Frames, PromptFile: protocolPath(promptPaths[index])}
	}
	if constraintsFile != "" {
		request.ConstraintsFile = protocolPath(constraintsFile)
	}
	var line bytes.Buffer
	encoder := json.NewEncoder(&line)
	encoder.SetEscapeHTML(false)
	// Encode emits compact JSON (strings never contain raw newlines) followed
	// by exactly one '\n', i.e. one request per line.
	if err := encoder.Encode(request); err != nil {
		return nil, err
	}
	return line.Bytes(), nil
}
