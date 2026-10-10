package main

// Browser assets: index.html, the constraint editor ES modules (demo/editor)
// and the vendored three.js build (demo/assets/three). They are embedded so the
// demo stays a single binary; -web-dir serves them from a demo/ checkout
// instead, which lets the editor be iterated on without rebuilding.

import (
	"bytes"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
)

// webFiles locates the browser assets; `files` (main.go) embeds index.html and
// the whole editor and three.js directories, so new modules need no change here.
type webFiles struct {
	index  func() ([]byte, error)
	editor fs.FS // rooted at demo/editor
	three  fs.FS // rooted at demo/assets/three
}

func embeddedWebFiles() webFiles {
	editor, err := fs.Sub(files, "editor")
	if err != nil {
		panic(err)
	}
	three, err := fs.Sub(files, "assets/three")
	if err != nil {
		panic(err)
	}
	return webFiles{index: func() ([]byte, error) { return files.ReadFile("index.html") }, editor: editor, three: three}
}

func directoryWebFiles(dir string) webFiles {
	return webFiles{
		index:  func() ([]byte, error) { return os.ReadFile(filepath.Join(dir, "index.html")) },
		editor: os.DirFS(filepath.Join(dir, "editor")),
		three:  os.DirFS(filepath.Join(dir, "assets", "three")),
	}
}

// Explicit types: on Windows the mime package consults the registry, which
// often maps .js to text/plain and makes browsers refuse ES modules.
var staticContentTypes = map[string]string{
	".js":   "text/javascript; charset=utf-8",
	".mjs":  "text/javascript; charset=utf-8",
	".css":  "text/css; charset=utf-8",
	".json": "application/json",
	".map":  "application/json",
	".html": "text/html; charset=utf-8",
	".md":   "text/markdown; charset=utf-8",
	".txt":  "text/plain; charset=utf-8",
	"":      "text/plain; charset=utf-8", // LICENSE
	".svg":  "image/svg+xml",
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".webp": "image/webp",
	".glb":  "model/gltf-binary",
	".wasm": "application/wasm",
}

func staticContentType(name string) string {
	if contentType, ok := staticContentTypes[strings.ToLower(path.Ext(name))]; ok {
		return contentType
	}
	return "application/octet-stream"
}

// serveStatic serves files below prefix from root. Every path element must be
// a plain file name (safePathPart), which rules out traversal, backslashes,
// drive letters and directory listings.
func serveStatic(root fs.FS, prefix, cacheControl string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "GET required", http.StatusMethodNotAllowed)
			return
		}
		name := strings.TrimPrefix(r.URL.Path, prefix)
		for _, part := range strings.Split(name, "/") {
			if !safePathPart(part) {
				http.NotFound(w, r)
				return
			}
		}
		data, err := fs.ReadFile(root, name)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", staticContentType(name))
		w.Header().Set("Cache-Control", cacheControl)
		w.Header().Set("X-Content-Type-Options", "nosniff")
		http.ServeContent(w, r, path.Base(name), time.Time{}, bytes.NewReader(data))
	}
}
