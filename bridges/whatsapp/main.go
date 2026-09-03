// Package main implements a small WhatsApp bridge for the personal AI agent.
//
// It owns exactly one WhatsApp session (whatsmeow, the WhatsApp Web protocol)
// and exposes a tiny authenticated HTTP API that the Python agent calls:
//
//	GET  /health              liveness
//	GET  /status              connected / logged in / own JID
//	POST /login/qr            start pairing, return the QR code (ASCII + raw)
//	POST /logout              drop the session
//	POST /send/text           {"to","text"}            -> message_id
//	POST /send/file           {"to","path","caption"}  -> message_id
//	GET  /messages?limit=&since=  recent inbound messages
//
// Inbound messages are pushed to the agent's webhook and also kept in a small
// in-memory ring buffer so the agent can poll after a restart.
//
// Design notes:
//   - session state lives in a SQLite file on a mounted volume, so a container
//     restart does NOT require re-scanning the QR code
//   - every endpoint requires the shared BRIDGE_TOKEN
//   - media is written into the shared /data workspace, never outside it
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mdp/qrterminal/v3"
	"github.com/skip2/go-qrcode"
	"go.mau.fi/whatsmeow"
	waProto "go.mau.fi/whatsmeow/binary/proto"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"
	"google.golang.org/protobuf/proto"

	_ "github.com/mattn/go-sqlite3"
)

const (
	maxInboundBuffer = 200
	maxMediaBytes    = 64 << 20 // 64 MB
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type config struct {
	Port        string
	Token       string
	StorePath   string
	DataDir     string
	WebhookURL  string
	DownloadAll bool
}

func loadConfig() config {
	return config{
		Port:        envOr("BRIDGE_PORT", "8081"),
		Token:       os.Getenv("BRIDGE_TOKEN"),
		StorePath:   envOr("WA_STORE_PATH", "/session/whatsapp.db"),
		DataDir:     envOr("WORKSPACE_DIR", "/data"),
		WebhookURL:  os.Getenv("AGENT_WEBHOOK_URL"),
		DownloadAll: envOr("WA_DOWNLOAD_MEDIA", "true") == "true",
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ---------------------------------------------------------------------------
// Bridge
// ---------------------------------------------------------------------------

type inboundMessage struct {
	ID         string `json:"id"`
	Chat       string `json:"chat"`
	Sender     string `json:"sender"`
	SenderName string `json:"sender_name"`
	Text       string `json:"text"`
	MediaPath  string `json:"media_path,omitempty"`
	FromMe     bool   `json:"from_me"`
	Timestamp  int64  `json:"timestamp"`
}

type bridge struct {
	cfg    config
	client *whatsmeow.Client

	mu      sync.RWMutex
	inbound []inboundMessage

	qrMu   sync.Mutex
	lastQR string

	http *http.Client
}

func newBridge(cfg config) (*bridge, error) {
	if err := os.MkdirAll(filepath.Dir(cfg.StorePath), 0o755); err != nil {
		return nil, fmt.Errorf("create session dir: %w", err)
	}

	logger := waLog.Stdout("whatsmeow", "WARN", true)
	container, err := sqlstore.New(context.Background(), "sqlite3",
		"file:"+cfg.StorePath+"?_foreign_keys=on", logger)
	if err != nil {
		return nil, fmt.Errorf("open session store: %w", err)
	}

	device, err := container.GetFirstDevice(context.Background())
	if err != nil {
		return nil, fmt.Errorf("load device: %w", err)
	}

	b := &bridge{
		cfg:     cfg,
		client:  whatsmeow.NewClient(device, logger),
		inbound: make([]inboundMessage, 0, maxInboundBuffer),
		http:    &http.Client{Timeout: 30 * time.Second},
	}
	b.client.AddEventHandler(b.handleEvent)
	return b, nil
}

// connect starts the session. If the device is not paired yet it publishes a
// QR code that the owner scans from Telegram.
func (b *bridge) connect() error {
	if b.client.Store.ID == nil {
		qrChan, err := b.client.GetQRChannel(context.Background())
		if err != nil {
			return fmt.Errorf("qr channel: %w", err)
		}
		if err := b.client.Connect(); err != nil {
			return fmt.Errorf("connect: %w", err)
		}
		go func() {
			for evt := range qrChan {
				switch evt.Event {
				case "code":
					b.setQR(evt.Code)
					log.Println("whatsapp: waiting for QR scan")
				case "success":
					b.setQR("")
					log.Println("whatsapp: paired successfully")
				case "timeout":
					log.Println("whatsapp: QR timed out, request /login/qr again")
				}
			}
		}()
		return nil
	}
	return b.client.Connect()
}

func (b *bridge) setQR(code string) {
	b.qrMu.Lock()
	b.lastQR = code
	b.qrMu.Unlock()
}

func (b *bridge) getQR() string {
	b.qrMu.Lock()
	defer b.qrMu.Unlock()
	return b.lastQR
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

func (b *bridge) handleEvent(rawEvt interface{}) {
	switch evt := rawEvt.(type) {
	case *events.Message:
		b.onMessage(evt)
	case *events.Connected:
		log.Println("whatsapp: connected")
	case *events.LoggedOut:
		log.Println("whatsapp: logged out - re-pair with /login/qr")
	case *events.Disconnected:
		log.Println("whatsapp: disconnected, whatsmeow will retry")
	}
}

func (b *bridge) onMessage(evt *events.Message) {
	text := extractText(evt.Message)

	msg := inboundMessage{
		ID:         evt.Info.ID,
		Chat:       evt.Info.Chat.String(),
		Sender:     evt.Info.Sender.User,
		SenderName: evt.Info.PushName,
		Text:       text,
		FromMe:     evt.Info.IsFromMe,
		Timestamp:  evt.Info.Timestamp.Unix(),
	}

	if b.cfg.DownloadAll {
		if path, err := b.downloadMedia(evt); err != nil {
			log.Printf("whatsapp: media download failed: %v", err)
		} else if path != "" {
			msg.MediaPath = path
		}
	}

	if msg.Text == "" && msg.MediaPath == "" {
		return // protocol message, reaction, etc.
	}

	b.mu.Lock()
	b.inbound = append(b.inbound, msg)
	if len(b.inbound) > maxInboundBuffer {
		b.inbound = b.inbound[len(b.inbound)-maxInboundBuffer:]
	}
	b.mu.Unlock()

	if !msg.FromMe {
		b.pushWebhook(msg)
	}
}

func extractText(m *waProto.Message) string {
	if m == nil {
		return ""
	}
	if t := m.GetConversation(); t != "" {
		return t
	}
	if ext := m.GetExtendedTextMessage(); ext != nil {
		return ext.GetText()
	}
	if img := m.GetImageMessage(); img != nil {
		return img.GetCaption()
	}
	if vid := m.GetVideoMessage(); vid != nil {
		return vid.GetCaption()
	}
	if doc := m.GetDocumentMessage(); doc != nil {
		if c := doc.GetCaption(); c != "" {
			return c
		}
		return doc.GetFileName()
	}
	return ""
}

// downloadMedia saves any attached media into the shared workspace.
func (b *bridge) downloadMedia(evt *events.Message) (string, error) {
	var (
		downloadable whatsmeow.DownloadableMessage
		extension    string
		size         uint64
	)

	switch {
	case evt.Message.GetImageMessage() != nil:
		m := evt.Message.GetImageMessage()
		downloadable, extension, size = m, extFromMime(m.GetMimetype(), ".jpg"), m.GetFileLength()
	case evt.Message.GetVideoMessage() != nil:
		m := evt.Message.GetVideoMessage()
		downloadable, extension, size = m, extFromMime(m.GetMimetype(), ".mp4"), m.GetFileLength()
	case evt.Message.GetAudioMessage() != nil:
		m := evt.Message.GetAudioMessage()
		downloadable, extension, size = m, extFromMime(m.GetMimetype(), ".ogg"), m.GetFileLength()
	case evt.Message.GetDocumentMessage() != nil:
		m := evt.Message.GetDocumentMessage()
		ext := filepath.Ext(m.GetFileName())
		if ext == "" {
			ext = extFromMime(m.GetMimetype(), ".bin")
		}
		downloadable, extension, size = m, ext, m.GetFileLength()
	default:
		return "", nil
	}

	if size > maxMediaBytes {
		return "", fmt.Errorf("media too large: %d bytes", size)
	}

	data, err := b.client.Download(context.Background(), downloadable)
	if err != nil {
		return "", err
	}

	dir := filepath.Join(b.cfg.DataDir, "downloads", "whatsapp")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("%s_%s%s", time.Now().UTC().Format("20060102T150405"),
		sanitize(evt.Info.ID), extension)
	full := filepath.Join(dir, name)
	if err := os.WriteFile(full, data, 0o644); err != nil {
		return "", err
	}
	// Return a workspace-relative path: the agent's sandbox understands that.
	return filepath.ToSlash(filepath.Join("downloads", "whatsapp", name)), nil
}

func extFromMime(mimeType, fallback string) string {
	if mimeType == "" {
		return fallback
	}
	if exts, err := mime.ExtensionsByType(strings.Split(mimeType, ";")[0]); err == nil && len(exts) > 0 {
		return exts[0]
	}
	return fallback
}

func sanitize(s string) string {
	var out strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			out.WriteRune(r)
		}
	}
	result := out.String()
	if len(result) > 40 {
		result = result[:40]
	}
	if result == "" {
		result = "media"
	}
	return result
}

func (b *bridge) pushWebhook(msg inboundMessage) {
	if b.cfg.WebhookURL == "" {
		return
	}
	payload, err := json.Marshal(map[string]any{"channel": "whatsapp", "message": msg})
	if err != nil {
		return
	}
	req, err := http.NewRequest(http.MethodPost, b.cfg.WebhookURL, bytes.NewReader(payload))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Bridge-Token", b.cfg.Token)

	resp, err := b.http.Do(req)
	if err != nil {
		log.Printf("whatsapp: webhook failed: %v", err)
		return
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
}

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------

func parseJID(raw string) (types.JID, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return types.JID{}, errors.New("empty recipient")
	}
	if strings.Contains(raw, "@") {
		return types.ParseJID(raw)
	}
	digits := strings.Map(func(r rune) rune {
		if r >= '0' && r <= '9' {
			return r
		}
		return -1
	}, raw)
	if len(digits) < 8 {
		return types.JID{}, fmt.Errorf("invalid phone number: %q", raw)
	}
	return types.NewJID(digits, types.DefaultUserServer), nil
}

func (b *bridge) sendText(to, text string) (string, error) {
	jid, err := parseJID(to)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(text) == "" {
		return "", errors.New("empty message text")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := b.client.SendMessage(ctx, jid, &waProto.Message{
		Conversation: proto.String(text),
	})
	if err != nil {
		return "", err
	}
	return resp.ID, nil
}

func (b *bridge) sendFile(to, relPath, caption string) (string, error) {
	jid, err := parseJID(to)
	if err != nil {
		return "", err
	}

	full, err := b.safePath(relPath)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(full)
	if err != nil {
		return "", fmt.Errorf("file not found: %s", relPath)
	}
	if info.Size() == 0 {
		return "", errors.New("refusing to send an empty file")
	}
	if info.Size() > maxMediaBytes {
		return "", fmt.Errorf("file too large (%d bytes, max %d)", info.Size(), int64(maxMediaBytes))
	}

	data, err := os.ReadFile(full)
	if err != nil {
		return "", err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	uploaded, err := b.client.Upload(ctx, data, whatsmeow.MediaDocument)
	if err != nil {
		return "", fmt.Errorf("upload: %w", err)
	}

	mimeType := mime.TypeByExtension(filepath.Ext(full))
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}

	resp, err := b.client.SendMessage(ctx, jid, &waProto.Message{
		DocumentMessage: &waProto.DocumentMessage{
			URL:           proto.String(uploaded.URL),
			DirectPath:    proto.String(uploaded.DirectPath),
			MediaKey:      uploaded.MediaKey,
			FileEncSHA256: uploaded.FileEncSHA256,
			FileSHA256:    uploaded.FileSHA256,
			FileLength:    proto.Uint64(uint64(len(data))),
			Mimetype:      proto.String(mimeType),
			FileName:      proto.String(filepath.Base(full)),
			Caption:       proto.String(caption),
		},
	})
	if err != nil {
		return "", err
	}
	return resp.ID, nil
}

// safePath keeps every file operation inside the shared workspace.
func (b *bridge) safePath(rel string) (string, error) {
	root, err := filepath.Abs(b.cfg.DataDir)
	if err != nil {
		return "", err
	}
	candidate := rel
	if !filepath.IsAbs(candidate) {
		candidate = filepath.Join(root, rel)
	}
	resolved, err := filepath.Abs(candidate)
	if err != nil {
		return "", err
	}
	if resolved != root && !strings.HasPrefix(resolved, root+string(os.PathSeparator)) {
		return "", fmt.Errorf("path escapes workspace: %s", rel)
	}
	return resolved, nil
}

// ---------------------------------------------------------------------------
// HTTP API
// ---------------------------------------------------------------------------

func (b *bridge) routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "service": "whatsapp-bridge"})
	})

	mux.HandleFunc("/status", b.auth(func(w http.ResponseWriter, r *http.Request) {
		jid := ""
		if b.client.Store.ID != nil {
			jid = b.client.Store.ID.String()
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"connected":  b.client.IsConnected(),
			"logged_in":  b.client.IsLoggedIn(),
			"jid":        jid,
			"qr_pending": b.getQR() != "",
		})
	}))

	mux.HandleFunc("/login/qr", b.auth(func(w http.ResponseWriter, r *http.Request) {
		if b.client.IsLoggedIn() {
			writeJSON(w, http.StatusOK, map[string]any{
				"logged_in": true, "message": "already paired",
			})
			return
		}
		code := b.getQR()
		if code == "" {
			// Not connected yet (or QR expired): restart pairing.
			b.client.Disconnect()
			if err := b.connect(); err != nil {
				writeError(w, http.StatusInternalServerError, err.Error())
				return
			}
			for i := 0; i < 40 && code == ""; i++ {
				time.Sleep(250 * time.Millisecond)
				code = b.getQR()
			}
		}
		if code == "" {
			writeError(w, http.StatusServiceUnavailable, "no QR code available yet, retry shortly")
			return
		}
		var ascii bytes.Buffer
		qrterminal.GenerateHalfBlock(code, qrterminal.L, &ascii)
		writeJSON(w, http.StatusOK, map[string]any{
			"logged_in": false,
			"qr":        code,
			"qr_ascii":  ascii.String(),
		})
	}))

	// PNG version of the pairing code, so Telegram can display a scannable image.
	mux.HandleFunc("/login/qr.png", b.auth(func(w http.ResponseWriter, r *http.Request) {
		if b.client.IsLoggedIn() {
			writeError(w, http.StatusConflict, "already paired")
			return
		}
		code := b.getQR()
		if code == "" {
			b.client.Disconnect()
			if err := b.connect(); err != nil {
				writeError(w, http.StatusInternalServerError, err.Error())
				return
			}
			for i := 0; i < 40 && code == ""; i++ {
				time.Sleep(250 * time.Millisecond)
				code = b.getQR()
			}
		}
		if code == "" {
			writeError(w, http.StatusServiceUnavailable, "no QR code available yet, retry shortly")
			return
		}
		png, err := qrcode.Encode(code, qrcode.Medium, 512)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "could not render QR: "+err.Error())
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusOK)
		w.Write(png)
	}))

	mux.HandleFunc("/logout", b.auth(func(w http.ResponseWriter, r *http.Request) {
		if err := b.client.Logout(context.Background()); err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"logged_out": true})
	}))

	mux.HandleFunc("/send/text", b.auth(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			To   string `json:"to"`
			Text string `json:"text"`
		}
		if !decode(w, r, &body) {
			return
		}
		if !b.client.IsLoggedIn() {
			writeError(w, http.StatusServiceUnavailable, "whatsapp is not linked; use /login/qr")
			return
		}
		id, err := b.sendText(body.To, body.Text)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"sent": true, "message_id": id, "to": body.To})
	}))

	mux.HandleFunc("/send/file", b.auth(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			To      string `json:"to"`
			Path    string `json:"path"`
			Caption string `json:"caption"`
		}
		if !decode(w, r, &body) {
			return
		}
		if !b.client.IsLoggedIn() {
			writeError(w, http.StatusServiceUnavailable, "whatsapp is not linked; use /login/qr")
			return
		}
		id, err := b.sendFile(body.To, body.Path, body.Caption)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"sent": true, "message_id": id, "to": body.To, "path": body.Path,
		})
	}))

	mux.HandleFunc("/messages", b.auth(func(w http.ResponseWriter, r *http.Request) {
		limit := 20
		if raw := r.URL.Query().Get("limit"); raw != "" {
			if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 && parsed <= maxInboundBuffer {
				limit = parsed
			}
		}
		var since int64
		if raw := r.URL.Query().Get("since"); raw != "" {
			since, _ = strconv.ParseInt(raw, 10, 64)
		}

		b.mu.RLock()
		all := make([]inboundMessage, len(b.inbound))
		copy(all, b.inbound)
		b.mu.RUnlock()

		filtered := make([]inboundMessage, 0, limit)
		for i := len(all) - 1; i >= 0 && len(filtered) < limit; i-- {
			if since > 0 && all[i].Timestamp <= since {
				continue
			}
			filtered = append(filtered, all[i])
		}
		writeJSON(w, http.StatusOK, map[string]any{"count": len(filtered), "messages": filtered})
	}))

	return mux
}

func (b *bridge) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if b.cfg.Token == "" {
			writeError(w, http.StatusServiceUnavailable, "BRIDGE_TOKEN is not configured")
			return
		}
		if r.Header.Get("X-Bridge-Token") != b.cfg.Token {
			writeError(w, http.StatusUnauthorized, "invalid bridge token")
			return
		}
		next(w, r)
	}
}

func decode(w http.ResponseWriter, r *http.Request, target any) bool {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return false
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(target); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]any{"ok": false, "error": message})
}

// ---------------------------------------------------------------------------

func main() {
	cfg := loadConfig()
	if cfg.Token == "" {
		log.Fatal("BRIDGE_TOKEN must be set")
	}

	b, err := newBridge(cfg)
	if err != nil {
		log.Fatalf("bridge init: %v", err)
	}
	if err := b.connect(); err != nil {
		log.Printf("initial connect failed (will retry on demand): %v", err)
	}

	server := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           b.routes(),
		ReadHeaderTimeout: 15 * time.Second,
	}
	log.Printf("whatsapp bridge listening on :%s", cfg.Port)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("http server: %v", err)
	}
}
