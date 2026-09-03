// Package main implements a Microsoft Teams bridge for the personal AI agent.
//
// It talks to Microsoft Graph over plain HTTPS - no SDK, no browser
// automation - using the official application-permission (client credentials)
// flow. This satisfies the "API-first, no fragile automation" rule.
//
// HTTP API consumed by the Python agent:
//
//	GET  /health                       liveness
//	GET  /status                       token state + configured tenant
//	POST /send/text   {"chat","text"}  send to a chat or channel
//	GET  /messages?chat=&limit=        read recent messages
//	GET  /chats?limit=                 list chats the app can see
//	POST /download    {"message_id","chat","path"}  save a hosted attachment
//
// Addressing:
//   - chat id  ("19:abc...@thread.v2")            -> /chats/{id}/messages
//   - team/channel ("teamId/channelId")           -> /teams/{t}/channels/{c}/messages
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	graphBase     = "https://graph.microsoft.com/v1.0"
	graphScope    = "https://graph.microsoft.com/.default"
	maxDownload   = 64 << 20
	tokenLeeway   = 60 * time.Second
	requestExpiry = 60 * time.Second
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type config struct {
	Port         string
	Token        string
	TenantID     string
	ClientID     string
	ClientSecret string
	DataDir      string
	WebhookURL   string
	DefaultChat  string
}

// authMode reports how this bridge talks to Graph.
//
//	delegated - device-code sign-in: the agent acts AS the owner. No client
//	            secret, no admin consent, MFA works normally.
//	app       - client credentials: the agent acts as itself. Needs a secret
//	            and admin consent on protected APIs.
func (c config) authMode() string {
	if c.ClientSecret != "" {
		return "app"
	}
	return "delegated"
}

// Scopes requested during device-code sign-in. offline_access is what gives us
// a refresh token, so the owner signs in once and stays signed in.
const deviceScopes = "offline_access User.Read Chat.ReadWrite ChannelMessage.Send ChannelMessage.Read.All"

// Microsoft Graph Command Line Tools: the public client Microsoft ships for
// delegated Graph access (this is what Graph PowerShell uses). It is
// preauthorized for Graph, so device-code sign-in works without registering an
// app. The Azure CLI id is NOT preauthorized for Graph and fails with
// AADSTS65002. Override with TEAMS_CLIENT_ID if your tenant blocks this one.
const defaultPublicClientID = "14d82eec-204b-4c2f-b7e8-296a70dab67e"

func loadConfig() config {
	return config{
		Port:         envOr("BRIDGE_PORT", "8082"),
		Token:        os.Getenv("BRIDGE_TOKEN"),
		TenantID:     envOr("TEAMS_TENANT_ID", "organizations"),
		ClientID:     envOr("TEAMS_CLIENT_ID", defaultPublicClientID),
		ClientSecret: os.Getenv("TEAMS_CLIENT_SECRET"),
		DataDir:      envOr("WORKSPACE_DIR", "/data"),
		WebhookURL:   os.Getenv("AGENT_WEBHOOK_URL"),
		DefaultChat:  os.Getenv("TEAMS_DEFAULT_CHAT"),
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func (c config) configured() bool {
	if c.TenantID == "" || c.ClientID == "" {
		return false
	}
	// App mode needs a secret; delegated mode needs a signed-in user instead.
	return c.authMode() == "delegated" || c.ClientSecret != ""
}

// ---------------------------------------------------------------------------
// Graph client
// ---------------------------------------------------------------------------

type bridge struct {
	http *http.Client

	mu           sync.Mutex
	cfg          config
	accessToken  string
	refreshToken string
	expiresAt    time.Time
	account      string

	devMu   sync.Mutex
	pending *deviceLogin
}

// deviceLogin holds an in-flight device-code sign-in.
type deviceLogin struct {
	DeviceCode string
	UserCode   string
	VerifyURL  string
	Interval   int
	ExpiresAt  time.Time
}

// credsPath is where runtime credentials (set from Telegram) are persisted so
// the bridge stays configured across restarts.
func credsPath(cfg config) string {
	return filepath.Join(cfg.DataDir, ".teams_creds.json")
}

type storedCreds struct {
	TenantID     string `json:"tenant_id"`
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"client_secret"`
	DefaultChat  string `json:"default_chat"`
	RefreshToken string `json:"refresh_token"`
	Account      string `json:"account"`
}

func (b *bridge) snapshot() config {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.cfg
}

// loadCreds restores credentials written by a previous /config call.
func (b *bridge) loadCreds() {
	b.mu.Lock()
	path := credsPath(b.cfg)
	b.mu.Unlock()

	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var creds storedCreds
	if err := json.Unmarshal(data, &creds); err != nil {
		log.Printf("teams: stored credentials unreadable: %v", err)
		return
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if creds.TenantID != "" {
		b.cfg.TenantID = creds.TenantID
	}
	if creds.ClientID != "" {
		b.cfg.ClientID = creds.ClientID
	}
	if creds.ClientSecret != "" {
		b.cfg.ClientSecret = creds.ClientSecret
	}
	if creds.DefaultChat != "" {
		b.cfg.DefaultChat = creds.DefaultChat
	}
	if creds.RefreshToken != "" {
		b.refreshToken = creds.RefreshToken
		b.account = creds.Account
	}
	log.Println("teams: restored stored credentials")
}

// saveCreds persists the current credentials with owner-only permissions.
func (b *bridge) saveCreds() error {
	b.mu.Lock()
	creds := storedCreds{
		TenantID:     b.cfg.TenantID,
		ClientID:     b.cfg.ClientID,
		ClientSecret: b.cfg.ClientSecret,
		DefaultChat:  b.cfg.DefaultChat,
		RefreshToken: b.refreshToken,
		Account:      b.account,
	}
	path := credsPath(b.cfg)
	b.mu.Unlock()

	data, err := json.Marshal(creds)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

func newBridge(cfg config) *bridge {
	return &bridge{
		cfg:  cfg,
		http: &http.Client{Timeout: 60 * time.Second},
	}
}

func (b *bridge) tokenEndpoint() string {
	tenant := b.cfg.TenantID
	if tenant == "" {
		tenant = "organizations"
	}
	return fmt.Sprintf("https://login.microsoftonline.com/%s/oauth2/v2.0/token", tenant)
}

type tokenResponse struct {
	AccessToken      string `json:"access_token"`
	RefreshToken     string `json:"refresh_token"`
	ExpiresIn        int    `json:"expires_in"`
	Error            string `json:"error"`
	ErrorDescription string `json:"error_description"`
}

// postForm performs an OAuth form POST against the token endpoint.
func (b *bridge) postForm(ctx context.Context, endpoint string, form url.Values) (*tokenResponse, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint,
		strings.NewReader(form.Encode()))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := b.http.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("token request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var payload tokenResponse
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, resp.StatusCode, fmt.Errorf("decode token response: %s",
			truncate(string(body), 200))
	}
	return &payload, resp.StatusCode, nil
}

// token returns a valid Graph access token for whichever auth mode is active.
// Caller must not hold b.mu.
func (b *bridge) token(ctx context.Context) (string, error) {
	b.mu.Lock()
	if b.accessToken != "" && time.Now().Add(tokenLeeway).Before(b.expiresAt) {
		token := b.accessToken
		b.mu.Unlock()
		return token, nil
	}
	cfg := b.cfg
	refresh := b.refreshToken
	b.mu.Unlock()

	if !cfg.configured() {
		return "", errors.New("teams credentials are not configured")
	}

	var form url.Values
	switch {
	case cfg.authMode() == "app":
		form = url.Values{
			"client_id":     {cfg.ClientID},
			"client_secret": {cfg.ClientSecret},
			"scope":         {graphScope},
			"grant_type":    {"client_credentials"},
		}
	case refresh != "":
		form = url.Values{
			"client_id":     {cfg.ClientID},
			"scope":         {deviceScopes},
			"grant_type":    {"refresh_token"},
			"refresh_token": {refresh},
		}
	default:
		return "", errors.New("not signed in; start device sign-in with /login")
	}

	payload, status, err := b.postForm(ctx, b.tokenEndpoint(), form)
	if err != nil {
		return "", err
	}
	if status != http.StatusOK || payload.AccessToken == "" {
		detail := payload.ErrorDescription
		if detail == "" {
			detail = payload.Error
		}
		// A dead refresh token means the owner must sign in again.
		if payload.Error == "invalid_grant" {
			b.mu.Lock()
			b.refreshToken = ""
			b.mu.Unlock()
			return "", errors.New("sign-in expired; run /teams login again")
		}
		return "", fmt.Errorf("token endpoint returned %d: %s", status, truncate(detail, 200))
	}

	b.mu.Lock()
	b.accessToken = payload.AccessToken
	b.expiresAt = time.Now().Add(time.Duration(payload.ExpiresIn) * time.Second)
	if payload.RefreshToken != "" {
		b.refreshToken = payload.RefreshToken
	}
	token := b.accessToken
	b.mu.Unlock()

	if payload.RefreshToken != "" {
		if err := b.saveCreds(); err != nil {
			log.Printf("teams: could not persist refresh token: %v", err)
		}
	}
	return token, nil
}

// startDeviceLogin asks Microsoft for a user code the owner types in a browser.
// explainAuthError turns Microsoft's raw AADSTS text into something the owner
// can act on. The underlying detail is kept so nothing is hidden.
func explainAuthError(detail string) string {
	lower := strings.ToLower(detail)
	switch {
	case strings.Contains(lower, "personal account") ||
		strings.Contains(lower, "aadsts50020") ||
		strings.Contains(lower, "consumer"):
		return "this is a personal Microsoft account (outlook/hotmail/live). " +
			"Teams is only exposed to work or school accounts, so a personal " +
			"account cannot be connected. " + truncate(detail, 160)
	case strings.Contains(lower, "aadsts65002") || strings.Contains(lower, "preauthorization"):
		return "this tenant blocks the built-in sign-in app. Register your own " +
			"app in portal.azure.com (public client flows = Yes) and retry with " +
			"/teams login <client_id>. " + truncate(detail, 160)
	case strings.Contains(lower, "aadsts50059") || strings.Contains(lower, "tenant-identifying"):
		return "no tenant could be determined for that account. Retry with your " +
			"organisation domain: /teams login <yourcompany.com>. " + truncate(detail, 160)
	case strings.Contains(lower, "aadsts7000218") || strings.Contains(lower, "client_assertion"):
		return "the app registration is not marked as a public client. In Azure, " +
			"Authentication -> Allow public client flows -> Yes. " + truncate(detail, 160)
	default:
		return truncate(detail, 250)
	}
}

func (b *bridge) startDeviceLogin(ctx context.Context) (*deviceLogin, error) {
	cfg := b.snapshot()
	if cfg.ClientID == "" {
		return nil, errors.New("no client id configured")
	}

	tenant := cfg.TenantID
	if tenant == "" {
		tenant = "organizations"
	}
	endpoint := fmt.Sprintf(
		"https://login.microsoftonline.com/%s/oauth2/v2.0/devicecode", tenant)

	form := url.Values{"client_id": {cfg.ClientID}, "scope": {deviceScopes}}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint,
		strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := b.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("device code request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var payload struct {
		DeviceCode       string `json:"device_code"`
		UserCode         string `json:"user_code"`
		VerificationURI  string `json:"verification_uri"`
		ExpiresIn        int    `json:"expires_in"`
		Interval         int    `json:"interval"`
		Error            string `json:"error"`
		ErrorDescription string `json:"error_description"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("decode device code: %s", truncate(string(body), 200))
	}
	if payload.Error != "" || payload.DeviceCode == "" {
		detail := payload.ErrorDescription
		if detail == "" {
			detail = payload.Error
		}
		return nil, fmt.Errorf("device code refused: %s", explainAuthError(detail))
	}

	interval := payload.Interval
	if interval < 1 {
		interval = 5
	}
	verify := payload.VerificationURI
	if verify == "" {
		verify = "https://microsoft.com/devicelogin"
	}

	login := &deviceLogin{
		DeviceCode: payload.DeviceCode,
		UserCode:   payload.UserCode,
		VerifyURL:  verify,
		Interval:   interval,
		ExpiresAt:  time.Now().Add(time.Duration(payload.ExpiresIn) * time.Second),
	}

	b.devMu.Lock()
	b.pending = login
	b.devMu.Unlock()

	log.Println("teams: device sign-in started, waiting for the owner")
	return login, nil
}

// pollDeviceLogin checks once whether the owner has finished signing in.
// Returns (done, error). done=false with a nil error means "still waiting".
func (b *bridge) pollDeviceLogin(ctx context.Context) (bool, error) {
	b.devMu.Lock()
	login := b.pending
	b.devMu.Unlock()

	if login == nil {
		return false, errors.New("no sign-in in progress")
	}
	if time.Now().After(login.ExpiresAt) {
		b.devMu.Lock()
		b.pending = nil
		b.devMu.Unlock()
		return false, errors.New("the code expired; start again")
	}

	cfg := b.snapshot()
	form := url.Values{
		"client_id":   {cfg.ClientID},
		"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
		"device_code": {login.DeviceCode},
	}
	payload, status, err := b.postForm(ctx, b.tokenEndpoint(), form)
	if err != nil {
		return false, err
	}

	if status == http.StatusOK && payload.AccessToken != "" {
		b.mu.Lock()
		b.accessToken = payload.AccessToken
		b.refreshToken = payload.RefreshToken
		b.expiresAt = time.Now().Add(time.Duration(payload.ExpiresIn) * time.Second)
		b.mu.Unlock()

		b.devMu.Lock()
		b.pending = nil
		b.devMu.Unlock()

		if name, err := b.fetchAccount(ctx); err == nil {
			b.mu.Lock()
			b.account = name
			b.mu.Unlock()
		}
		if err := b.saveCreds(); err != nil {
			log.Printf("teams: could not persist sign-in: %v", err)
		}
		log.Println("teams: device sign-in complete")
		return true, nil
	}

	switch payload.Error {
	case "authorization_pending":
		return false, nil
	case "slow_down":
		b.devMu.Lock()
		if b.pending != nil {
			b.pending.Interval += 5
		}
		b.devMu.Unlock()
		return false, nil
	case "authorization_declined":
		b.devMu.Lock()
		b.pending = nil
		b.devMu.Unlock()
		return false, errors.New("sign-in was declined")
	case "expired_token":
		b.devMu.Lock()
		b.pending = nil
		b.devMu.Unlock()
		return false, errors.New("the code expired; start again")
	}

	detail := payload.ErrorDescription
	if detail == "" {
		detail = payload.Error
	}
	return false, fmt.Errorf("sign-in failed: %s", explainAuthError(detail))
}

// fetchAccount reads the signed-in user's display name / UPN.
func (b *bridge) fetchAccount(ctx context.Context) (string, error) {
	data, _, err := b.graph(ctx, http.MethodGet, "/me", nil)
	if err != nil {
		return "", err
	}
	var me struct {
		DisplayName       string `json:"displayName"`
		UserPrincipalName string `json:"userPrincipalName"`
	}
	if err := json.Unmarshal(data, &me); err != nil {
		return "", err
	}
	if me.UserPrincipalName != "" {
		return me.UserPrincipalName, nil
	}
	return me.DisplayName, nil
}

// graph performs an authenticated Graph call and returns the raw body.
func (b *bridge) graph(ctx context.Context, method, path string, body any) ([]byte, int, error) {
	token, err := b.token(ctx)
	if err != nil {
		return nil, 0, err
	}

	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
		reader = bytes.NewReader(encoded)
	}

	endpoint := path
	if !strings.HasPrefix(path, "http") {
		endpoint = graphBase + path
	}

	req, err := http.NewRequestWithContext(ctx, method, endpoint, reader)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := b.http.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxDownload))
	if err != nil {
		return nil, resp.StatusCode, err
	}

	if resp.StatusCode == http.StatusUnauthorized {
		// Force a token refresh on the next call.
		b.mu.Lock()
		b.accessToken = ""
		b.mu.Unlock()
	}
	if resp.StatusCode >= 400 {
		return data, resp.StatusCode, fmt.Errorf("graph %s %s -> %d: %s",
			method, path, resp.StatusCode, truncate(graphError(data), 300))
	}
	return data, resp.StatusCode, nil
}

func graphError(data []byte) string {
	var payload struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &payload); err == nil && payload.Error.Message != "" {
		return payload.Error.Code + ": " + payload.Error.Message
	}
	return string(data)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// ---------------------------------------------------------------------------
// Message helpers
// ---------------------------------------------------------------------------

// messagePath maps a friendly target to the Graph messages collection.
// "teamId/channelId" -> channel, anything else -> chat.
func messagePath(target string) (string, error) {
	target = strings.TrimSpace(target)
	if target == "" {
		return "", errors.New("empty chat/channel id")
	}
	if strings.Count(target, "/") == 1 {
		parts := strings.SplitN(target, "/", 2)
		if parts[0] == "" || parts[1] == "" {
			return "", fmt.Errorf("invalid team/channel target: %q", target)
		}
		return fmt.Sprintf("/teams/%s/channels/%s/messages",
			url.PathEscape(parts[0]), url.PathEscape(parts[1])), nil
	}
	return "/chats/" + url.PathEscape(target) + "/messages", nil
}

type teamsMessage struct {
	ID         string `json:"id"`
	Chat       string `json:"chat"`
	Sender     string `json:"sender"`
	SenderName string `json:"sender_name"`
	Text       string `json:"text"`
	CreatedAt  string `json:"created_at"`
	HasFiles   bool   `json:"has_files"`
}

func parseMessages(data []byte, chat string) []teamsMessage {
	var payload struct {
		Value []struct {
			ID              string `json:"id"`
			CreatedDateTime string `json:"createdDateTime"`
			Body            struct {
				Content     string `json:"content"`
				ContentType string `json:"contentType"`
			} `json:"body"`
			From struct {
				User struct {
					ID          string `json:"id"`
					DisplayName string `json:"displayName"`
				} `json:"user"`
			} `json:"from"`
			Attachments []struct {
				ID string `json:"id"`
			} `json:"attachments"`
		} `json:"value"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil
	}

	out := make([]teamsMessage, 0, len(payload.Value))
	for _, item := range payload.Value {
		text := item.Body.Content
		if strings.EqualFold(item.Body.ContentType, "html") {
			text = stripHTML(text)
		}
		if strings.TrimSpace(text) == "" && len(item.Attachments) == 0 {
			continue
		}
		out = append(out, teamsMessage{
			ID:         item.ID,
			Chat:       chat,
			Sender:     item.From.User.ID,
			SenderName: item.From.User.DisplayName,
			Text:       text,
			CreatedAt:  item.CreatedDateTime,
			HasFiles:   len(item.Attachments) > 0,
		})
	}
	return out
}

// stripHTML turns Teams' HTML message bodies into readable plain text.
func stripHTML(input string) string {
	var out strings.Builder
	depth := 0
	for _, r := range input {
		switch r {
		case '<':
			depth++
		case '>':
			if depth > 0 {
				depth--
				out.WriteRune(' ')
			}
		default:
			if depth == 0 {
				out.WriteRune(r)
			}
		}
	}
	text := out.String()
	for _, pair := range [][2]string{
		{"&nbsp;", " "}, {"&amp;", "&"}, {"&lt;", "<"},
		{"&gt;", ">"}, {"&quot;", "\""}, {"&#39;", "'"},
	} {
		text = strings.ReplaceAll(text, pair[0], pair[1])
	}
	return strings.TrimSpace(strings.Join(strings.Fields(text), " "))
}

func (b *bridge) safePath(rel string) (string, error) {
	root, err := filepath.Abs(b.snapshot().DataDir)
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
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "service": "teams-bridge"})
	})

	mux.HandleFunc("/status", b.auth(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), requestExpiry)
		defer cancel()

		live := b.snapshot()
		if !live.configured() {
			writeJSON(w, http.StatusOK, map[string]any{
				"configured": false, "mode": live.authMode(),
				"error":      "not configured - run /teams login to sign in",
			})
			return
		}
		if _, err := b.token(ctx); err != nil {
			writeJSON(w, http.StatusOK, map[string]any{
				"configured": true, "authenticated": false,
				"mode": live.authMode(), "error": err.Error(),
			})
			return
		}
		b.mu.Lock()
		account := b.account
		b.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]any{
			"configured": true, "authenticated": true, "mode": live.authMode(),
			"account": account,
			"tenant":  live.TenantID, "default_chat": live.DefaultChat,
		})
	}))

	// --- device-code sign-in (no client secret, no admin consent) -------
	mux.HandleFunc("/login/start", b.auth(func(w http.ResponseWriter, r *http.Request) {
		// Optional overrides: some tenants block the default public client, so
		// the owner can supply their own app registration's client id.
		var body struct {
			ClientID string `json:"client_id"`
			Tenant   string `json:"tenant"`
		}
		if r.Body != nil {
			_ = json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body)
		}
		if body.ClientID != "" || body.Tenant != "" {
			b.mu.Lock()
			if body.ClientID != "" {
				b.cfg.ClientID = body.ClientID
			}
			if body.Tenant != "" {
				b.cfg.TenantID = body.Tenant
			}
			b.accessToken, b.refreshToken = "", ""
			b.mu.Unlock()
		}

		ctx, cancel := context.WithTimeout(r.Context(), requestExpiry)
		defer cancel()

		login, err := b.startDeviceLogin(ctx)
		if err != nil {
			writeError(w, http.StatusBadGateway, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"user_code":        login.UserCode,
			"verification_url": login.VerifyURL,
			"interval":         login.Interval,
			"expires_in":       int(time.Until(login.ExpiresAt).Seconds()),
			"client_id":        b.snapshot().ClientID,
		})
	}))

	mux.HandleFunc("/login/poll", b.auth(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), requestExpiry)
		defer cancel()

		done, err := b.pollDeviceLogin(ctx)
		if err != nil {
			writeJSON(w, http.StatusOK, map[string]any{
				"done": false, "pending": false, "error": err.Error(),
			})
			return
		}
		if !done {
			writeJSON(w, http.StatusOK, map[string]any{"done": false, "pending": true})
			return
		}
		b.mu.Lock()
		account := b.account
		b.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]any{
			"done": true, "pending": false, "account": account,
		})
	}))

	// Paste a Graph access token directly. Useful when device-code sign-in is
	// blocked by tenant policy: grab a token from Graph Explorer and paste it.
	// Short-lived (about an hour) because a bare access token has no refresh.
	mux.HandleFunc("/login/token", b.auth(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			AccessToken string `json:"access_token"`
		}
		if !decode(w, r, &body) {
			return
		}
		token := strings.TrimSpace(body.AccessToken)
		token = strings.TrimPrefix(token, "Bearer ")
		if len(token) < 40 {
			writeError(w, http.StatusBadRequest, "that does not look like an access token")
			return
		}

		b.mu.Lock()
		b.accessToken = token
		// Assume the Graph default of one hour; a real expiry is not visible
		// without decoding the token, and we re-check on 401 anyway.
		b.expiresAt = time.Now().Add(55 * time.Minute)
		b.refreshToken = ""
		b.mu.Unlock()

		ctx, cancel := context.WithTimeout(r.Context(), requestExpiry)
		defer cancel()

		account, err := b.fetchAccount(ctx)
		if err != nil {
			b.mu.Lock()
			b.accessToken = ""
			b.mu.Unlock()
			writeError(w, http.StatusUnauthorized, "token rejected by Graph: "+err.Error())
			return
		}

		b.mu.Lock()
		b.account = account
		b.mu.Unlock()
		if err := b.saveCreds(); err != nil {
			log.Printf("teams: could not persist token session: %v", err)
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"authenticated": true, "account": account, "expires_in_minutes": 55,
			"note": "access tokens expire in about an hour; re-paste when it does",
		})
	}))

	// Runtime credential configuration (driven from Telegram).
	mux.HandleFunc("/config", b.auth(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			TenantID     string `json:"tenant_id"`
			ClientID     string `json:"client_id"`
			ClientSecret string `json:"client_secret"`
			DefaultChat  string `json:"default_chat"`
		}
		if !decode(w, r, &body) {
			return
		}
		if body.TenantID == "" || body.ClientID == "" || body.ClientSecret == "" {
			writeError(w, http.StatusBadRequest,
				"tenant_id, client_id and client_secret are all required")
			return
		}

		b.mu.Lock()
		b.cfg.TenantID = body.TenantID
		b.cfg.ClientID = body.ClientID
		b.cfg.ClientSecret = body.ClientSecret
		if body.DefaultChat != "" {
			b.cfg.DefaultChat = body.DefaultChat
		}
		b.accessToken = "" // force a fresh token with the new credentials
		b.expiresAt = time.Time{}
		b.mu.Unlock()

		// Verify the credentials actually work before reporting success.
		ctx, cancel := context.WithTimeout(r.Context(), requestExpiry)
		defer cancel()
		if _, err := b.token(ctx); err != nil {
			writeError(w, http.StatusUnauthorized, "credentials rejected: "+err.Error())
			return
		}
		if err := b.saveCreds(); err != nil {
			log.Printf("teams: could not persist credentials: %v", err)
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"configured": true, "authenticated": true, "tenant": body.TenantID,
		})
	}))

	mux.HandleFunc("/disconnect", b.auth(func(w http.ResponseWriter, r *http.Request) {
		b.mu.Lock()
		b.cfg.ClientSecret = ""
		b.accessToken, b.refreshToken, b.account = "", "", ""
		b.expiresAt = time.Time{}
		path := credsPath(b.cfg)
		b.mu.Unlock()

		b.devMu.Lock()
		b.pending = nil
		b.devMu.Unlock()
		os.Remove(path)
		writeJSON(w, http.StatusOK, map[string]any{"disconnected": true})
	}))

	mux.HandleFunc("/send/text", b.auth(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Chat string `json:"chat"`
			Text string `json:"text"`
		}
		if !decode(w, r, &body) {
			return
		}
		if body.Chat == "" {
			body.Chat = b.snapshot().DefaultChat
		}
		if strings.TrimSpace(body.Text) == "" {
			writeError(w, http.StatusBadRequest, "text must not be empty")
			return
		}
		path, err := messagePath(body.Chat)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), requestExpiry)
		defer cancel()

		payload := map[string]any{
			"body": map[string]any{"contentType": "text", "content": body.Text},
		}
		data, status, err := b.graph(ctx, http.MethodPost, path, payload)
		if err != nil {
			writeError(w, mapStatus(status), err.Error())
			return
		}
		var created struct {
			ID string `json:"id"`
		}
		json.Unmarshal(data, &created)
		writeJSON(w, http.StatusOK, map[string]any{
			"sent": true, "message_id": created.ID, "chat": body.Chat,
		})
	}))

	mux.HandleFunc("/messages", b.auth(func(w http.ResponseWriter, r *http.Request) {
		chat := r.URL.Query().Get("chat")
		if chat == "" {
			chat = b.snapshot().DefaultChat
		}
		path, err := messagePath(chat)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		limit := 20
		if raw := r.URL.Query().Get("limit"); raw != "" {
			if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 && parsed <= 50 {
				limit = parsed
			}
		}

		ctx, cancel := context.WithTimeout(r.Context(), requestExpiry)
		defer cancel()

		data, status, err := b.graph(ctx, http.MethodGet,
			fmt.Sprintf("%s?$top=%d", path, limit), nil)
		if err != nil {
			writeError(w, mapStatus(status), err.Error())
			return
		}
		messages := parseMessages(data, chat)
		writeJSON(w, http.StatusOK, map[string]any{
			"count": len(messages), "chat": chat, "messages": messages,
		})
	}))

	mux.HandleFunc("/chats", b.auth(func(w http.ResponseWriter, r *http.Request) {
		user := r.URL.Query().Get("user")
		mode := b.snapshot().authMode()
		if user == "" && mode == "app" {
			writeError(w, http.StatusBadRequest,
				"a 'user' query parameter (user id or UPN) is required in app-only mode")
			return
		}
		limit := 20
		if raw := r.URL.Query().Get("limit"); raw != "" {
			if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 && parsed <= 50 {
				limit = parsed
			}
		}

		// Delegated sign-in can just read the signed-in user's own chats.
		path := fmt.Sprintf("/me/chats?$top=%d", limit)
		if user != "" {
			path = fmt.Sprintf("/users/%s/chats?$top=%d", url.PathEscape(user), limit)
		}

		ctx, cancel := context.WithTimeout(r.Context(), requestExpiry)
		defer cancel()

		data, status, err := b.graph(ctx, http.MethodGet, path, nil)
		if err != nil {
			writeError(w, mapStatus(status), err.Error())
			return
		}
		var payload struct {
			Value []struct {
				ID       string `json:"id"`
				Topic    string `json:"topic"`
				ChatType string `json:"chatType"`
			} `json:"value"`
		}
		json.Unmarshal(data, &payload)
		writeJSON(w, http.StatusOK, map[string]any{
			"count": len(payload.Value), "chats": payload.Value,
		})
	}))

	mux.HandleFunc("/download", b.auth(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			URL  string `json:"url"`
			Path string `json:"path"`
		}
		if !decode(w, r, &body) {
			return
		}
		if body.URL == "" || body.Path == "" {
			writeError(w, http.StatusBadRequest, "both 'url' and 'path' are required")
			return
		}
		full, err := b.safePath(body.Path)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
		defer cancel()

		data, status, err := b.graph(ctx, http.MethodGet, body.URL, nil)
		if err != nil {
			writeError(w, mapStatus(status), err.Error())
			return
		}
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if err := os.WriteFile(full, data, 0o644); err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"downloaded": true, "path": body.Path, "size_bytes": len(data),
		})
	}))

	return mux
}

func mapStatus(graphStatus int) int {
	switch {
	case graphStatus == http.StatusUnauthorized || graphStatus == http.StatusForbidden:
		return http.StatusUnauthorized
	case graphStatus == http.StatusTooManyRequests:
		return http.StatusTooManyRequests
	case graphStatus >= 500:
		return http.StatusBadGateway
	case graphStatus >= 400:
		return http.StatusBadRequest
	default:
		return http.StatusInternalServerError
	}
}

func (b *bridge) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		token := b.snapshot().Token
		if token == "" {
			writeError(w, http.StatusServiceUnavailable, "BRIDGE_TOKEN is not configured")
			return
		}
		if r.Header.Get("X-Bridge-Token") != token {
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
	log.Printf("teams: auth mode = %s", cfg.authMode())
	if cfg.authMode() == "delegated" {
		log.Println("teams: sign in from Telegram with /teams login (device code)")
	}

	b := newBridge(cfg)
	b.loadCreds()

	server := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           b.routes(),
		ReadHeaderTimeout: 15 * time.Second,
	}
	log.Printf("teams bridge listening on :%s", cfg.Port)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("http server: %v", err)
	}
}
