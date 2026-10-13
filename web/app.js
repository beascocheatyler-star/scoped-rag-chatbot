/* Scoped Document Chatbot — frontend. Vanilla JS, no build step. */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const el = {
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    uploadStatus: $("upload-status"),
    docList: $("doc-list"),
    corpusSummary: $("corpus-summary"),
    clearDocs: $("clear-docs"),
    presetSelect: $("preset-select"),
    personaName: $("persona-name"),
    personaTopic: $("persona-topic"),
    personaStyle: $("persona-style"),
    savePersona: $("save-persona"),
    personaStatus: $("persona-status"),
    llmBadge: $("llm-badge"),
    chatTitle: $("chat-title"),
    chatSubtitle: $("chat-subtitle"),
    messages: $("messages"),
    suggestions: $("suggestions"),
    chatForm: $("chat-form"),
    chatInput: $("chat-input"),
    sendBtn: $("send-btn"),
    resetChat: $("reset-chat"),
    sidebar: $("sidebar"),
    sidebarToggle: $("sidebar-toggle"),
    scrollTop: $("scroll-top"),
    scrollBottom: $("scroll-bottom"),
  };

  /** Conversation history sent to the API (prose only, no context blocks). */
  let history = [];
  let greeting = "";
  let busy = false;

  // ------------------------------------------------------------- scrolling

  /** px from the bottom within which we consider the user "pinned" to the end. */
  const STICK_THRESHOLD = 120;

  /**
   * True when the transcript is scrolled at (or very near) the bottom.
   * Drives whether new messages are allowed to pull the view down: if the
   * user has deliberately scrolled up to re-read something, we leave them
   * where they are instead of yanking them back.
   */
  function isPinnedToBottom() {
    const m = el.messages;
    return m.scrollHeight - m.scrollTop - m.clientHeight <= STICK_THRESHOLD;
  }

  /**
   * Scroll the transcript to the bottom.
   * `force` bypasses the pinned check (used for user-initiated actions such as
   * sending a message or pressing the jump button); otherwise this is a no-op
   * while the user is reading further up.
   */
  function scrollToBottom(force = false, smooth = false) {
    if (!force && !isPinnedToBottom()) return;
    const m = el.messages;
    // `scroll-behavior: smooth` is applied per-call rather than globally so
    // that programmatic appends land instantly and never animate against an
    // in-progress user scroll.
    m.scrollTo({ top: m.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }

  function scrollToTop() {
    el.messages.scrollTo({ top: 0, behavior: "smooth" });
  }

  /**
   * Show each jump button only when it would actually do something, so they
   * stay out of the way in a short conversation.
   */
  function updateScrollButtons() {
    const m = el.messages;
    const scrollable = m.scrollHeight - m.clientHeight > STICK_THRESHOLD;
    el.scrollTop.hidden = !scrollable || m.scrollTop <= STICK_THRESHOLD;
    el.scrollBottom.hidden = !scrollable || isPinnedToBottom();
  }

  function bindScrollControls() {
    // rAF-throttled: scroll events fire far faster than we need to re-evaluate.
    let queued = false;
    el.messages.addEventListener("scroll", () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        updateScrollButtons();
      });
    }, { passive: true });

    el.scrollTop.addEventListener("click", scrollToTop);
    el.scrollBottom.addEventListener("click", () => scrollToBottom(true, true));

    // Keep the buttons honest when the viewport (and so clientHeight) changes.
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(updateScrollButtons).observe(el.messages);
    } else {
      window.addEventListener("resize", updateScrollButtons);
    }

    // Home/End jump the transcript when it has focus, matching normal
    // scroll-container behaviour that the custom layout would otherwise eat.
    el.messages.addEventListener("keydown", (e) => {
      if (e.key === "Home") { e.preventDefault(); scrollToTop(); }
      else if (e.key === "End") { e.preventDefault(); scrollToBottom(true, true); }
    });
  }

  // --------------------------------------------------------------- sidebar

  function setSidebar(open) {
    document.body.classList.toggle("sidebar-collapsed", !open);
    el.sidebarToggle.setAttribute("aria-expanded", String(open));
    try { localStorage.setItem("sidebarOpen", open ? "1" : "0"); } catch (_) {}
  }

  function bindSidebar() {
    let open = true;
    try { open = localStorage.getItem("sidebarOpen") !== "0"; } catch (_) {}
    // On narrow screens the sidebar stacks above the chat and eats the
    // viewport, so start it collapsed there regardless of the stored value.
    if (window.matchMedia("(max-width: 900px)").matches) open = false;
    setSidebar(open);

    el.sidebarToggle.addEventListener("click", () => {
      const nowOpen = el.sidebarToggle.getAttribute("aria-expanded") !== "true";
      setSidebar(nowOpen);
      if (nowOpen) el.sidebar.scrollTop = 0; // always reveal persona controls first
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" &&
          document.body.classList.contains("sidebar-collapsed") === false &&
          window.matchMedia("(max-width: 900px)").matches) {
        setSidebar(false);
      }
    });
  }

  // ---------------------------------------------------------------- utils

  /** Escape before any markup is generated. Document text is untrusted. */
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Sentinel marking where a fenced code block was lifted out of the text.
  //
  // It must be something `escapeHtml` can never emit, so a document cannot
  // forge one. U+0000 qualifies: escapeHtml only ever produces `&...;`
  // entities and the original characters. A readable marker like "FENCE0"
  // would be forgeable — a document containing that literal text would be
  // substituted away and render as "undefined".
  const FENCE = "\u0000";
  const FENCE_LINE_RE = /^\u0000\d+\u0000$/;
  const FENCE_RE = /\u0000(\d+)\u0000/g;

  /**
   * Minimal markdown renderer, operating strictly on already-escaped text.
   * Covers what the model actually emits: headings, lists, tables, code,
   * bold/italic, links, blockquotes.
   */
  function renderMarkdown(raw) {
    // Strip any NUL already present so it cannot collide with the sentinel.
    const source = escapeHtml(String(raw).replace(/\u0000/g, ""));
    const fences = [];

    // Pull fenced code out first so its contents escape further processing.
    let text = source.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const cls = lang ? ` class="language-${lang}"` : "";
      fences.push(`<pre><code${cls}>${code.replace(/\n$/, "")}</code></pre>`);
      return `${FENCE}${fences.length - 1}${FENCE}`;
    });

    const lines = text.split("\n");
    const out = [];
    let list = null;      // "ul" | "ol" | null
    let inTable = false;
    let paragraph = [];

    const flushParagraph = () => {
      if (paragraph.length) {
        out.push(`<p>${paragraph.join(" ")}</p>`);
        paragraph = [];
      }
    };
    const closeList = () => {
      if (list) { out.push(`</${list}>`); list = null; }
    };
    const closeTable = () => {
      if (inTable) { out.push("</tbody></table>"); inTable = false; }
    };
    const closeAll = () => { flushParagraph(); closeList(); closeTable(); };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      if (!trimmed) { closeAll(); continue; }

      if (FENCE_LINE_RE.test(trimmed)) {
        closeAll();
        out.push(trimmed);
        continue;
      }

      // Table: a header row followed by a separator row.
      if (
        !inTable &&
        /^\|.*\|$/.test(trimmed) &&
        i + 1 < lines.length &&
        /^\|[\s:|-]+\|$/.test(lines[i + 1].trim())
      ) {
        closeAll();
        const headers = trimmed.slice(1, -1).split("|").map((c) => c.trim());
        out.push(
          "<table><thead><tr>" +
            headers.map((h) => `<th>${inline(h)}</th>`).join("") +
            "</tr></thead><tbody>"
        );
        inTable = true;
        i++; // consume the separator row
        continue;
      }
      if (inTable) {
        if (/^\|.*\|$/.test(trimmed)) {
          const cells = trimmed.slice(1, -1).split("|").map((c) => c.trim());
          out.push("<tr>" + cells.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>");
          continue;
        }
        closeTable();
      }

      let m;
      if ((m = trimmed.match(/^(#{1,6})\s+(.*)$/))) {
        closeAll();
        const level = Math.min(m[1].length + 2, 6); // demote: h1 -> h3
        out.push(`<h${level}>${inline(m[2])}</h${level}>`);
        continue;
      }

      if ((m = trimmed.match(/^[-*+]\s+(.*)$/))) {
        flushParagraph();
        closeTable();
        if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
        out.push(`<li>${inline(m[1])}</li>`);
        continue;
      }

      if ((m = trimmed.match(/^\d+[.)]\s+(.*)$/))) {
        flushParagraph();
        closeTable();
        if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
        out.push(`<li>${inline(m[1])}</li>`);
        continue;
      }

      if ((m = trimmed.match(/^&gt;\s?(.*)$/))) {
        closeAll();
        out.push(`<blockquote>${inline(m[1])}</blockquote>`);
        continue;
      }

      if (/^(---|\*\*\*|___)$/.test(trimmed)) {
        closeAll();
        out.push("<hr />");
        continue;
      }

      closeList();
      paragraph.push(inline(trimmed));
    }
    closeAll();

    let html = out.join("");
    html = html.replace(FENCE_RE, (_, n) => fences[Number(n)]);
    return html;
  }

  /** Inline span-level markdown. Input is already HTML-escaped. */
  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|\W)\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/(^|\W)_([^_\n]+)_/g, "$1<em>$2</em>")
      .replace(
        /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      );
  }

  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    let payload = null;
    try { payload = await res.json(); } catch { /* empty body */ }
    if (!res.ok) {
      throw new Error((payload && payload.detail) || `Request failed (${res.status})`);
    }
    return payload;
  }

  // ------------------------------------------------------------- messages

  function clearEmptyState() {
    const existing = el.messages.querySelector(".empty-state");
    if (existing) existing.remove();
  }

  function showEmptyState() {
    el.messages.innerHTML = `
      <div class="empty-state">
        <h3>Ready when you are</h3>
        <p>${escapeHtml(greeting || "Upload documents, then ask a question.")}</p>
      </div>`;
  }

  function addMessage(role, text, meta = {}) {
    clearEmptyState();

    const wrap = document.createElement("div");
    wrap.className = `message ${role}`;
    if (meta.variant) wrap.classList.add(meta.variant);

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "You" : "AI";
    avatar.setAttribute("aria-hidden", "true");

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (role === "assistant" && meta.scope) {
      const tag = document.createElement("span");
      const inScope = meta.scope === "in";
      tag.className = `scope-tag ${meta.scope}`;
      tag.textContent = inScope ? "Answered from documents" : "Outside scope";
      bubble.appendChild(tag);
    }

    const body = document.createElement("div");
    body.innerHTML = role === "assistant" ? renderMarkdown(text) : `<p>${escapeHtml(text)}</p>`;
    bubble.appendChild(body);

    if (meta.citations && meta.citations.length) {
      bubble.appendChild(buildCitations(meta.citations));
    }

    wrap.append(avatar, bubble);
    el.messages.appendChild(wrap);
    // Only follow the new message if the user was already at the bottom.
    scrollToBottom(role === "user");
    updateScrollButtons();
    return wrap;
  }

  function buildCitations(citations) {
    const details = document.createElement("details");
    details.className = "citations";

    const summary = document.createElement("summary");
    summary.textContent = `${citations.length} source passage${citations.length === 1 ? "" : "s"}`;
    details.appendChild(summary);

    citations.forEach((c) => {
      const item = document.createElement("div");
      item.className = "citation";

      const head = document.createElement("div");
      head.className = "citation-head";

      const name = document.createElement("span");
      name.textContent = `${c.doc_name} · passage ${c.position}`;

      const score = document.createElement("span");
      score.className = "citation-score";
      score.textContent = c.score.toFixed(3);

      head.append(name, score);

      const text = document.createElement("div");
      text.className = "citation-text";
      text.textContent = c.excerpt;

      item.append(head, text);
      details.appendChild(item);
    });
    return details;
  }

  function showTyping() {
    clearEmptyState();
    const wrap = document.createElement("div");
    wrap.className = "message assistant";
    wrap.innerHTML =
      '<div class="avatar" aria-hidden="true">AI</div>' +
      '<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
    el.messages.appendChild(wrap);
    scrollToBottom();
    updateScrollButtons();
    return wrap;
  }

  // ---------------------------------------------------------------- chat

  async function sendMessage(text) {
    if (busy) return;
    const question = text.trim();
    if (!question) return;

    busy = true;
    el.sendBtn.disabled = true;
    el.chatInput.value = "";
    autosize();
    el.suggestions.innerHTML = "";

    addMessage("user", question);
    const typing = showTyping();

    try {
      const data = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, history }),
      });

      typing.remove();
      addMessage("assistant", data.answer, {
        scope: data.in_scope ? "in" : "out",
        variant: data.in_scope ? null : "refusal",
        citations: data.in_scope ? data.citations : [],
      });

      history.push({ role: "user", content: question });
      history.push({ role: "assistant", content: data.answer });
      if (history.length > 16) history = history.slice(-16);
    } catch (err) {
      typing.remove();
      addMessage("assistant", err.message, { variant: "error" });
    } finally {
      busy = false;
      el.sendBtn.disabled = false;
      el.chatInput.focus();
    }
  }

  function autosize() {
    el.chatInput.style.height = "auto";
    el.chatInput.style.height = `${Math.min(el.chatInput.scrollHeight, 180)}px`;
  }

  // ----------------------------------------------------------- documents

  function setUploadStatus(message, kind) {
    el.uploadStatus.innerHTML = message
      ? `<div class="msg ${kind}">${escapeHtml(message)}</div>`
      : "";
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;

    setUploadStatus(`Processing ${files.length} file${files.length === 1 ? "" : "s"}…`, "busy");

    const form = new FormData();
    files.forEach((f) => form.append("files", f));

    try {
      const data = await api("/api/documents", { method: "POST", body: form });
      const parts = [];
      if (data.uploaded.length) parts.push(`Added ${data.uploaded.length}`);
      if (data.skipped.length) parts.push(`skipped ${data.skipped.length}`);
      setUploadStatus(
        `${parts.join(", ")} — ${data.total_chunks} passages indexed.`,
        data.skipped.length ? "error" : "ok"
      );
      if (data.skipped.length) {
        data.skipped.forEach((s) => {
          const div = document.createElement("div");
          div.className = "msg error";
          div.textContent = `${s.name}: ${s.reason}`;
          el.uploadStatus.appendChild(div);
        });
      }
      await refreshDocuments();
      await refreshStatus();
    } catch (err) {
      setUploadStatus(err.message, "error");
    }
  }

  async function refreshDocuments() {
    try {
      const docs = await api("/api/documents");
      el.docList.innerHTML = "";

      docs.forEach((doc) => {
        const li = document.createElement("li");
        li.className = "doc-item";

        const name = document.createElement("span");
        name.className = "doc-name";
        name.textContent = doc.name;
        name.title = doc.name;

        const meta = document.createElement("span");
        meta.className = "doc-meta";
        meta.textContent = `${doc.chunk_count}p · ${formatBytes(doc.size_bytes)}`;

        const remove = document.createElement("button");
        remove.className = "doc-remove";
        remove.innerHTML = "&times;";
        remove.title = `Remove ${doc.name}`;
        remove.setAttribute("aria-label", `Remove ${doc.name}`);
        remove.addEventListener("click", async () => {
          await api(`/api/documents/${doc.doc_id}`, { method: "DELETE" });
          await refreshDocuments();
          await refreshStatus();
        });

        li.append(name, meta, remove);
        el.docList.appendChild(li);
      });

      const total = docs.reduce((n, d) => n + d.chunk_count, 0);
      el.corpusSummary.textContent = docs.length
        ? `${docs.length} document${docs.length === 1 ? "" : "s"} · ${total} passages`
        : "No documents yet";
      el.clearDocs.hidden = docs.length === 0;
    } catch (err) {
      el.corpusSummary.textContent = err.message;
    }
  }

  // ------------------------------------------------------------- persona

  function fillPersona(p) {
    el.personaName.value = p.name || "";
    el.personaTopic.value = p.topic || "";
    el.personaStyle.value = p.style || "";
    el.chatTitle.textContent = p.name || "Assistant";
    el.chatSubtitle.textContent = p.topic ? `Scope: ${p.topic}` : "";
    renderSuggestions(p.suggested_questions || []);
  }

  function renderSuggestions(questions) {
    el.suggestions.innerHTML = "";
    if (history.length) return; // only offer starters on a fresh conversation
    questions.slice(0, 4).forEach((q) => {
      const btn = document.createElement("button");
      btn.className = "suggestion";
      btn.type = "button";
      btn.textContent = q;
      btn.addEventListener("click", () => sendMessage(q));
      el.suggestions.appendChild(btn);
    });
  }

  async function refreshStatus() {
    try {
      const data = await api("/api/status");
      greeting = data.greeting;
      fillPersona(data.persona);

      if (data.llm_configured) {
        el.llmBadge.textContent = data.model;
        el.llmBadge.className = "badge ok";
        el.llmBadge.title = `Connected to ${data.model}`;
      } else {
        el.llmBadge.textContent = "no API key";
        el.llmBadge.className = "badge error";
        el.llmBadge.title = "Set ANTHROPIC_API_KEY in your .env file.";
      }

      if (!history.length) showEmptyState();
    } catch (err) {
      el.llmBadge.textContent = "offline";
      el.llmBadge.className = "badge error";
    }
  }

  // -------------------------------------------------------------- events

  el.dropzone.addEventListener("click", () => el.fileInput.click());
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
  });
  el.fileInput.addEventListener("change", (e) => {
    uploadFiles(e.target.files);
    e.target.value = "";
  });

  ["dragenter", "dragover"].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.add("dragging");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove("dragging");
    })
  );
  el.dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

  // Prevent a stray drop elsewhere on the page from navigating away.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  el.clearDocs.addEventListener("click", async () => {
    if (!confirm("Remove all documents from the knowledge base?")) return;
    await api("/api/documents", { method: "DELETE" });
    await refreshDocuments();
    await refreshStatus();
    setUploadStatus("Knowledge base cleared.", "ok");
  });

  el.presetSelect.addEventListener("change", async (e) => {
    const preset = e.target.value;
    if (!preset) return;
    try {
      const p = await api("/api/persona/preset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset }),
      });
      fillPersona(p);
      history = [];
      await refreshStatus();
      el.personaStatus.textContent = "Preset applied.";
      setTimeout(() => (el.personaStatus.textContent = ""), 2500);
    } catch (err) {
      el.personaStatus.textContent = err.message;
    }
  });

  el.savePersona.addEventListener("click", async () => {
    try {
      const p = await api("/api/persona", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: el.personaName.value || "Assistant",
          topic: el.personaTopic.value || "the uploaded documents",
          style: el.personaStyle.value || "Be clear, concise, and professional.",
          greeting: "",
          suggested_questions: [],
        }),
      });
      fillPersona(p);
      el.presetSelect.value = "";
      el.personaStatus.textContent = "Saved.";
      setTimeout(() => (el.personaStatus.textContent = ""), 2500);
    } catch (err) {
      el.personaStatus.textContent = err.message;
    }
  });

  el.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(el.chatInput.value);
  });

  el.chatInput.addEventListener("input", autosize);
  el.chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(el.chatInput.value);
    }
  });

  el.resetChat.addEventListener("click", async () => {
    history = [];
    await refreshStatus();
    el.chatInput.focus();
  });

  // ---------------------------------------------------------------- init

  (async function init() {
    bindScrollControls();
    bindSidebar();
    await refreshStatus();
    await refreshDocuments();
    updateScrollButtons();
    el.chatInput.focus();
  })();
})();
