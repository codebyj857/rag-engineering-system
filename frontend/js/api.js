/* =========================================================
   RAG ENGINEERING SYSTEM
   API Client
========================================================= */

const API = (() => {
    "use strict";

    const BASE_URL = window.RAG_API_URL || "http://127.0.0.1:8000";


    /* ---------------------------------------------------------
       Generic Request
    --------------------------------------------------------- */

    async function request(
        endpoint,
        options = {}
    ) {
        const response = await fetch(
            `${BASE_URL}${endpoint}`,
            {
                ...options,
                headers: {
                    "Content-Type": "application/json",
                    ...(options.headers || {})
                }
            }
        );

        let data = null;

        try {
            data = await response.json();
        } catch {
            data = null;
        }

        if (!response.ok) {
            const message =
                data?.detail ||
                data?.message ||
                `API request failed: ${response.status}`;

            throw new Error(message);
        }

        return data;
    }


    /* ---------------------------------------------------------
       Health
    --------------------------------------------------------- */

    async function health() {
        return request("/health", {
            method: "GET"
        });
    }


    /* ---------------------------------------------------------
       Ingest Document
    --------------------------------------------------------- */

    async function ingest(file) {
        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );

        const response =
            await fetch(
                `${BASE_URL}/ingest`,
                {
                    method: "POST",
                    body: formData
                }
            );

        let data = null;

        try {
            data = await response.json();
        } catch {
            data = null;
        }

        if (!response.ok) {
            throw new Error(
                data?.detail ||
                "Document ingestion failed."
            );
        }

        return data;
    }


    /* ---------------------------------------------------------
       Chat
    --------------------------------------------------------- */

    async function chat({
        message,
        sessionId,
        strategy = "hybrid",
        topK = 5,
        verify = true
    }) {
        return request("/chat", {
            method: "POST",

            body: JSON.stringify({
                message,
                session_id: sessionId,
                strategy,
                top_k: topK,
                verify
            })
        });
    }


    /* ---------------------------------------------------------
       Retrieval
    --------------------------------------------------------- */

    async function retrieve({
        question,
        strategy = "hybrid",
        topK = 5
    }) {
        return request("/retrieve", {
            method: "POST",

            body: JSON.stringify({
                question,
                strategy,
                top_k: topK
            })
        });
    }


    /* ---------------------------------------------------------
       New Session
    --------------------------------------------------------- */

    async function createSession() {
        return request("/sessions/new", {
            method: "POST"
        });
    }


    /* ---------------------------------------------------------
       Get Session
    --------------------------------------------------------- */

    async function getSession(sessionId) {
        return request(
            `/sessions/${encodeURIComponent(sessionId)}`,
            {
                method: "GET"
            }
        );
    }


    /* ---------------------------------------------------------
       Reset Session
    --------------------------------------------------------- */

    async function resetSession(sessionId) {
        return request("/sessions/reset", {
            method: "POST",

            body: JSON.stringify({
                session_id: sessionId
            })
        });
    }


    /* ---------------------------------------------------------
       Evaluation
    --------------------------------------------------------- */

    async function evaluate({
        strategy = "hybrid",
        topK = 5,
        fullRag = true
    }) {
        return request("/evaluate", {
            method: "POST",

            body: JSON.stringify({
                strategy,
                top_k: topK,
                full_rag: fullRag
            })
        });
    }


    /* ---------------------------------------------------------
       Public API
    --------------------------------------------------------- */

    return {
        BASE_URL,

        health,
        ingest,

        chat,
        retrieve,

        createSession,
        getSession,
        resetSession,

        evaluate
    };
})();