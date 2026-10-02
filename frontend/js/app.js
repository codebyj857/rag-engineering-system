/* =========================================================
   RAG ENGINEERING SYSTEM
   Application Controller
========================================================= */

(() => {
    "use strict";

    /* ---------------------------------------------------------
       Application State
    --------------------------------------------------------- */

    const state = {
        sessionId: null,
        strategy: "reranked",
        topK: 5,
        verify: true,
        sending: false,
        backendAvailable: false
    };

    /* ---------------------------------------------------------
       DOM References
    --------------------------------------------------------- */

    const {
        chatInput,
        sendBtn,
        newChatBtn,
        fileInput,
        uploadZone,
        retrievalStrategy,
        mobileMenuBtn,
        sidebarOverlay,
        closeInspectionBtn
    } = UI.elements;

    /* ---------------------------------------------------------
       Initialization
    --------------------------------------------------------- */

    async function init() {
        bindEvents();

        UI.setStrategy(state.strategy);

        UI.setConnectionStatus(
            "checking",
            "Checking backend..."
        );

        await checkBackend();

        if (!state.sessionId) {
            if (state.backendAvailable) {
                try {
                    const response =
                        await API.createSession();

                    state.sessionId =
                        response.session_id ||
                        response.id;

                    if (!state.sessionId) {
                        createLocalSession();
                    }
                } catch (error) {
                    createLocalSession();
                }
            } else {
                createLocalSession();
            }
        }

        UI.setSessionId(state.sessionId);
        UI.clearInspection();
    }

    /* ---------------------------------------------------------
       Event Binding
    --------------------------------------------------------- */

    function bindEvents() {
        /* Chat */

        sendBtn?.addEventListener(
            "click",
            handleSend
        );

        chatInput?.addEventListener(
            "keydown",
            handleInputKeydown
        );

        chatInput?.addEventListener(
            "input",
            () => {
                UI.autoResizeTextarea();

                UI.setSendEnabled(
                    UI.getInputValue().length > 0 &&
                    !state.sending
                );
            }
        );

        /* New conversation */

        newChatBtn?.addEventListener(
            "click",
            handleNewConversation
        );

        /* Retrieval strategy */

        retrievalStrategy?.addEventListener(
            "change",
            handleStrategyChange
        );

        /* File upload */

        fileInput?.addEventListener(
            "change",
            handleFileSelection
        );

        uploadZone?.addEventListener(
            "click",
            () => {
                fileInput?.click();
            }
        );

        uploadZone?.addEventListener(
            "dragover",
            event => {
                event.preventDefault();

                uploadZone.classList.add(
                    "dragging"
                );
            }
        );

        uploadZone?.addEventListener(
            "dragleave",
            () => {
                uploadZone.classList.remove(
                    "dragging"
                );
            }
        );

        uploadZone?.addEventListener(
            "drop",
            handleFileDrop
        );

        /* Mobile sidebar */

        mobileMenuBtn?.addEventListener(
            "click",
            UI.openSidebar
        );

        sidebarOverlay?.addEventListener(
            "click",
            UI.closeSidebar
        );

        /* Inspection panel */

        closeInspectionBtn?.addEventListener(
            "click",
            UI.closeInspection
        );

        /* Capability cards */

        document.addEventListener(
            "click",
            handleCapabilityCard
        );
    }

    /* ---------------------------------------------------------
       Backend Health Check
    --------------------------------------------------------- */

    async function checkBackend() {
        try {
            const response =
                await API.health();

            state.backendAvailable = true;

            UI.setConnectionStatus(
                "connected",
                response?.status === "ok"
                    ? "Connected"
                    : "Backend online"
            );
        } catch (error) {
            state.backendAvailable = false;

            UI.setConnectionStatus(
                "error",
                "Backend offline"
            );
        }
    }

    /* ---------------------------------------------------------
       Local Session Fallback
    --------------------------------------------------------- */

    function createLocalSession() {
        state.sessionId =
            `local-${crypto.randomUUID()}`;

        UI.setSessionId(
            state.sessionId
        );
    }

    /* ---------------------------------------------------------
       New Conversation
    --------------------------------------------------------- */

    async function handleNewConversation() {
        if (state.sending) {
            return;
        }

        UI.clearMessages();
        UI.clearInput();

        state.sending = false;

        UI.setSendEnabled(false);

        if (state.backendAvailable) {
            try {
                const response =
                    await API.createSession();

                state.sessionId =
                    response.session_id ||
                    response.id;

                if (!state.sessionId) {
                    createLocalSession();
                }
            } catch (error) {
                createLocalSession();
            }
        } else {
            createLocalSession();
        }

        UI.setSessionId(
            state.sessionId
        );

        UI.clearInspection();

        UI.toast(
            "New conversation started.",
            "success"
        );
    }

    /* ---------------------------------------------------------
       Strategy Change
    --------------------------------------------------------- */

    function handleStrategyChange(event) {
        state.strategy =
            event.target.value;

        UI.setStrategy(
            state.strategy
        );

        UI.toast(
            `${getStrategyName(state.strategy)} selected.`,
            "default"
        );
    }

    function getStrategyName(strategy) {
        const names = {
            naive: "Naive retrieval",
            hybrid: "Hybrid retrieval",
            reranked: "Reranked retrieval"
        };

        return (
            names[strategy] ||
            "Reranked retrieval"
        );
    }

    /* ---------------------------------------------------------
       Keyboard Handling
    --------------------------------------------------------- */

    function handleInputKeydown(event) {
        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();

            handleSend();

            return;
        }

        if (
            event.key === "Enter" &&
            event.shiftKey
        ) {
            return;
        }
    }

    /* ---------------------------------------------------------
       Send Message
    --------------------------------------------------------- */

    async function handleSend() {
        if (state.sending) {
            return;
        }

        const message =
            UI.getInputValue();

        if (!message) {
            return;
        }

        state.sending = true;

        UI.setSendEnabled(false);

        UI.addUserMessage(message);

        UI.clearInput();

        UI.showTypingIndicator();

        /* -----------------------------------------------------
           Backend available
        ----------------------------------------------------- */

        if (state.backendAvailable) {
            try {
                const response =
                    await API.chat({
                        message,
                        sessionId:
                            state.sessionId,
                        strategy:
                            state.strategy,
                        topK:
                            state.topK,
                        verify:
                            state.verify
                    });

                UI.removeTypingIndicator();

                renderChatResponse(
                    response
                );

                state.sending = false;

                UI.setSendEnabled(true);

                return;
            } catch (error) {
                UI.removeTypingIndicator();

                UI.addAssistantMessage(
                    `I couldn't complete that request. ${error.message}`,
                    {
                        label:
                            "RAG Assistant · Error"
                    }
                );

                UI.toast(
                    "The backend returned an error.",
                    "error"
                );

                state.sending = false;

                UI.setSendEnabled(true);

                return;
            }
        }

        /* -----------------------------------------------------
           Frontend-only mode
        ----------------------------------------------------- */

        window.setTimeout(
            () => {
                UI.removeTypingIndicator();

                UI.addAssistantMessage(
                    "The frontend is ready, but the RAG backend is not connected yet. Once we build the FastAPI backend, this conversation will use your real retrieval, generation, memory, and verification pipeline.",
                    {
                        label:
                            "RAG Assistant · Frontend Mode"
                    }
                );

                state.sending = false;

                UI.setSendEnabled(true);
            },
            700
        );
    }

    /* ---------------------------------------------------------
       Render Backend Chat Response
    --------------------------------------------------------- */

    function renderChatResponse(response) {
        /* -----------------------------------------------------
           Answer
        ----------------------------------------------------- */

        const answer =
            response?.answer ||
            response?.response ||
            response?.message ||
            "No response was returned.";

        UI.addAssistantMessage(
            answer
        );

        /* -----------------------------------------------------
           Retrieval Evidence

           Backend:
             response.evidence[]

           Each evidence item contains:
             chunk_id
             content
             source
             score
        ----------------------------------------------------- */

        const evidence =
            Array.isArray(response?.evidence)
                ? response.evidence
                : [];

        if (evidence.length > 0) {
            UI.renderRetrievalDetails(
                evidence
            );

            UI.setRetrievalStatus(
                `${evidence.length} chunks`
            );
        } else {
            UI.renderRetrievalDetails([]);

            UI.setRetrievalStatus(
                "No evidence"
            );
        }

        /* -----------------------------------------------------
           Verification

           Backend:
             response.verification.score

           Confidence:
             response.confidence

           Current API returns confidence as a NUMBER:
             "confidence": 0.827
        ----------------------------------------------------- */

        const verification =
            response?.verification || null;

        const groundedness =
            typeof verification?.score === "number"
                ? verification.score
                : null;

        const confidenceScore =
            typeof response?.confidence === "number"
                ? response.confidence
                : null;

        const verificationStatus =
            verification
                ? "Completed"
                : "—";

        UI.updateVerification({
            groundedness,
            confidence:
                confidenceScore,
            status:
                verificationStatus
        });

        /* -----------------------------------------------------
           Open inspection panel
        ----------------------------------------------------- */

        UI.openInspection();
    }

    /* ---------------------------------------------------------
       File Selection
    --------------------------------------------------------- */

    function handleFileSelection(event) {
        const files =
            [...event.target.files];

        processFiles(files);

        event.target.value = "";
    }

    /* ---------------------------------------------------------
       File Drop
    --------------------------------------------------------- */

    function handleFileDrop(event) {
        event.preventDefault();

        uploadZone?.classList.remove(
            "dragging"
        );

        const files =
            [...event.dataTransfer.files];

        processFiles(files);
    }

    /* ---------------------------------------------------------
       Process Files
    --------------------------------------------------------- */

    async function processFiles(files) {
        if (!files.length) {
            return;
        }

        const allowedExtensions = [
            "pdf",
            "txt",
            "docx",
            "csv",
            "json"
        ];

        for (const file of files) {
            const extension =
                file.name
                    .split(".")
                    .pop()
                    .toLowerCase();

            if (
                !allowedExtensions.includes(
                    extension
                )
            ) {
                UI.toast(
                    `${file.name} is not a supported file type.`,
                    "error"
                );

                continue;
            }

            /* -------------------------------------------------
               Backend available
            ------------------------------------------------- */

            if (state.backendAvailable) {
                let documentItem = null;

                try {
                    documentItem =
                        UI.addDocument(
                            file,
                            "Indexing..."
                        );

                    await API.ingest(
                        file
                    );

                    const statusElement =
                        documentItem?.querySelector(
                            ".document-status"
                        );

                    if (statusElement) {
                        statusElement.textContent =
                            "Indexed";
                    }

                    UI.toast(
                        `${file.name} indexed successfully.`,
                        "success"
                    );
                } catch (error) {
                    const statusElement =
                        documentItem?.querySelector(
                            ".document-status"
                        );

                    if (statusElement) {
                        statusElement.textContent =
                            "Failed";
                    }

                    UI.toast(
                        `Failed to index ${file.name}: ${error.message}`,
                        "error"
                    );
                }

                continue;
            }

            /* -------------------------------------------------
               Frontend-only mode
            ------------------------------------------------- */

            UI.addDocument(
                file,
                "Local"
            );

            UI.toast(
                `${file.name} added to the interface.`,
                "default"
            );
        }
    }

    /* ---------------------------------------------------------
       Capability Cards
    --------------------------------------------------------- */

    function handleCapabilityCard(event) {
        const card =
            event.target.closest(
                "[data-prompt]"
            );

        if (!card) {
            return;
        }

        const prompt =
            card.dataset.prompt;

        if (!prompt || !chatInput) {
            return;
        }

        chatInput.value =
            prompt;

        UI.autoResizeTextarea();

        UI.setSendEnabled(true);

        chatInput.focus();
    }

    /* ---------------------------------------------------------
       Start Application
    --------------------------------------------------------- */

    init();

})();