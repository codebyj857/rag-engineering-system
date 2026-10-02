/* =========================================================
   RAG ENGINEERING SYSTEM
   UI Utilities & DOM Interactions
========================================================= */

const UI = (() => {
    "use strict";

    /* ---------------------------------------------------------
       DOM Helpers
    --------------------------------------------------------- */

    const $ = (selector, parent = document) =>
        parent.querySelector(selector);

    const $$ = (selector, parent = document) =>
        [...parent.querySelectorAll(selector)];

    /* ---------------------------------------------------------
       Cached DOM Elements
    --------------------------------------------------------- */

    const elements = {
        messages: $("#messages"),
        welcomeState: $("#welcomeState"),
        chatInput: $("#chatInput"),
        sendBtn: $("#sendBtn"),
        newChatBtn: $("#newChatBtn"),
        uploadZone: $("#uploadZone"),
        fileInput: $("#fileInput"),
        documentList: $("#documentList"),
        emptyDocuments: $("#emptyDocuments"),
        documentCount: $("#documentCount"),
        retrievalStrategy: $("#retrievalStrategy"),
        activeStrategyLabel: $("#activeStrategyLabel"),
        sessionIdDisplay: $("#sessionIdDisplay"),
        connectionDot: $("#connectionDot"),
        connectionStatus: $("#connectionStatus"),
        inspectionPanel: $("#inspectionPanel"),
        closeInspectionBtn: $("#closeInspectionBtn"),
        mobileMenuBtn: $("#mobileMenuBtn"),
        sidebar: $("#sidebar"),
        sidebarOverlay: $("#sidebarOverlay"),
        retrievalStatus: $("#retrievalStatus"),
        retrievalDetails: $("#retrievalDetails"),
        verificationStatus: $("#verificationStatus"),
        groundednessScore: $("#groundednessScore"),
        groundednessFill: $("#groundednessFill"),
        confidenceScore: $("#confidenceScore"),
        confidenceFill: $("#confidenceFill"),
        faithfulnessMetric: $("#faithfulnessMetric"),
        correctnessMetric: $("#correctnessMetric"),
        retrievalMetric: $("#retrievalMetric"),
        latencyMetric: $("#latencyMetric"),
        runEvaluationBtn: $("#runEvaluationBtn"),
        toastContainer: $("#toastContainer")
    };

    /* ---------------------------------------------------------
       Strategy Labels
    --------------------------------------------------------- */

    const strategyLabels = {
        naive: "Naive Retrieval",
        hybrid: "Hybrid Retrieval",
        reranked: "Reranked Retrieval"
    };

    /* ---------------------------------------------------------
       Public: Set Active Strategy
    --------------------------------------------------------- */

    function setStrategy(strategy) {
        const label =
            strategyLabels[strategy] ||
            strategyLabels.reranked;

        if (elements.activeStrategyLabel) {
            elements.activeStrategyLabel.textContent =
                label;
        }

        if (
            elements.retrievalStrategy &&
            elements.retrievalStrategy.value !== strategy
        ) {
            elements.retrievalStrategy.value =
                strategy;
        }
    }

    /* ---------------------------------------------------------
       Public: Set Session ID
    --------------------------------------------------------- */

    function setSessionId(sessionId) {
        if (!elements.sessionIdDisplay) {
            return;
        }

        if (!sessionId) {
            elements.sessionIdDisplay.textContent =
                "—";

            return;
        }

        const shortId =
            sessionId.length > 16
                ? `${sessionId.slice(0, 8)}…${sessionId.slice(-5)}`
                : sessionId;

        elements.sessionIdDisplay.textContent =
            shortId;

        elements.sessionIdDisplay.title =
            sessionId;
    }

    /* ---------------------------------------------------------
       Public: Connection Status
    --------------------------------------------------------- */

    function setConnectionStatus(
        status,
        message
    ) {
        if (
            !elements.connectionDot ||
            !elements.connectionStatus
        ) {
            return;
        }

        elements.connectionDot.classList.remove(
            "connected",
            "error"
        );

        if (status === "connected") {
            elements.connectionDot.classList.add(
                "connected"
            );

            elements.connectionStatus.textContent =
                message || "Connected";

            return;
        }

        if (status === "error") {
            elements.connectionDot.classList.add(
                "error"
            );

            elements.connectionStatus.textContent =
                message || "Unavailable";

            return;
        }

        elements.connectionStatus.textContent =
            message || "Checking...";
    }

    /* ---------------------------------------------------------
       Public: Show / Hide Welcome State
    --------------------------------------------------------- */

    function setWelcomeVisible(visible) {
        if (!elements.welcomeState) {
            return;
        }

        elements.welcomeState.style.display =
            visible ? "flex" : "none";
    }

    /* ---------------------------------------------------------
       Public: Add User Message
    --------------------------------------------------------- */

    function addUserMessage(message) {
        removeWelcomeState();

        const wrapper =
            createMessageElement(
                "user",
                message
            );

        elements.messages.appendChild(
            wrapper
        );

        scrollMessagesToBottom();

        return wrapper;
    }

    /* ---------------------------------------------------------
       Public: Add Assistant Message
    --------------------------------------------------------- */

    function addAssistantMessage(
        message,
        options = {}
    ) {
        removeWelcomeState();

        const wrapper =
            createMessageElement(
                "assistant",
                message,
                options
            );

        elements.messages.appendChild(
            wrapper
        );

        scrollMessagesToBottom();

        return wrapper;
    }

    /* ---------------------------------------------------------
       Create Message
    --------------------------------------------------------- */

    function createMessageElement(
        role,
        message,
        options = {}
    ) {
        const article =
            document.createElement(
                "article"
            );

        article.className =
            `message ${role}`;

        const inner =
            document.createElement(
                "div"
            );

        inner.className =
            "message-inner";

        const avatar =
            document.createElement(
                "div"
            );

        avatar.className =
            "message-avatar";

        avatar.textContent =
            role === "user"
                ? "YOU"
                : "R";

        const content =
            document.createElement(
                "div"
            );

        content.className =
            "message-content";

        const meta =
            document.createElement(
                "div"
            );

        meta.className =
            "message-meta";

        meta.textContent =
            role === "user"
                ? "You"
                : options.label ||
                  "RAG Assistant";

        const bubble =
            document.createElement(
                "div"
            );

        bubble.className =
            "message-bubble";

        /*
         * For now messages are treated as plain text.
         * Later the backend response renderer can support
         * structured Markdown safely.
         */

        const paragraph =
            document.createElement(
                "p"
            );

        paragraph.textContent =
            message;

        bubble.appendChild(
            paragraph
        );

        content.appendChild(
            meta
        );

        content.appendChild(
            bubble
        );

        inner.appendChild(
            avatar
        );

        inner.appendChild(
            content
        );

        article.appendChild(
            inner
        );

        return article;
    }

    /* ---------------------------------------------------------
       Public: Typing Indicator
    --------------------------------------------------------- */

    function showTypingIndicator() {
        removeTypingIndicator();

        removeWelcomeState();

        const article =
            document.createElement(
                "article"
            );

        article.className =
            "message assistant";

        article.id =
            "typingIndicator";

        const inner =
            document.createElement(
                "div"
            );

        inner.className =
            "message-inner";

        const avatar =
            document.createElement(
                "div"
            );

        avatar.className =
            "message-avatar";

        avatar.textContent =
            "R";

        const content =
            document.createElement(
                "div"
            );

        content.className =
            "message-content";

        const meta =
            document.createElement(
                "div"
            );

        meta.className =
            "message-meta";

        meta.textContent =
            "RAG Assistant";

        const bubble =
            document.createElement(
                "div"
            );

        bubble.className =
            "message-bubble";

        const indicator =
            document.createElement(
                "div"
            );

        indicator.className =
            "typing-indicator";

        indicator.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;

        bubble.appendChild(
            indicator
        );

        content.appendChild(
            meta
        );

        content.appendChild(
            bubble
        );

        inner.appendChild(
            avatar
        );

        inner.appendChild(
            content
        );

        article.appendChild(
            inner
        );

        elements.messages.appendChild(
            article
        );

        scrollMessagesToBottom();
    }

    /* ---------------------------------------------------------
       Public: Remove Typing Indicator
    --------------------------------------------------------- */

    function removeTypingIndicator() {
        const indicator =
            document.getElementById(
                "typingIndicator"
            );

        if (indicator) {
            indicator.remove();
        }
    }

    /* ---------------------------------------------------------
       Public: Clear Messages
    --------------------------------------------------------- */

    function clearMessages() {
        if (!elements.messages) {
            return;
        }

        elements.messages.innerHTML =
            "";

        if (elements.welcomeState) {
            elements.messages.appendChild(
                elements.welcomeState
            );
        }

        setWelcomeVisible(true);

        clearInspection();
    }

    /* ---------------------------------------------------------
       Remove Welcome
    --------------------------------------------------------- */

    function removeWelcomeState() {
        if (elements.welcomeState) {
            elements.welcomeState.style.display =
                "none";
        }
    }

    /* ---------------------------------------------------------
       Public: Add Document
    --------------------------------------------------------- */

    function addDocument(
        file,
        status = "Indexed"
    ) {
        if (!elements.documentList) {
            return null;
        }

        if (elements.emptyDocuments) {
            elements.emptyDocuments.remove();
        }

        const item =
            document.createElement(
                "div"
            );

        item.className =
            "document-item";

        const icon =
            document.createElement(
                "div"
            );

        icon.className =
            "document-icon";

        icon.textContent =
            getFileIcon(file.name);

        const info =
            document.createElement(
                "div"
            );

        info.className =
            "document-info";

        const name =
            document.createElement(
                "span"
            );

        name.className =
            "document-name";

        name.textContent =
            file.name;

        name.title =
            file.name;

        const meta =
            document.createElement(
                "div"
            );

        meta.className =
            "document-meta";

        const size =
            document.createElement(
                "span"
            );

        size.textContent =
            formatFileSize(
                file.size
            );

        const separator =
            document.createElement(
                "span"
            );

        separator.textContent =
            "·";

        const statusElement =
            document.createElement(
                "span"
            );

        statusElement.className =
            "document-status";

        statusElement.textContent =
            status;

        meta.appendChild(
            size
        );

        meta.appendChild(
            separator
        );

        meta.appendChild(
            statusElement
        );

        info.appendChild(
            name
        );

        info.appendChild(
            meta
        );

        item.appendChild(
            icon
        );

        item.appendChild(
            info
        );

        elements.documentList.appendChild(
            item
        );

        updateDocumentCount();

        return item;
    }

    /* ---------------------------------------------------------
       Public: Set Documents
    --------------------------------------------------------- */

    function setDocuments(documents) {
        if (!elements.documentList) {
            return;
        }

        elements.documentList.innerHTML =
            "";

        if (
            !documents ||
            documents.length === 0
        ) {
            const empty =
                document.createElement(
                    "div"
                );

            empty.className =
                "empty-documents";

            empty.id =
                "emptyDocuments";

            empty.innerHTML = `
                <span class="empty-icon">◫</span>
                <span>No documents indexed</span>
            `;

            elements.documentList.appendChild(
                empty
            );

            updateDocumentCount();

            return;
        }

        documents.forEach(
            documentItem => {
                addDocument(
                    normalizeFileObject(
                        documentItem
                    ),
                    documentItem.status ||
                        "Indexed"
                );
            }
        );
    }

    /* ---------------------------------------------------------
       Document Count
    --------------------------------------------------------- */

    function updateDocumentCount() {
        if (!elements.documentCount) {
            return;
        }

        const count =
            elements.documentList.querySelectorAll(
                ".document-item"
            ).length;

        elements.documentCount.textContent =
            count;
    }

    /* ---------------------------------------------------------
       File Helpers
    --------------------------------------------------------- */

    function normalizeFileObject(file) {
        return {
            name:
                file.name ||
                file.filename ||
                "Document",

            size:
                Number(file.size) || 0
        };
    }

    function getFileIcon(filename) {
        const extension =
            filename
                .split(".")
                .pop()
                .toLowerCase();

        const icons = {
            pdf: "PDF",
            txt: "TXT",
            docx: "DOC",
            csv: "CSV",
            json: "{}"
        };

        return icons[extension] ||
            "FILE";
    }

    function formatFileSize(bytes) {
        if (!bytes || bytes <= 0) {
            return "Unknown size";
        }

        if (bytes < 1024) {
            return `${bytes} B`;
        }

        if (bytes < 1024 * 1024) {
            return `${(bytes / 1024).toFixed(1)} KB`;
        }

        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    /* ---------------------------------------------------------
       Public: Render Retrieval Details
    --------------------------------------------------------- */

    function renderRetrievalDetails(
        results = []
    ) {
        if (!elements.retrievalDetails) {
            return;
        }

        if (!results.length) {
            elements.retrievalDetails.innerHTML = `
                <div class="empty-detail">
                    No retrieval results available.
                </div>
            `;

            return;
        }

        elements.retrievalDetails.innerHTML =
            results.map(
                (result, index) => {
                    const source =
                        escapeHtml(
                            result.source ||
                            result.document ||
                            `Source ${index + 1}`
                        );

                    const score =
                        typeof result.score === "number"
                            ? result.score.toFixed(3)
                            : "—";

                    const content =
                        escapeHtml(
                            result.content ||
                            result.text ||
                            "No preview available."
                        );

                    return `
                        <div class="retrieval-item">
                            <div class="retrieval-item-header">
                                <span
                                    class="retrieval-source"
                                    title="${source}"
                                >
                                    ${source}
                                </span>

                                <span class="retrieval-score">
                                    ${score}
                                </span>
                            </div>

                            <p>
                                ${content}
                            </p>
                        </div>
                    `;
                }
            ).join("");
    }

    /* ---------------------------------------------------------
       Public: Update Retrieval Status
    --------------------------------------------------------- */

    function setRetrievalStatus(
        status = "—"
    ) {
        if (elements.retrievalStatus) {
            elements.retrievalStatus.textContent =
                status;
        }
    }

    /* ---------------------------------------------------------
       Public: Update Verification
    --------------------------------------------------------- */

    function updateVerification({
        groundedness = null,
        confidence = null,
        status = "—"
    } = {}) {
        if (elements.verificationStatus) {
            elements.verificationStatus.textContent =
                status;
        }

        updateScore(
            elements.groundednessScore,
            elements.groundednessFill,
            groundedness
        );

        updateScore(
            elements.confidenceScore,
            elements.confidenceFill,
            confidence
        );
    }

    function updateScore(
        valueElement,
        fillElement,
        value
    ) {
        if (
            !valueElement ||
            !fillElement
        ) {
            return;
        }

        if (
            value === null ||
            value === undefined ||
            Number.isNaN(
                Number(value)
            )
        ) {
            valueElement.textContent =
                "—";

            fillElement.style.width =
                "0%";

            return;
        }

        let numericValue =
            Number(value);

        /*
         * Support both:
         * 0.85
         * 85
         */

        if (numericValue <= 1) {
            numericValue *= 100;
        }

        numericValue =
            Math.max(
                0,
                Math.min(
                    100,
                    numericValue
                )
            );

        valueElement.textContent =
            `${numericValue.toFixed(1)}%`;

        fillElement.style.width =
            `${numericValue}%`;
    }

    /* ---------------------------------------------------------
       Public: Evaluation Metrics
    --------------------------------------------------------- */

    function updateEvaluation(
        metrics = {}
    ) {
        if (elements.faithfulnessMetric) {
            elements.faithfulnessMetric.textContent =
                formatMetric(
                    metrics.faithfulness
                );
        }

        if (elements.correctnessMetric) {
            elements.correctnessMetric.textContent =
                formatMetric(
                    metrics.correctness
                );
        }

        if (elements.retrievalMetric) {
            elements.retrievalMetric.textContent =
                formatMetric(
                    metrics.retrieval
                );
        }

        if (elements.latencyMetric) {
            elements.latencyMetric.textContent =
                formatLatency(
                    metrics.latency
                );
        }
    }

    function formatMetric(value) {
        if (
            value === null ||
            value === undefined ||
            value === ""
        ) {
            return "—";
        }

        const number =
            Number(value);

        if (Number.isNaN(number)) {
            return String(value);
        }

        return number <= 1
            ? number.toFixed(3)
            : number.toFixed(2);
    }

    function formatLatency(value) {
        if (
            value === null ||
            value === undefined
        ) {
            return "—";
        }

        const number =
            Number(value);

        if (Number.isNaN(number)) {
            return String(value);
        }

        return `${number.toFixed(0)} ms`;
    }

    /* ---------------------------------------------------------
       Public: Clear Inspection
    --------------------------------------------------------- */

    function clearInspection() {
        setRetrievalStatus("—");

        if (elements.retrievalDetails) {
            elements.retrievalDetails.innerHTML = `
                <div class="empty-detail">
                    Retrieval details will appear here after
                    your first query.
                </div>
            `;
        }

        updateVerification({
            groundedness: null,
            confidence: null,
            status: "—"
        });

        updateEvaluation({
            faithfulness: null,
            correctness: null,
            retrieval: null,
            latency: null
        });
    }

    /* ---------------------------------------------------------
       Public: Open Inspection Panel
    --------------------------------------------------------- */

    function openInspection() {
        if (elements.inspectionPanel) {
            elements.inspectionPanel.classList.add(
                "open"
            );
        }
    }

    /* ---------------------------------------------------------
       Public: Close Inspection Panel
    --------------------------------------------------------- */

    function closeInspection() {
        if (elements.inspectionPanel) {
            elements.inspectionPanel.classList.remove(
                "open"
            );
        }
    }

    /* ---------------------------------------------------------
       Public: Mobile Sidebar
    --------------------------------------------------------- */

    function openSidebar() {
        elements.sidebar?.classList.add(
            "open"
        );

        elements.sidebarOverlay?.classList.add(
            "active"
        );
    }

    function closeSidebar() {
        elements.sidebar?.classList.remove(
            "open"
        );

        elements.sidebarOverlay?.classList.remove(
            "active"
        );
    }

    /* ---------------------------------------------------------
       Public: Toast
    --------------------------------------------------------- */

    function toast(
        message,
        type = "default",
        duration = 3200
    ) {
        if (!elements.toastContainer) {
            return;
        }

        const item =
            document.createElement(
                "div"
            );

        item.className =
            `toast ${type}`;

        item.textContent =
            message;

        elements.toastContainer.appendChild(
            item
        );

        window.setTimeout(
            () => {
                item.style.opacity =
                    "0";

                item.style.transform =
                    "translateY(5px)";

                window.setTimeout(
                    () => {
                        item.remove();
                    },
                    180
                );
            },
            duration
        );
    }

    /* ---------------------------------------------------------
       Public: Auto Resize Textarea
    --------------------------------------------------------- */

    function autoResizeTextarea() {
        if (!elements.chatInput) {
            return;
        }

        elements.chatInput.style.height =
            "auto";

        elements.chatInput.style.height =
            `${Math.min(
                elements.chatInput.scrollHeight,
                180
            )}px`;
    }

    /* ---------------------------------------------------------
       Public: Get Input Value
    --------------------------------------------------------- */

    function getInputValue() {
        return elements.chatInput
            ? elements.chatInput.value.trim()
            : "";
    }

    /* ---------------------------------------------------------
       Public: Clear Input
    --------------------------------------------------------- */

    function clearInput() {
        if (!elements.chatInput) {
            return;
        }

        elements.chatInput.value =
            "";

        autoResizeTextarea();
    }

    /* ---------------------------------------------------------
       Public: Set Send State
    --------------------------------------------------------- */

    function setSendEnabled(
        enabled
    ) {
        if (!elements.sendBtn) {
            return;
        }

        elements.sendBtn.disabled =
            !enabled;
    }

    /* ---------------------------------------------------------
       Public: Scroll Messages
    --------------------------------------------------------- */

    function scrollMessagesToBottom() {
        if (!elements.messages) {
            return;
        }

        requestAnimationFrame(
            () => {
                elements.messages.scrollTo({
                    top:
                        elements.messages
                            .scrollHeight,
                    behavior:
                        "smooth"
                });
            }
        );
    }

    /* ---------------------------------------------------------
       Escape HTML
    --------------------------------------------------------- */

    function escapeHtml(value) {
        return String(value)
            .replaceAll(
                "&",
                "&amp;"
            )
            .replaceAll(
                "<",
                "&lt;"
            )
            .replaceAll(
                ">",
                "&gt;"
            )
            .replaceAll(
                '"',
                "&quot;"
            )
            .replaceAll(
                "'",
                "&#039;"
            );
    }

    /* ---------------------------------------------------------
       Public API
    --------------------------------------------------------- */

    return {
        elements,
        setStrategy,
        setSessionId,
        setConnectionStatus,
        setWelcomeVisible,
        addUserMessage,
        addAssistantMessage,
        showTypingIndicator,
        removeTypingIndicator,
        clearMessages,
        addDocument,
        setDocuments,
        updateDocumentCount,
        renderRetrievalDetails,
        setRetrievalStatus,
        updateVerification,
        updateEvaluation,
        clearInspection,
        openInspection,
        closeInspection,
        openSidebar,
        closeSidebar,
        toast,
        autoResizeTextarea,
        getInputValue,
        clearInput,
        setSendEnabled,
        scrollMessagesToBottom
    };
})();