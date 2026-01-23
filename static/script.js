document.addEventListener('DOMContentLoaded', () => {
    const promptInput = document.getElementById('prompt');
    const generateBtn = document.getElementById('generateBtn');
    const stopBtn = document.getElementById('stopBtn');
    const output = document.getElementById('output');
    const status = document.getElementById('status');
    const tokenCount = document.getElementById('tokenCount');

    let eventSource = null;
    let isGenerating = false;
    let tokenCounter = 0;

    // Set default placeholder
    promptInput.value = 'Sherlock Holmes was a detective who';

    generateBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();

        if (!prompt) {
            status.textContent = '❌ Please enter a prompt';
            status.style.color = '#ef4444';
            return;
        }

        // Reset UI
        output.textContent = '';
        output.classList.remove('error');
        tokenCounter = 0;
        updateTokenCount();
        isGenerating = true;

        // Update button states
        generateBtn.style.display = 'none';
        generateBtn.disabled = true;
        stopBtn.style.display = 'inline-flex';
        status.textContent = '⏳ Generating...';
        status.className = 'status generating';

        try {
            await generateText(prompt);
        } catch (error) {
            output.textContent = `Error: ${error.message}`;
            output.classList.add('error');
            status.textContent = '❌ Generation failed';
            status.className = 'status';
        } finally {
            isGenerating = false;
            generateBtn.style.display = 'inline-flex';
            generateBtn.disabled = false;
            stopBtn.style.display = 'none';
        }
    });

    stopBtn.addEventListener('click', () => {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        isGenerating = false;
        status.textContent = '⏹️ Generation stopped';
        status.className = 'status';
        generateBtn.style.display = 'inline-flex';
        generateBtn.disabled = false;
        stopBtn.style.display = 'none';
    });

    async function generateText(prompt) {
        return new Promise((resolve, reject) => {
            try {
                // Close any existing connection
                if (eventSource) {
                    eventSource.close();
                }

                // Fetch the SSE stream
                fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ prompt }),
                })
                    .then((response) => {
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}`);
                        }

                        // Get the response body as a readable stream
                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let buffer = '';

                        function read() {
                            reader.read().then(({ done, value }) => {
                                if (done) {
                                    status.textContent = '✅ Done!';
                                    status.className = 'status';
                                    resolve();
                                    return;
                                }

                                // Decode chunk and add to buffer
                                buffer += decoder.decode(value, { stream: true });

                                // Process complete lines
                                const lines = buffer.split('\n');
                                buffer = lines.pop() || ''; // Keep incomplete line in buffer

                                for (const line of lines) {
                                    if (line.startsWith('data: ')) {
                                        const jsonStr = line.slice(6).trim();
                                        if (jsonStr) {
                                            try {
                                                const event = JSON.parse(jsonStr);
                                                if (event.done) {
                                                    status.textContent = '✅ Done!';
                                                    status.className = 'status';
                                                    reader.cancel();
                                                    resolve();
                                                    return;
                                                }
                                                if (event.token) {
                                                    output.textContent = event.token;
                                                    tokenCounter = countTokens(event.token);
                                                    updateTokenCount();
                                                    // Auto-scroll to bottom
                                                    output.scrollTop = output.scrollHeight;
                                                }
                                            } catch (e) {
                                                console.error('Failed to parse event:', jsonStr, e);
                                            }
                                        }
                                    }
                                }

                                read();
                            }).catch((error) => {
                                if (!isGenerating) {
                                    // User cancelled
                                    resolve();
                                } else {
                                    reject(error);
                                }
                            });
                        }

                        read();
                    })
                    .catch((error) => {
                        reject(error);
                    });
            } catch (error) {
                reject(error);
            }
        });
    }

    function countTokens(text) {
        // Simple approximation: count words
        return text.trim().split(/\s+/).filter((w) => w.length > 0).length;
    }

    function updateTokenCount() {
        tokenCount.textContent = `${tokenCounter} token${tokenCounter !== 1 ? 's' : ''}`;
    }
});
