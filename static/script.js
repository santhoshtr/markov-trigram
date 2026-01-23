document.addEventListener('DOMContentLoaded', () => {
    const promptInput = document.getElementById('prompt');
    const generateBtn = document.getElementById('generateBtn');
    const stopBtn = document.getElementById('stopBtn');
    const output = document.getElementById('output');
    const status = document.getElementById('status');
    const tokenCount = document.getElementById('tokenCount');

    let abortController = null;
    let currentReader = null;
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
        // Signal generation should stop
        isGenerating = false;
        
        // Abort the fetch request at network level
        if (abortController) {
            abortController.abort();
            abortController = null;
        }
        
        // Cancel the reader stream
        if (currentReader) {
            currentReader.cancel();
            currentReader = null;
        }
        
        status.textContent = '⏹️ Generation stopped';
        status.className = 'status';
        generateBtn.style.display = 'inline-flex';
        generateBtn.disabled = false;
        stopBtn.style.display = 'none';
    });

    async function generateText(prompt) {
        return new Promise((resolve, reject) => {
            try {
                // Create a new abort controller for this generation session
                abortController = new AbortController();

                // Fetch the SSE stream with abort signal
                fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ prompt }),
                    signal: abortController.signal,  // Enable cancellation
                })
                    .then((response) => {
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}`);
                        }

                        // Get the response body as a readable stream
                        const reader = response.body.getReader();
                        currentReader = reader;
                        const decoder = new TextDecoder();
                        let buffer = '';

                        function read() {
                            // Check if generation was stopped by user
                            if (!isGenerating) {
                                reader.cancel('User stopped generation');
                                resolve();
                                return;
                            }

                            reader.read().then(({ done, value }) => {
                                if (done) {
                                    status.textContent = '✅ Done!';
                                    status.className = 'status';
                                    currentReader = null;
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
                                                    currentReader = null;
                                                    reader.cancel('Generation complete');
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
                                currentReader = null;
                                if (error.name === 'AbortError') {
                                    // User stopped generation
                                    resolve();
                                } else if (!isGenerating) {
                                    // Generation was stopped
                                    resolve();
                                } else {
                                    // Actual error occurred
                                    reject(error);
                                }
                            });
                        }

                        read();
                    })
                    .catch((error) => {
                        if (error.name === 'AbortError') {
                            // Fetch was aborted
                            resolve();
                        } else {
                            reject(error);
                        }
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
