document.addEventListener("DOMContentLoaded", () => {
	const promptInput = document.getElementById("prompt");
	const generateBtn = document.getElementById("generateBtn");
	const stopBtn = document.getElementById("stopBtn");
	const output = document.getElementById("output");
	const status = document.getElementById("status");
	const tokenCount = document.getElementById("tokenCount");

	generateBtn.addEventListener("click", () => {
		const prompt = promptInput.value.trim();

		if (!prompt) {
			status.textContent = "❌ Please enter a prompt";
			status.className = "status";
			return;
		}

		generateText(prompt);
	});

	// Stop button is hidden - kept for potential future use
	stopBtn.addEventListener("click", () => {
		status.textContent = "⏹ Generation stopped";
		status.className = "status";
	});

	async function typeText(text, element, delay = 20) {
		const words = text.split(" ");
		element.textContent = "";

		for (let i = 0; i < words.length; i++) {
			if (i > 0) element.textContent += " ";
			element.textContent += words[i];
			element.scrollTop = element.scrollHeight;
			await new Promise((resolve) => setTimeout(resolve, delay));
		}
	}

	async function generateText(prompt) {
		// Reset UI
		output.textContent = "";
		output.classList.remove("error");
		tokenCount.textContent = "0 tokens";
		isGenerating = true;

		// Update button states
		generateBtn.disabled = true;
		stopBtn.style.display = "none";
		status.textContent = "⏳ Generating...";
		status.className = "status generating";

		// Show prompt text while waiting for API
		output.textContent = prompt;

		try {
			const response = await fetch("/api/generate", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({ prompt }),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.error || `HTTP ${response.status}`);
			}

			const data = await response.json();

			// Update status for typing phase
			status.textContent = "✨ Generating...";
			status.className = "status generating";

			// Type the text with animation
			await typeText(data.text, output, 20);

			// Mark as complete
			tokenCount.textContent = `${data.tokens} tokens`;
			status.textContent = "✅ Done!";
			status.className = "status";
		} catch (error) {
			output.textContent = `Error: ${error.message}`;
			output.classList.add("error");
			status.textContent = "❌ Generation failed";
			status.className = "status";
		} finally {
			generateBtn.disabled = false;
			isGenerating = false;
		}
	}
});
