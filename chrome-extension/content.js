// Content script — injected into every page.
// Handles communication between page context and background service worker.
// Currently lightweight — listens for messages from background if needed.

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "GET_IMAGE_URL") {
    // Find the most prominent image on the page (fallback for when srcUrl isn't available)
    const imgs = Array.from(document.querySelectorAll("img"))
      .filter((img) => img.naturalWidth > 100 && img.naturalHeight > 100)
      .sort((a, b) => (b.naturalWidth * b.naturalHeight) - (a.naturalWidth * a.naturalHeight));
    sendResponse({ url: imgs[0]?.src || null });
  }
});
