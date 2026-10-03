// Polls the ticket status JSON endpoint every 5 seconds while a ticket is
// still "waiting", and reloads the page automatically once its status
// changes (e.g. waiting -> serving, or waiting -> done/cancelled).
function startTicketPolling(statusUrl) {
  const poll = async () => {
    try {
      const res = await fetch(statusUrl);
      if (!res.ok) return;
      const data = await res.json();
      if (data.status && data.status !== "waiting") {
        window.location.reload();
      }
    } catch (err) {
      // Network hiccup - just try again on the next interval.
      console.warn("Status poll failed:", err);
    }
  };
  setInterval(poll, 5000);
}
