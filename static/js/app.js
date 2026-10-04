// Polls the ticket status JSON endpoint every 5 seconds while a ticket is
// still "waiting", and reloads the page automatically once its status
// changes (e.g. waiting -> serving, or waiting -> done/cancelled).
//
// When the page is running inside the Smart Queue Android app, it also asks
// the app to show a real phone notification ("almost your turn" / "your turn").
// In a normal browser nothing extra happens.

const notificationsSent = new Set();

function notifyApp(key, title, message) {
  if (notificationsSent.has(key)) return;   // only once per event
  try {
    if (localStorage.getItem(key)) return;  // also survives page reloads
    localStorage.setItem(key, "1");
  } catch (err) {
    // Storage unavailable - the in-memory Set above still prevents repeats.
  }
  notificationsSent.add(key);
  if (window.AndroidBridge && typeof window.AndroidBridge.notify === "function") {
    try {
      window.AndroidBridge.notify(title, message);
    } catch (err) {
      console.warn("Notification failed:", err);
    }
  }
}

function startTicketPolling(statusUrl) {
  const poll = async () => {
    try {
      const res = await fetch(statusUrl);
      if (!res.ok) return;
      const data = await res.json();

      if (data.status === "serving") {
        notifyApp(statusUrl + ":serving", "It's your turn!", "Please go to your counter now.");
      }

      if (data.status && data.status !== "waiting") {
        window.location.reload();
        return;
      }

      if (data.status === "waiting" && data.position && data.position <= 2) {
        const msg = data.position === 1
          ? "You're next in line!"
          : "You're #" + data.position + " in line. Almost your turn!";
        notifyApp(statusUrl + ":almost", "Almost your turn", msg);
      }
    } catch (err) {
      // Network hiccup - just try again on the next interval.
      console.warn("Status poll failed:", err);
    }
  };
  setInterval(poll, 5000);
}
