// Polls the ticket status JSON endpoint every 5 seconds while a ticket is
// still "waiting", and reloads the page automatically once its status
// changes (e.g. waiting -> serving, or waiting -> done/cancelled).
//
// When the page is running inside the Smart Queue Android app, it also asks
// the app to show a real phone notification:
//   - "Your turn is coming up" once the estimated wait is down to the number
//     of minutes the customer chose (e.g. 5 minutes before their turn)
//   - "It's your turn!" when they are called
// In a normal browser nothing extra happens.

const notificationsSent = new Set();

function notifyApp(key, title, message) {
  // Only once per event while this page is open. (Not saved in localStorage:
  // ticket numbers can repeat after a server reset, and a saved "already
  // sent" flag from an old ticket would silence the new one.)
  if (notificationsSent.has(key)) return;
  notificationsSent.add(key);
  if (window.AndroidBridge && typeof window.AndroidBridge.notify === "function") {
    try {
      window.AndroidBridge.notify(title, message);
    } catch (err) {
      console.warn("Notification failed:", err);
    }
  }
}

function turnIsNear(data) {
  // Time-based when the server sends an estimate; otherwise fall back to
  // "2 places or fewer from the front".
  if (typeof data.eta_seconds === "number" && typeof data.notify_minutes === "number") {
    return data.eta_seconds <= data.notify_minutes * 60;
  }
  return Boolean(data.position) && data.position <= 2;
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

      if (data.status === "waiting" && turnIsNear(data)) {
        let msg;
        if (data.position === 1) {
          msg = "You're next in line!";
        } else if (typeof data.eta_seconds === "number") {
          msg = "Your turn is in about " + Math.max(1, Math.round(data.eta_seconds / 60)) +
                " min. Please be ready.";
        } else {
          msg = "You're #" + data.position + " in line. Almost your turn!";
        }
        notifyApp(statusUrl + ":soon", "Your turn is coming up", msg);
      }
    } catch (err) {
      // Network hiccup - just try again on the next interval.
      console.warn("Status poll failed:", err);
    }
  };
  setInterval(poll, 5000);
}
