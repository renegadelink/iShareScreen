document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".quit-app").forEach((button) =>
    button.addEventListener("click", async () => {
      try {
        const response = await fetch("/quit", { method: "POST" });
        if (!response.ok) throw new Error("Quit failed");
        document.body.replaceChildren();
        const main = document.createElement("main");
        main.className = "waiting";
        main.textContent = "iShareScreen is closed. You can close this tab.";
        document.body.append(main);
      } catch (_) {
        button.textContent = "Unable to quit. Try again";
      }
    }),
  );
  const toggle = document.querySelector(".password-toggle");
  toggle?.addEventListener("click", () => {
    const input = document.getElementById("password");
    const visible = input.type === "password";
    input.type = visible ? "text" : "password";
    toggle.setAttribute(
      "aria-label",
      visible ? "Hide password" : "Show password",
    );
    toggle.setAttribute("aria-pressed", String(visible));
  });
  document.querySelectorAll(".effects-toggle").forEach((button) => {
    const update = () => {
      const reduced = document.documentElement.dataset.effects === "reduced";
      button.setAttribute("aria-pressed", String(reduced));
      button.textContent = reduced ? "Restore effects" : "Reduce effects";
    };
    update();
    button.addEventListener("click", () => {
      const reduced = document.documentElement.dataset.effects !== "reduced";
      window.LiquidGlass?.setEffects(reduced ? "reduced" : "full");
      window.LiquidGlass?.setMotion(reduced ? "reduced" : "system");
      update();
    });
  });
  document.querySelectorAll(".display-options").forEach((details) => {
    details.addEventListener("toggle", () => {
      if (details.open)
        window.LiquidGlass?.reveal(
          [details.querySelector(".options-content")],
          { effect: "fade" },
        );
    });
  });
});
