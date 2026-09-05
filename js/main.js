document.getElementById("year").textContent = new Date().getFullYear();

(function () {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;

  const root = document.documentElement;
  const KEY = "theme";
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  function isDark() {
    const attr = root.getAttribute("data-theme");
    if (attr === "dark") return true;
    if (attr === "light") return false;
    return media.matches;
  }

  function updateIcon() {
    btn.classList.toggle("is-dark", isDark());
  }

  btn.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem(KEY, next);
    } catch (e) {}
    updateIcon();
  });

  media.addEventListener("change", () => {
    try {
      if (!localStorage.getItem(KEY)) updateIcon();
    } catch (e) {
      updateIcon();
    }
  });

  updateIcon();
})();

(function () {
  const el = document.getElementById("last-updated");
  if (!el) return;

  const CACHE_KEY = "site-last-updated";
  const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

  function render(isoDate) {
    el.textContent = new Date(isoDate).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  }

  function hide() {
    el.closest("p").style.display = "none";
  }

  let cached = null;
  try {
    cached = JSON.parse(localStorage.getItem(CACHE_KEY));
  } catch (e) {
    cached = null;
  }

  if (cached && Date.now() - cached.fetchedAt < CACHE_TTL_MS) {
    render(cached.date);
    return;
  }

  fetch("https://api.github.com/repos/xuzhongwm/xuzhongwang.github.io/commits?per_page=1")
    .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
    .then((commits) => {
      const date = commits[0].commit.committer.date;
      localStorage.setItem(CACHE_KEY, JSON.stringify({ date, fetchedAt: Date.now() }));
      render(date);
    })
    .catch(() => {
      if (cached) {
        render(cached.date);
      } else {
        hide();
      }
    });
})();
