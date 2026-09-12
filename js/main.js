/* Footer year ---------------------------------------------------------- */

(function () {
  const el = document.getElementById("year");
  if (el) el.textContent = new Date().getFullYear();
})();

/* Theme toggle --------------------------------------------------------- */

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

/* Scroll progress + sticky header state -------------------------------- */

(function () {
  const bar = document.getElementById("progress-bar");
  const header = document.getElementById("site-header");
  if (!bar && !header) return;

  let ticking = false;

  function update() {
    ticking = false;
    const scrolled = window.scrollY;

    if (bar) {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = max > 0 ? Math.min(scrolled / max, 1) : 0;
      bar.style.transform = "scaleX(" + ratio + ")";
    }

    if (header) {
      header.classList.toggle("is-stuck", scrolled > 8);
    }
  }

  window.addEventListener(
    "scroll",
    () => {
      if (!ticking) {
        ticking = true;
        window.requestAnimationFrame(update);
      }
    },
    { passive: true }
  );

  window.addEventListener("resize", update, { passive: true });
  update();
})();

/* Reveal on scroll ----------------------------------------------------- */

(function () {
  const targets = document.querySelectorAll(".reveal, .reveal-stagger");
  if (!targets.length) return;

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduced || !("IntersectionObserver" in window)) {
    targets.forEach((el) => el.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -12% 0px", threshold: 0.1 }
  );

  targets.forEach((el) => observer.observe(el));

  // Safety net: nothing may stay invisible if the observer never fires.
  window.setTimeout(() => {
    targets.forEach((el) => el.classList.add("is-visible"));
  }, 3000);
})();

/* Section scrollspy ---------------------------------------------------- */

(function () {
  const links = Array.from(document.querySelectorAll(".section-nav a"));
  if (!links.length || !("IntersectionObserver" in window)) return;

  const map = new Map();
  links.forEach((link) => {
    const id = link.getAttribute("href");
    if (!id || id.charAt(0) !== "#") return;
    const section = document.querySelector(id);
    if (section) map.set(section, link);
  });

  if (!map.size) return;

  function setActive(link) {
    links.forEach((l) => l.classList.toggle("is-active", l === link));
  }

  const visible = new Set();

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          visible.add(entry.target);
        } else {
          visible.delete(entry.target);
        }
      });

      if (!visible.size) {
        setActive(null);
        return;
      }

      // Highest section currently on screen wins.
      const top = Array.from(visible).sort(
        (a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top
      )[0];
      setActive(map.get(top));
    },
    { rootMargin: "-30% 0px -55% 0px", threshold: 0 }
  );

  map.forEach((_link, section) => observer.observe(section));
})();

/* Last updated (from GitHub) ------------------------------------------- */

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
      try {
        localStorage.setItem(CACHE_KEY, JSON.stringify({ date, fetchedAt: Date.now() }));
      } catch (e) {}
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
